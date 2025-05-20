"""
Author: 爱吃菠萝 zhangjia_liang@foxmail.com
Date: 2023-10-31 20:26:14
LastEditors: 爱吃菠萝 zhangjia_liang@foxmail.com
LastEditTime: 2023-10-31 20:26:23
FilePath: /arctic_-sic_prediction/layers/SpatioTemporalLSTMCell_Motion_Highway.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import torch
import torch.nn as nn
from layers.MotionRNN.SpatioTemporalLSTMCell_Motion_Highway import (
    SpatioTemporalLSTMCell,
)
from layers.MotionRNN.MotionGRU import MotionGRU


class MotionRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_ch = configs.img_channel * (configs.patch_size**2)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.neighbour = 3
        self.motion_hidden = 2 * self.neighbour * self.neighbour
        self.MSE_criterion = nn.MSELoss().to(self.configs.device)

        cell_list = []
        for i in range(num_layers):
            in_channel = self.patch_ch if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(
                    in_channel,
                    num_hidden[i],
                    self.patch_height,
                    self.patch_width,
                    configs.filter_size,
                    configs.stride,
                    configs.layer_norm,
                ),
            )
        enc_list = []
        for i in range(num_layers - 1):
            enc_list.append(
                nn.Conv2d(
                    num_hidden[i],
                    num_hidden[i] // 4,
                    kernel_size=configs.filter_size,
                    stride=2,
                    padding=configs.filter_size // 2,
                ),
            )
        motion_list = []
        for i in range(num_layers - 1):
            motion_list.append(
                MotionGRU(num_hidden[i] // 4, self.motion_hidden, self.neighbour)
            )
        dec_list = []
        for i in range(num_layers - 1):
            dec_list.append(
                nn.ConvTranspose2d(
                    num_hidden[i] // 4,
                    num_hidden[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
        gate_list = []
        for i in range(num_layers - 1):
            gate_list.append(
                nn.Conv2d(
                    num_hidden[i] * 2,
                    num_hidden[i],
                    kernel_size=configs.filter_size,
                    stride=1,
                    padding=configs.filter_size // 2,
                ),
            )
        self.gate_list = nn.ModuleList(gate_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.motion_list = nn.ModuleList(motion_list)
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1],
            self.patch_ch,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv_first_v = nn.Conv2d(
            self.patch_ch, num_hidden[0], 1, stride=1, padding=0, bias=False
        )

    def forward(self, all_frames, mask_true):
        frames = all_frames.contiguous()
        mask_true = mask_true.contiguous()
        next_frames = []
        h_t = []
        c_t = []
        h_t_conv = []
        h_t_conv_offset = []
        mean = []

        for i in range(self.num_layers):
            zeros = torch.empty(
                [
                    self.configs.batch_size,
                    self.num_hidden[i],
                    self.patch_height,
                    self.patch_width,
                ]
            ).to(self.configs.device)
            nn.init.xavier_normal_(zeros)
            h_t.append(zeros)
            c_t.append(zeros)

        for i in range(self.num_layers - 1):
            zeros = torch.empty(
                [
                    self.configs.batch_size,
                    self.num_hidden[i] // 4,
                    self.patch_height // 2,
                    self.patch_width // 2,
                ]
            ).to(self.configs.device)
            nn.init.xavier_normal_(zeros)
            h_t_conv.append(zeros)
            zeros = torch.empty(
                [
                    self.configs.batch_size,
                    self.motion_hidden,
                    self.patch_height // 2,
                    self.patch_width // 2,
                ]
            ).to(self.configs.device)
            nn.init.xavier_normal_(zeros)
            h_t_conv_offset.append(zeros)
            mean.append(zeros)

        mem = torch.empty(
            [
                self.configs.batch_size,
                self.num_hidden[0],
                self.patch_height,
                self.patch_width,
            ]
        ).to(self.configs.device)
        motion_highway = torch.empty(
            [
                self.configs.batch_size,
                self.num_hidden[0],
                self.patch_height,
                self.patch_width,
            ]
        ).to(self.configs.device)
        nn.init.xavier_normal_(mem)
        nn.init.xavier_normal_(motion_highway)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = (
                    mask_true[:, t - self.configs.input_length] * frames[:, t]
                    + (1 - mask_true[:, t - self.configs.input_length]) * x_gen
                )

            motion_highway = self.conv_first_v(net)
            h_t[0], c_t[0], mem, motion_highway = self.cell_list[0](
                net, h_t[0], c_t[0], mem, motion_highway
            )
            net = self.enc_list[0](h_t[0])
            h_t_conv[0], h_t_conv_offset[0], mean[0] = self.motion_list[0](
                net, h_t_conv_offset[0], mean[0]
            )
            h_t_tmp = self.dec_list[0](h_t_conv[0])
            o_t = torch.sigmoid(self.gate_list[0](torch.cat([h_t_tmp, h_t[0]], dim=1)))
            h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

            for i in range(1, self.num_layers - 1):
                h_t[i], c_t[i], mem, motion_highway = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], mem, motion_highway
                )
                net = self.enc_list[i](h_t[i])
                h_t_conv[i], h_t_conv_offset[i], mean[i] = self.motion_list[i](
                    net, h_t_conv_offset[i], mean[i]
                )
                h_t_tmp = self.dec_list[i](h_t_conv[i])
                o_t = torch.sigmoid(
                    self.gate_list[i](torch.cat([h_t_tmp, h_t[i]], dim=1))
                )
                h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]

            (
                h_t[self.num_layers - 1],
                c_t[self.num_layers - 1],
                mem,
                motion_highway,
            ) = self.cell_list[self.num_layers - 1](
                h_t[self.num_layers - 2],
                h_t[self.num_layers - 1],
                c_t[self.num_layers - 1],
                mem,
                motion_highway,
            )
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = (
            torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        )
        loss = self.MSE_criterion(next_frames, all_frames[:, 1:])
        return next_frames, loss
