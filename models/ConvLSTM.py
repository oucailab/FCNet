"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2023-11-20 13:48:52
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2023-11-20 15:39:59
FilePath: /arctic_sic_prediction/models/ConvLSTM.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""

import torch
import torch.nn as nn
from layers.ConvLSTM.ConvLSTMCell import ConvLSTMCell


class ConvLSTM(nn.Module):
    """
    输入参数如下：
    input_dim:输入张量对应的通道数，对于彩图为3，灰图为1
    hidden_dim:h,c两个状态张量的节点数，当多层的时候，可以是一个列表，表示每一层中状态张量的节点数
    output_dim:输出张量对应的通道数，对于彩图为3，灰图为1
    kernel_size:卷积核的尺寸，默认所有层的卷积核尺寸都是一样的，也可以设定不同的lstm层的卷积核尺寸不同
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        kernel_size,
    ):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(
            hidden_dim
        )  # num_layers: LSTM的层数，需要与len(hidden_dim)相等

        cell_list = []  # 每个ConvLSTMCell会存入该列表中
        for i in range(self.num_layers):  # 当LSTM为多层，每一层的单元输入
            cur_input_dim = (
                self.input_dim if i == 0 else self.hidden_dim[i - 1]
            )  # 一层的时候，单元输入就为input_dim，多层的时候，单元输入为对应的，前一层的隐藏层节点情况

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=kernel_size,
                )
            )

        self.cell_list = nn.ModuleList(
            cell_list
        )  # 把定义的多个LSTM层串联成网络模型，ModuleList中模型可以自动更新参数
        self.conv_output = nn.Conv2d(
            self.hidden_dim[-1], output_dim, kernel_size=1, bias=False
        )

        self.criterion = nn.HuberLoss()

    def forward(self, input_x, targets):
        """
        Args:
            input_x: 5D张量，[b, l, c, h, w]
            land_mask: 陆地掩码
        Returns:
            output: 5D张量，[b, 1, c, h, w]
        """
        input_x = input_x.contiguous()
        targets = targets.contiguous()
        device = input_x.device

        assert len(input_x.shape) == 5
        seq_len = input_x.size(
            1
        )  # 这里的seq_len就是config里面的那个input_length，时间线长度

        batch_size = input_x.size(0)
        height = input_x.size(3)
        width = input_x.size(4)

        h_t = []
        c_t = []

        next_frames = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                batch_size, self.hidden_dim[i], height, width, device=device
            )
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(seq_len):  # 序列里逐个计算，然后更新
            input_ = input_x[:, t]

            h_t[0], c_t[0] = self.cell_list[0](input_, h_t[0], c_t[0])

            for i in range(1, self.num_layers):  # 逐层计算
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_output(h_t[-1])
            x_gen = torch.clamp(x_gen, 0, 1)
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1).contiguous()
        loss = self.criterion(next_frames, targets)

        return next_frames, loss
