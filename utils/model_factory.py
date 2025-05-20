"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2023-11-23 12:45:21
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2023-11-25 00:23:25
FilePath: /arctic_sic_prediction/model_factory.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""

import torch.nn as nn
from utils.tools import unfold_stack_over_channel, fold_tensor
from models.ConvLSTM import ConvLSTM
from models.PredRNN import PredRNN
from models.SimVP import SimVP
from models.TAU import TAU
from models.Swin_Transformer import Swin_Transformer
from models.FCNet import FCNet
from models.SICFN import SICFN
from config import configs


class IceNet(nn.Module):
    def __init__(self):
        super().__init__()

        if configs.model == "ConvLSTM":
            self.net = ConvLSTM(
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hidden_dim,
                configs.output_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.kernel_size,
            )
        elif configs.model == "PredRNN":
            self.net = PredRNN(
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hidden_dim,
                configs.output_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.kernel_size,
                configs.layer_norm,
            )
        elif configs.model == "SimVP":
            self.net = SimVP(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
            )
        elif configs.model == "TAU":
            self.net = TAU(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
            )
        elif configs.model == "Swin_Transformer":
            self.net = Swin_Transformer(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
            )
        elif configs.model == "FCNet":
            self.net = FCNet(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.patch_embed_size,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
                configs.dropcls,
            )
        elif configs.model == "SICFN":
            self.net = SICFN(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.patch_embed_size,
                configs.fno_blocks,
                configs.fno_bias,
                configs.fno_softshrink,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
                configs.dropcls,
            )
        else:
            raise ValueError("错误的网络名称，不存在%s这个网络" % configs.model)

    def forward(self, inputs, targets):
        B, T, C, H, W = inputs.shape
        padding = abs(H - W) // 2  # 两侧填充的数量
        if configs.model in ["Swin_Transformer", "FCNet"]:
            if H > W:
                # 指定在 W 轴方向左侧和右侧各填充多少个零
                inputs = nn.functional.pad(inputs, (padding, padding, 0, 0), value=0)
                targets = nn.functional.pad(targets, (padding, padding, 0, 0), value=0)
            elif H < W:
                # 指定在 H 轴方向上侧和下侧各填充多少个零
                inputs = nn.functional.pad(inputs, (0, 0, padding, padding), value=0)
                targets = nn.functional.pad(targets, (0, 0, padding, padding), value=0)

        pred, loss = self.net(
            unfold_stack_over_channel(inputs, configs.patch_size),
            unfold_stack_over_channel(targets, configs.patch_size),
        )

        pred = fold_tensor(pred, configs.img_size, configs.patch_size)

        return pred, loss
