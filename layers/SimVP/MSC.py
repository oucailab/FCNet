"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2023-12-01 18:58:33
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2023-12-11 12:46:29
FilePath: /arctic_sic_prediction/layers/SimVP/Inception.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""
from torch import nn


class GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        groups=1,
        act_norm=False,
        act_inplace=True,
    ):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Multi_Scale_Conv_Block(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 9], groups=4):
        super(Multi_Scale_Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in incep_ker:
            layers.append(
                GroupConv2d(
                    C_hid,
                    C_out,
                    kernel_size=ker,
                    stride=1,
                    padding=ker // 2,
                    groups=groups,
                    act_norm=True,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y
