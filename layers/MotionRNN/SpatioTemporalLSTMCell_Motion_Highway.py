"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2023-11-24 18:46:24
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2023-11-24 18:47:35
FilePath: /root/arctic_sic_prediction/layers/MotionRNN/SpatioTemporalLSTMCell_Motion_Highway.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""
import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    """
    单元输入参数如下：
    input_dim: 输入张量对应的通道数，对于彩图为3，灰图为1
    hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数，应该可以按计算需要“随意”设置
    kernel_size: (int, int)，卷积核，并且卷积核通常都需要为奇数
    """

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = (
            kernel_size[0] // 2,
            kernel_size[1] // 2,
        )  #  //表示除法后取整数，为使池化后图片依然对称，故这样操作
        self._forget_bias = 1.0

        """
        相应符号可对应参照论文
        conv_h for gt, it, ft
        conv_m for gt', it', ft'
        conv_o for ot
        conv_last for Ht
        """
        self.conv_x = nn.Sequential(
            nn.Conv2d(
                self.input_dim,
                self.hidden_dim * 7,
                kernel_size=kernel_size,
                padding=self.padding,
            ),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim * 4,
                kernel_size=kernel_size,
                padding=self.padding,
            ),
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim * 3,
                kernel_size=kernel_size,
                padding=self.padding,
            ),
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim * 2,
                self.hidden_dim,
                kernel_size=kernel_size,
                padding=self.padding,
            ),
        )
        self.conv_last = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=1,
        )

    def forward(self, x_t, h_t, c_t, m_t, motion_highway):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.hidden_dim, dim=1
        )
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.hidden_dim, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.hidden_dim, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        m_new_new = self.conv_last(mem)

        # Motion Highway
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(m_new_new) + (1 - o_t) * motion_highway
        motion_highway = h_new

        return h_new, c_new, m_new, motion_highway
