"""
Author: 爱吃菠萝 zhangjia_liang@foxmail.com
Date: 2023-11-07 14:47:21
LastEditors: 爱吃菠萝 zhangjia_liang@foxmail.com
LastEditTime: 2023-11-07 14:47:45
FilePath: /arctic_sic_prediction/layers/PredRNN/PredRNNCell.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import torch
import torch.nn as nn


class PredRNNv2Cell(nn.Module):
    """
    单元输入参数如下：
    input_dim: 输入张量对应的通道数，对于彩图为3，灰图为1
    hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数，应该可以按计算需要“随意”设置
    kernel_size: (int, int)，卷积核，并且卷积核通常都需要为奇数
    """

    def __init__(self, input_dim, hidden_dim, height, width, kernel_size, layer_norm):
        super(PredRNNv2Cell, self).__init__()

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
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    self.input_dim,
                    self.hidden_dim * 7,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([hidden_dim * 7, height, width]),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(
                    self.hidden_dim,
                    self.hidden_dim * 4,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([hidden_dim * 4, height, width]),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(
                    self.hidden_dim,
                    self.hidden_dim * 3,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([hidden_dim * 3, height, width]),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    bias=False,
                ),
                nn.LayerNorm([hidden_dim, height, width]),
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    self.input_dim,
                    self.hidden_dim * 7,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    bias=False,
                ),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(
                    self.hidden_dim,
                    self.hidden_dim * 4,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    bias=False,
                ),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(
                    self.hidden_dim,
                    self.hidden_dim * 3,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    bias=False,
                ),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    bias=False,
                ),
            )
        self.conv_last = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=1,
        )

    def forward(self, x_t, h_t, c_t, m_t):
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

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m
