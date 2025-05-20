"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2023-11-20 13:48:52
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2023-11-20 15:48:13
FilePath: /arctic_sic_prediction/layers/ConvLSTM/ConvLSTMCell.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""

"""
定义ConvLSTM每一层的、每个时间点的模型单元，及其计算
"""
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    单元输入参数如下：
    input_dim: 输入张量对应的通道数，对于彩图为3，灰图为1
    hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数，应该可以按计算需要“随意”设置
    kernel_size: (int, int)，卷积核，并且卷积核通常都需要为奇数
    """

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        pad = kernel_size[0] // 2, kernel_size[1] // 2  #  //表示除法后取整数，为使池化后图片依然对称，故这样操作
        """
        nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding,dilation=1,groups=1,bias)
        二维的卷积神经网络
        """
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,  # 每个单元的输入为上个单元的h和这个单元的x，
            # 输入门，遗忘门，输出门，激活门是LSTM的体现，
            # 每个门的维度和隐藏层维度一样，这样才便于进行+和*的操作
            # 输出了四个门，连接在一起，后面会想办法把门的输出单独分开，只要想要的
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            bias=False,
        )

    def forward(self, inputs, h_cur, c_cur):
        """
        把inputs与状态张量h，沿inputs通道维度(h的节点个数）,串联
        combined:[batch_size,input_dim+hidden_dim,height,weight]
        """
        combined = torch.cat([inputs, h_cur], dim=1)

        """ 
        将conv的输出combined_conv([batch_size,output_dim,height,width])
        分成output_dim这个维度去分块，每个块包含hidden_dim个节点信息
        四个块分别对于i,f,o,g四道门，每道门的size为[b,hidden_dim,h,w]
        """
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)  # 激活门

        c_next = f * c_cur + i * g  # 主线，遗忘门选择遗忘的+被激活一次的输入，更新长期记忆
        h_next = o * torch.tanh(c_next)  # 短期记忆，通过主线的激活和输出门后，更新短期记忆（即每个单元的输出）
        return h_next, c_next
