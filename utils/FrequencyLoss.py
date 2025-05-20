"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-09-23 22:07:54
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-09-23 22:08:11
FilePath: /root/osi-450-a/utils/FrequencyLoss.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import torch
import torch.nn as nn
import torch.fft


class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def loss_formulation(self, predicted_values, target_values):
        difference = (predicted_values - target_values) ** 2
        sqrt_difference = torch.sqrt(difference[..., 0] + difference[..., 1])
        sqrt_difference = torch.log(sqrt_difference + 1.0)
        max_value = sqrt_difference.max(-1).values.max(-1).values[:, :, None, None]
        normalized_matrix = sqrt_difference / max_value
        normalized_matrix[torch.isnan(normalized_matrix)] = 0.0
        normalized_matrix = torch.clamp(normalized_matrix, min=0.0, max=1.0)
        weight_matrix = normalized_matrix.clone().detach()
        frequency_distance = (predicted_values - target_values) ** 2
        frequency_distance = frequency_distance[..., 0] + frequency_distance[..., 1]
        weighted_loss = weight_matrix * frequency_distance
        return torch.mean(weighted_loss)

    def forward(self, pred, target):
        B, T, C, H, W = pred.shape

        pred = pred.view(B * T, C, H, W).to(torch.float32)
        target = target.view(B * T, C, H, W).to(torch.float32)

        # 对预测值和目标值应用二维实数快速傅里叶变换，并堆叠实部和虚部
        pred = torch.fft.rfft2(pred, norm="ortho")
        pred = torch.stack([pred.real, pred.imag], dim=-1)
        target = torch.fft.rfft2(target, norm="ortho")
        target = torch.stack([target.real, target.imag], dim=-1)

        # 调用loss_formulation方法计算损失并返回
        return self.loss_formulation(pred, target)
