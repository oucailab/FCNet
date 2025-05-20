"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-10-29 11:33:23
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-11-07 10:51:29
FilePath: /root/OSI-SAF/view.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import matplotlib.pyplot as plt
import torch
from train import Trainer
import torchvision


def show_single_filter(gf):
    """
    显示单个滤波器
    :param gf: 全局滤波器 (h, w // 2 + 1, 2)
    :return: 滤波器的绝对值
    """
    h = gf.size(0)
    w = (gf.size(1) - 1) * 2
    gf_complex = torch.view_as_complex(gf)
    gf_spatial = torch.fft.irfft2(gf_complex, dim=(0, 1), s=(h, w), norm="ortho")
    gf_complex = torch.fft.fft2(gf_spatial, dim=(0, 1), norm="ortho")
    gf_complex = torch.fft.fftshift(gf_complex, dim=(0, 1))
    gf_abs = gf_complex.abs()
    return gf_abs

def visualize_filters(tester):
    """
    Visualize filters with added labels for each layer and channel, centered within rows and columns.
    :param tester: Trainer instance
    :return: Visualization result
    """
    global_filters = []
    for i_layer in range(len(tester.network.net.AFFB_module.blocks)):
        weight = tester.network.net.AFFB_module.blocks[i_layer].filter.complex_weight
        weight_mean = torch.mean(weight, dim=3)
        for i_channel in range(weight_mean.size(0)):
            filter_abs = show_single_filter(weight_mean[i_channel, :, :])
            global_filters.append(filter_abs[None])
    global_filters = torch.stack(global_filters)

    # Define figure and plot settings
    plt.figure(figsize=(16, 30))
    viz = torchvision.utils.make_grid(
        global_filters,
        nrow=weight_mean.size(0),
        padding=1,
        pad_value=0,
    )
    viz = viz.permute(1, 2, 0)[:, :, 0].cpu().numpy()
    
    # Display the image grid
    plt.imshow(viz, cmap="YlGnBu")
    plt.axis("off")
    
    # Calculate the width and height of each cell in the grid
    num_blocks = len(tester.network.net.AFFB_module.blocks)
    num_channels = weight_mean.size(0)
    cell_height = viz.shape[0] / num_blocks
    cell_width = viz.shape[1] / num_channels
    
    # Add labels for each block (i_layer) and channel (i_channel), centered
    for i_layer in range(num_blocks):
        plt.text(-cell_width * 0.2, i_layer * cell_height + cell_height / 2,
                 f"block={i_layer}", ha="right", va="center", fontsize=10, color="black")

    for i_channel in range(num_channels):
        plt.text(i_channel * cell_width + cell_width / 2, -cell_height * 0.2,
                 f"t={i_channel+1}", ha="center", va="bottom", fontsize=10, color="black")

    # Save and return the visualization
    plt.savefig("frequency_domain.png", bbox_inches="tight", pad_inches=0)
    

# 初始化 Trainer 并加载模型权重
tester = Trainer()
tester.network.load_state_dict(
    torch.load(f"checkpoints/checkpoint_FCNet_14.pt", weights_only=True)["net"],
    strict=False,
)
# 调用可视化函数
visualize_filters(tester)
