"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-09-26 18:28:47
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-09-27 13:12:52
FilePath: /root/osi-450-a/utils/metrics.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import torch

Max_SIE = 25889.0


def mse_func(pred, true, mask):
    mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    return mse.mean()


def rmse_func(pred, true, mask):
    # mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]) / torch.sum(mask)
    # rmse = torch.sqrt(mse)
    # print(rmse)

    mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    rmse = torch.sqrt(mse)
    return rmse.mean()


def mae_func(pred, true, mask):
    # mae = torch.abs(pred - true) * mask
    # mae = torch.sum(mae, dim=[2, 3, 4]) / torch.sum(mask)
    # print(mae)

    mae = torch.abs(pred - true) * mask
    mae = torch.sum(mae, dim=[2, 3, 4]).mean(dim=1) / torch.sum(mask)
    return mae.mean()


def nse_func(pred, true, mask):
    # squared_error = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4])
    # mean_observation = torch.sum(true * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
    #     mask
    # )
    # mean_observation = (
    #     mean_observation.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # )
    # squared_deviation = torch.sum((true - mean_observation) ** 2 * mask, dim=[2, 3, 4])
    # nse = 1 - squared_error / squared_deviation
    # print(nse)

    squared_error = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1)
    mean_observation = torch.sum(true * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    mean_observation = (
        mean_observation.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )
    squared_deviation = torch.sum(
        (true - mean_observation) ** 2 * mask, dim=[2, 3, 4]
    ).mean(dim=1)
    nse = 1 - squared_error / squared_deviation
    return nse.mean()


def PSNR_func(pred, true, mask):
    mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    PSNR = 10 * torch.log10(1 * 1 / mse.mean())
    return PSNR


def BACC_func(pred, true, mask):
    # 使用布尔索引将大于等于0.15的位置设置为1，其他地方设置为0
    pred_binary = (pred >= 0.15).float()
    true_binary = (true >= 0.15).float()

    # a = torch.sum(torch.abs(pred_binary - true_binary) * mask, dim=[2, 3, 4])
    # BACC = 1 - a / Max_SIE
    # print(BACC)

    # 计算 IIEE
    IIEE = torch.sum(torch.abs(pred_binary - true_binary) * mask, dim=[2, 3, 4]).mean(
        dim=1
    )

    # 计算 BACC
    BACC = 1 - IIEE.mean() / Max_SIE
    return BACC
