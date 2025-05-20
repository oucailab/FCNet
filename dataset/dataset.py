"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-09-27 08:46:48
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-09-27 08:47:08
FilePath: /root/osi-450-a/dataset/dataset.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import numpy as np
from torch.utils.data import Dataset
import datetime
import xarray as xr
from utils.tools import (
    generate_date_list,
    prepare_input_target_indices,
    process_sea_ice_data,
)


class SIC_dataset(Dataset):
    def __init__(
        self,
        data_paths,
        start_time,
        end_time,
        input_gap,
        input_length,
        pred_shift,
        pred_gap,
        pred_length,
        samples_gap,
    ):
        super().__init__()
        self.time_list = generate_date_list(start_time, end_time)

        # 计算索引列表
        self.input_indices, self.target_indices = prepare_input_target_indices(
            len(self.time_list),
            input_gap,
            input_length,
            pred_shift,
            pred_gap,
            pred_length,
            samples_gap,
        )

        self.paths = np.genfromtxt(data_paths, dtype=str)

        # 获取 time 维度的开始
        self.time_coords_begin = self.paths[0].split("_")[-1][0:8]

        self.offset = (
            datetime.datetime.strptime(str(start_time), "%Y%m%d")
            - datetime.datetime.strptime(str(self.time_coords_begin), "%Y%m%d")
        ).days

    def __len__(self):
        return len(self.input_indices)

    def __getitem__(self, index):
        # 获取输入和目标时间索引列表，[0 1 2 3 4 5 6]这种的
        input_index = self.input_indices[index]
        target_index = self.target_indices[index]

        input_start_index = input_index[0] + self.offset
        input_end_index = input_index[-1] + self.offset

        target_start_index = target_index[0] + self.offset
        target_end_index = target_index[-1] + self.offset

        inputs = np.array(
            [
                process_sea_ice_data(xr.open_dataset(path))
                for path in self.paths[input_start_index : input_end_index + 1]
            ]
        )[:, None]

        targets = np.array(
            [
                process_sea_ice_data(xr.open_dataset(path))
                for path in self.paths[target_start_index : target_end_index + 1]
            ]
        )[:, None]

        return (
            inputs,
            targets,
        )

    def get_inputs(self):
        input_index = self.input_indices[0]
        input_start_index = input_index[0] + self.offset
        input_end_index = input_index[-1] + self.offset

        inputs = np.array(
            [
                process_sea_ice_data(xr.open_dataset(path))
                for path in self.paths[input_start_index : input_end_index + 1]
            ]
        )

        return inputs

    def get_targets(self):
        target_index = self.target_indices[0]
        target_start_index = target_index[0] + self.offset
        target_end_index = target_index[-1] + self.offset

        targets = np.array(
            [
                process_sea_ice_data(xr.open_dataset(path))
                for path in self.paths[target_start_index : target_end_index + 1]
            ]
        )

        return targets

    def get_times(self):
        return self.time_list
