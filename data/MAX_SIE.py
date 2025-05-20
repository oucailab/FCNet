"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-09-26 18:48:04
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-09-27 15:37:30
FilePath: /root/OSI-SAF/data/MAX_SIE.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import numpy as np
import xarray as xr
from tqdm import tqdm


def process_sea_ice_data(data):
    """
    0 - 100 Sea ice concentration %
    -32767 Land

    处理数据，包括归一化、处理缺失数据、陆地屏蔽等
    Args:
        data: 输入的海冰数据
    Returns:
        ice_conc: 处理后的海冰密集度数据
    """
    ice_conc = np.array(data["ice_conc"][:][0])
    ice_conc = np.nan_to_num(ice_conc, nan=0)

    # 处理陆地
    ice_conc[ice_conc == -32767] = 0

    # 归一化至[0-1]
    ice_conc = ice_conc / 100

    # 确保没有超出范围的值
    assert not np.any(ice_conc > 1)

    return ice_conc


paths = np.genfromtxt("data_path.txt", dtype=str)

MAX_SIE = 0

loop = tqdm(paths, total=len(paths), leave=True)
for path in loop:
    ice_conc = xr.open_dataset(path)
    ice_conc = process_sea_ice_data(ice_conc)
    ice_conc[ice_conc > 0.15] = 1
    ice_conc[ice_conc <= 0.15] = 0
    if np.sum(ice_conc) > MAX_SIE:
        MAX_SIE = np.sum(ice_conc)
    loop.set_postfix(
        MAX_SIE=MAX_SIE,
    )

print(MAX_SIE)
