import logging
import numpy as np
import torch.nn.functional as F
from dateutil.relativedelta import relativedelta
import datetime
import xarray as xr

# import dask
# import dask.array as da
# from dask.diagnostics import ProgressBar


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger()


def generate_date_list(start_time, end_time):
    """
    Generate a list of dates.

    Args:
        start_time: Start date in the format YYYYMMDD.
        end_time: End date in the format YYYYMMDD.

    Returns:
        List of dates in the format of list[YYYYMMDD].
    """
    start = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    return [
        int(dt.strftime("%Y%m%d"))
        for dt in [start + relativedelta(days=i) for i in range((end - start).days + 1)]
    ]


# def write_processed_data_to_netcdf(data_paths_file, start_time, end_time, out_path):
#     """
#     Write processed data to a .nc file for future use.

#     Args:
#         data_paths_file: Path to a text file containing the file paths.
#         start_time: Start time for data filtering.
#         end_time: End time for data filtering.
#         out_path: Path of the output .nc file.
#     """
#     data_paths = np.genfromtxt(data_paths_file, dtype=str)
#     times = np.array(
#         [int(path.split("/")[-1].split("_")[5][0:8]) for path in data_paths]
#     )
#     indices = np.where((times >= start_time) & (times <= end_time))[0]
#     data_paths = data_paths[indices]
#     times = times[indices]

#     data_arrays = [
#         create_xarray_data_array(path, time) for path, time in zip(data_paths, times)
#     ]

#     dataset = xr.concat(data_arrays, dim="time").to_dataset(name="imgs")
#     with ProgressBar():
#         dataset.to_netcdf(
#             out_path,
#             encoding={"imgs": {"dtype": "f4", "zlib": True, "complevel": 4}},
#             compute=True,
#             engine="h5netcdf",
#         )


# def create_xarray_data_array(data_path, time):
#     """
#     Load data from a path and return a processed xarray DataArray.

#     Args:
#         data_path: Path to the data file.
#         time: Time associated with the data.

#     Returns:
#         xarray DataArray with added time dimension.
#     """
#     data = da.from_delayed(
#         dask.delayed(load_and_process_data)(data_path),
#         shape=(432, 432),
#         dtype="float32",
#     )
#     x_coords = da.arange(data.shape[1])  # x coordinates
#     y_coords = da.arange(data.shape[0])  # y coordinates

#     return xr.DataArray(
#         data, dims=["yc", "xc"], coords={"yc": y_coords, "xc": x_coords}
#     ).expand_dims({"time": [time]})


def load_and_process_data(data_path):
    """
    Load and process data from a given path.

    Args:
        data_path: Path to the data file.

    Returns:
        Processed data.
    """
    data = xr.open_dataset(data_path)
    return process_sea_ice_data(data)


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


def unfold_stack_over_channel(image, patch_size):
    """
    将图像切分成多个小块并沿着通道堆叠
    Args:
        image (N, *, C, H, W): 最后两个维度必须是空间维度
        patch_size(k_h,k_w): 长度为2的元组，就是configs.patch_size
    Returns:
        output (N, *, C*k_h*k_w, H/k_h, W/k_w)
    """
    n_dims = len(image.shape)
    assert n_dims == 4 or n_dims == 5
    if patch_size[0] == 1 and patch_size[1] == 1:
        return image

    patch = image.unfold(-2, size=patch_size[0], step=patch_size[0])
    patch = patch.unfold(-2, size=patch_size[1], step=patch_size[1]).flatten(-2)
    if n_dims == 4:  # (N, C, H, W)
        patch = patch.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dims == 5:  # (N, T, C, H, W)
        patch = patch.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert patch.shape[-3] == image.shape[-3] * patch_size[0] * patch_size[1]
    return patch


def fold_tensor(tensor, output_size, patch_size):
    """
    用non-overlapping的块重建图像
    Args:
        tensor shape (N, *, C*k_h*k_w, h, w)
        output_size: (H, W)，要重建的原始图像的大小
        patch_size: (k_h, k_w)
        请注意，对于non-overlapping的滑动窗口，通常stride等于patch_size
    Returns:
        output (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    n_dims = len(tensor.shape)
    assert n_dims == 4 or n_dims == 5

    if patch_size[0] == 1 and patch_size[1] == 1:
        return tensor

    # 展平输入
    flattened = tensor.flatten(0, 1) if n_dims == 5 else tensor

    # 使用 F.fold 函数进行重建
    folded = F.fold(
        flattened.flatten(-2),
        output_size=output_size,
        kernel_size=patch_size,
        stride=patch_size,
    )

    if n_dims == 5:
        folded = folded.view(tensor.size(0), tensor.size(1), *folded.shape[1:])

    return folded


def prepare_input_target_indices(
    time_length,
    input_gap,
    input_length,
    prediction_shift,
    prediction_gap,
    prediction_length,
    sample_gap,
):
    """
    Args:
        input_gap: 两个连续输入帧之间的时间间隔
        input_length: 输入帧的数量
        prediction_shift: 最后一个目标预测的前导时间
        prediction_gap: 两个连续输出帧之间的时间间隔
        prediction_length: 输出帧的数量
        sample_gap: 两个检索样本的起始时间之间的间隔
    Returns:
        input_indices: 指向输入样本位置的索引
        target_indices: 指向目标样本位置的索引
    """
    assert prediction_shift >= prediction_length
    input_span = input_gap * (input_length - 1) + 1
    input_index = np.arange(0, input_span, input_gap)
    target_index = (
        np.arange(0, prediction_shift, prediction_gap) + input_span + prediction_gap - 1
    )
    indices = np.concatenate([input_index, target_index]).reshape(
        1, input_length + prediction_length
    )
    max_sample_count = time_length - (input_span + prediction_shift - 1)
    indices = indices + np.arange(max_sample_count)[:, np.newaxis] @ np.ones(
        (1, input_length + prediction_length), dtype=int
    )
    input_indices = indices[::sample_gap, :input_length]
    target_indices = indices[::sample_gap, input_length:]
    assert len(input_indices) == len(target_indices)
    return input_indices, target_indices
