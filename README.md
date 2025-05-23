# 🎉 Frequency-Compensated Network for Daily Arctic Sea Ice Concentration Prediction, IEEE TGRS 2025

<div align=center>
 <img src="https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/wechat_2025-04-23_224612_150.jpg" width="80%" />
</div>

## 📌 **Introduction**

This repository contains the official implementation of our paper:  
📄[*Frequency-Compensated Network for Daily Arctic Sea Ice Concentration Prediction, IEEE TGRS 2025*](https://ieeexplore.ieee.org/document/10976656) 

Frequency-Compensated Network (FCNet) is designed for Arctic SIC prediction. In particular, we design a dual-branch network, including branches for frequency feature extraction and convolutional feature extraction. For frequency feature extraction, we design an adaptive frequency filter block, which integrates trainable layers with Fourier-based filters. By adding frequency features, the FCNet can achieve refined prediction of edges and details. For convolutional feature extraction, we propose a high-frequency enhancement block to separate high and low-frequency information. Moreover, high-frequency features are enhanced via channel-wise attention, and temporal attention unit is employed for low-frequency feature extraction to capture long-range sea ice changes. 

## How to Use

### Environment

Run `conda env create -f environment.yaml` to create the Python environment.
Run `conda activate arctic_sic_prediction` to activate the Python environment.

### Dataset

We recommend following the steps below to download and organize the dataset to avoid any issues.

#### 1. Download and Reorganize Data

The daily SIC data used in this study can be downloaded from OSI SAF: [https://osi-saf.eumetsat.int/products/osi-450-a](https://osi-saf.eumetsat.int/products/osi-450-a) and [https://osi-saf.eumetsat.int/products/osi-430-a](https://osi-saf.eumetsat.int/products/osi-430-a), which also contains a detailed description of the dataset and user guide.

Run the `download_and_organize_data.py` file in the `data` directory to download and reorganize the data.

The reorganized data structure should look like this:

```
├── 1991
├── 1992
├── 1993
......
├── 2019
│   ├── 01
│   │   ├── ice_conc_nh_ease2-250_cdr-v3p0_201901011200.nc
│   │   ├── ice_conc_nh_ease2-250_cdr-v3p0_201901021200.nc
│   │   ├── ice_conc_nh_ease2-250_cdr-v3p0_201901031200.nc
│   │   ├── ice_conc_nh_ease2-250_cdr-v3p0_201901041200.nc
......
├── 2024
```

#### 2. Generate File Containing the Path to All Data Files

```shell
bash gen_data_path.sh
```

The generated `data_path.txt` file will be located in the `data` directory.

The content of `data_path.txt` should look like this:

```
OSI-SAF/1991/01/ice_conc_nh_ease2-250_cdr-v3p0_199101011200.nc
OSI-SAF/1991/01/ice_conc_nh_ease2-250_cdr-v3p0_199101021200.nc
OSI-SAF/1991/01/ice_conc_nh_ease2-250_cdr-v3p0_199101031200.nc
OSI-SAF/1991/01/ice_conc_nh_ease2-250_cdr-v3p0_199101041200.nc
......
```

### Train

Change the relevant parameters and the `.nc` file path in the `config.py` file. Place all the scripts under the same folder and run:

```shell
python train.py
```

The training process will be printed, and you can also choose to direct this information to other logging files.

### Test

Specify the testing period and output directory and run:

```shell
python test.py -st 20160101 -et 20160128
```

Alternatively, perform batch testing with:

```shell
bash test.sh
```

Args:

```python
parser.add_argument('-st', '--start_time', type=int,
                        required=True, help="Starting time (six digits, YYYYMMDD)")
parser.add_argument('-et', '--end_time', type=int,
                        required=True, help="Ending time (six digits, YYYYMMDD)")
```

### Analysis Model

We also provide details for evaluating the model in `model_result_analysis.ipynb`, containing different metrics and plotting.

## 📬 **Contact**

🔥 We hope FCNet is helpful for your work. Thanks a lot for your attention.🔥

If you have any questions, feel free to contact us via Email:  
📧 Feng Gao: gaofeng@ouc.edu.cn  
📧 Jialiang Zhang: zhangjia_liang@foxmail.com  
We hope FCNet helps your research! ⭐ If you find our work useful, please cite:
```
@ARTICLE{10976656,
  author={Zhang, Jialiang and Gao, Feng and Gan, Yanhai and Dong, Junyu and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Frequency-Compensated Network for Daily Arctic Sea Ice Concentration Prediction}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Sea ice;Feature extraction;Arctic;Frequency-domain analysis;Predictive models;Adaptive filters;Numerical models;Biological system modeling;Information filters;Data models;Arctic sea ice prediction;deep learning;frequency compensation;sea ice concentration (SIC);spatial-temporal attention},
  doi={10.1109/TGRS.2025.3564457}}
```
