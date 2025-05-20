"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-09-26 18:28:47
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-10-16 14:38:03
FilePath: /root/OSI-SAF/config.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""


class Configs:
    def __init__(self):
        # self.model = "ConvLSTM"
        # self.model = "PredRNN"
        # self.model = "SimVP"
        # self.model = "TAU"
        # self.model = "Swin_Transformer"
        self.model = "FCNet"
        # self.model = "SICFN"

        # paths
        self.data_paths = "data/data_path.txt"
        self.train_log_path = "train_logs"
        self.test_results_path = "test_results"

        # trainer related
        self.batch_size_vali = 16
        self.batch_size = 4
        self.lr = 5e-4
        self.weight_decay = 1e-2
        self.num_epochs = 200
        self.early_stop = True
        self.patience = self.num_epochs // 10
        self.gradient_clip = True
        self.clip_threshold = 1.0
        self.num_workers = 32

        # data related
        self.img_size = (432, 432)

        # 这种数据的加载方式有些不方便，有更好的方式，可以联系我（爱吃菠萝）
        self.input_dim = 1  #  input_dim: 输入张量对应的通道数
        self.output_dim = 1  #  output_dim: 输入张量对应的通道数

        self.input_length = 14  # 每轮训练输入多少张数据
        self.pred_length = 14  # 每轮训练输出多少张数据

        self.input_gap = 1  # 每张输入数据之间的间隔
        self.pred_gap = 1  # 每张输出数据之间的间隔

        self.pred_shift = self.pred_gap * self.pred_length

        self.train_period = (19910101, 20100101)
        self.eval_period = (20100101, 20151231)

        # model related
        if self.model == "FCNet":
            self.hid_S = 64
            self.hid_T = 256
            self.N_T = 8
            self.N_S = 4
            self.spatio_kernel_enc = 3
            self.spatio_kernel_dec = 3
            self.mlp_ratio = 4.0
            self.drop = 0.05
            self.drop_path = 0.05
            self.dropcls = 0.05
            self.patch_embed_size = (8, 8)
            self.patch_size = (2, 2)

        elif self.model == "SICFN":
            self.hid_S = 64
            self.hid_T = 256
            self.N_T = 8
            self.N_S = 4
            self.spatio_kernel_enc = 3
            self.spatio_kernel_dec = 3
            self.mlp_ratio = 4.0
            self.drop = 0.05
            self.drop_path = 0.05
            self.dropcls = 0.05
            self.patch_embed_size = (8, 8)
            self.patch_size = (2, 2)
            self.fno_blocks = 8
            self.fno_bias = True
            self.fno_softshrink = 0.05

        elif self.model == "ConvLSTM":
            self.kernel_size = (3, 3)
            self.patch_size = (2, 2)
            self.hidden_dim = (
                64,
                64,
                64,
                64,
            )  # hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数。

        elif self.model == "PredRNN":
            self.kernel_size = (3, 3)
            self.patch_size = (2, 2)
            self.hidden_dim = (
                64,
                64,
                64,
                64,
            )  # hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数。
            self.layer_norm = False

        elif self.model == "SimVP":
            self.hid_S = 64
            self.hid_T = 256
            self.N_T = 8
            self.N_S = 4
            self.spatio_kernel_enc = 3
            self.spatio_kernel_dec = 3
            self.patch_size = (2, 2)

        elif self.model == "SimVP":
            self.hid_S = 64
            self.hid_T = 256
            self.N_T = 8
            self.N_S = 4
            self.spatio_kernel_enc = 3
            self.spatio_kernel_dec = 3
            self.patch_size = (2, 2)
            self.mlp_ratio = 4.0
            self.drop = 0.05
            self.drop_path = 0.05

        elif self.model == "TAU":
            self.hid_S = 64
            self.hid_T = 256
            self.N_T = 8
            self.N_S = 4
            self.spatio_kernel_enc = 3
            self.spatio_kernel_dec = 3
            self.patch_size = (2, 2)
            self.mlp_ratio = 4.0
            self.drop = 0.05
            self.drop_path = 0.05

        elif self.model == "Swin_Transformer":
            self.hid_S = 64
            self.hid_T = 256
            self.N_T = 8
            self.N_S = 4
            self.spatio_kernel_enc = 3
            self.spatio_kernel_dec = 3
            self.patch_size = (2, 2)
            self.mlp_ratio = 4.0
            self.drop = 0.05
            self.drop_path = 0.05


configs = Configs()
