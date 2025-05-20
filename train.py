import os
import time
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

# from fvcore.nn import parameter_count_table, FlopCountAnalysis

from config import configs
from dataset import SIC_dataset
from utils.model_factory import IceNet
from utils.tools import setup_logging
from utils.metrics import *

device = torch.device("cuda:0")

dataset_train = SIC_dataset(
    configs.data_paths,
    configs.train_period[0],
    configs.train_period[1],
    configs.input_gap,
    configs.input_length,
    configs.pred_shift,
    configs.pred_gap,
    configs.pred_length,
    samples_gap=1,
)

dataset_vali = SIC_dataset(
    configs.data_paths,
    configs.eval_period[0],
    configs.eval_period[1],
    configs.input_gap,
    configs.input_length,
    configs.pred_shift,
    configs.pred_gap,
    configs.pred_length,
    samples_gap=1,
)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=configs.batch_size,
    num_workers=configs.num_workers,
    shuffle=True,
)

dataloader_vali = DataLoader(
    dataset_vali,
    batch_size=configs.batch_size_vali,
    num_workers=configs.num_workers,
    shuffle=False,
)

arctic_mask = torch.from_numpy(np.load("data/AMAP_mask.npy"))


class Trainer:

    def __init__(self):

        self.network = IceNet().to(device)

        self.optimizer = AdamW(
            self.network.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

        self.lr_scheduler = OneCycleLR(
            optimizer=self.optimizer,
            epochs=configs.num_epochs,
            steps_per_epoch=len(dataloader_train),
            max_lr=configs.lr,
        )

        # 初始化混合精度训练的GradScaler
        self.scaler = GradScaler()

    def save_model(self, path):
        torch.save(
            {
                "net": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),  # 保存scaler的状态
            },
            path,
        )

    def train_once(self, inputs, targets, mask):
        self.optimizer.zero_grad()

        inputs = inputs.float().to(device)
        targets = targets.float().to(device)
        mask = mask.float().to(device)

        with autocast(device_type="cuda"):
            sic_pred, loss = self.network(inputs, targets)

        # # 分析FLOPs
        # flops = FlopCountAnalysis(
        #     self.network, (inputs, targets,)
        # )
        # print("FLOPs: ", flops.total())

        # # 分析parameters
        # print(parameter_count_table(self.network))

        mse = mse_func(sic_pred, targets, mask)
        rmse = rmse_func(sic_pred, targets, mask)
        mae = mae_func(sic_pred, targets, mask)
        nse = nse_func(sic_pred, targets, mask)
        PSNR = PSNR_func(sic_pred, targets, mask)
        BACC = BACC_func(sic_pred, targets, mask)

        self.scaler.scale(loss).backward()

        if configs.gradient_clip:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clip_threshold)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.lr_scheduler.step()

        return (
            mse,
            rmse,
            mae,
            nse,
            PSNR,
            BACC,
            loss,
        )

    def vali(self, dataloader, mask):
        """
        evaluation part.

        Args:
        - dataloader: dataloader of evaluation dataset.
        """
        mask = mask.float().to(device)
        total_loss = 0
        total_mse = 0
        total_rmse = 0
        total_mae = 0
        total_nse = 0
        total_PSNR = 0
        total_BACC = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)

                with autocast(device_type="cuda"):
                    sic_pred, loss = self.network(inputs, targets)

                total_loss += loss
                total_mse += mse_func(sic_pred, targets, mask)
                total_rmse += rmse_func(sic_pred, targets, mask)
                total_mae += mae_func(sic_pred, targets, mask)
                total_nse += nse_func(sic_pred, targets, mask)
                total_PSNR += PSNR_func(sic_pred, targets, mask)
                total_BACC += BACC_func(sic_pred, targets, mask)

        return (
            total_mse / len(dataloader),
            total_rmse / len(dataloader),
            total_mae / len(dataloader),
            total_nse / len(dataloader),
            total_PSNR / len(dataloader),
            total_BACC / len(dataloader),
            total_loss / len(dataloader),
        )

    def train(self, chk_path):
        torch.manual_seed(42)

        log_file = (
            f"{configs.train_log_path}/train_{configs.model}_{configs.input_length}.log"
        )
        logger = setup_logging(log_file)

        logger.info("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logger.info(
            "######################## Training begins! ########################\n"
        )
        logger.info("Model Configurations:\n")
        logger.info(configs.__dict__)

        count = 0
        best = math.inf

        for epoch in range(configs.num_epochs):
            start_time = time.time()

            # train
            self.network.train()
            loop = tqdm(dataloader_train, total=len(dataloader_train), leave=True)
            for inputs, targets in loop:
                mse, rmse, mae, nse, PSNR, BACC, loss = self.train_once(
                    inputs, targets, arctic_mask
                )
                loop.set_description(f"Epoch [{epoch+1}/{configs.num_epochs}]")
                loop.set_postfix(
                    mse=f"{mse:.5f}",
                    rmse=f"{rmse:.5f}",
                    mae=f"{mae:.5f}",
                    nse=f"{nse:.5f}",
                    PSNR=f"{PSNR:.5f}",
                    BACC=f"{BACC:.5f}",
                    loss=f"{loss:.5f}",
                )

            # evaluation
            self.network.eval()
            (
                mse_eval,
                rmse_eval,
                mae_eval,
                nse_eval,
                PSNR_eval,
                BACC_eval,
                loss_eval,
            ) = self.vali(dataloader_vali, arctic_mask)

            logger.info(f"\nEpoch {epoch+1} Validation Metrics:")
            logger.info(f"  MSE: {mse_eval:.5f}")
            logger.info(f"  RMSE: {rmse_eval:.5f}")
            logger.info(f"  MAE: {mae_eval:.5f}")
            logger.info(f"  NSE: {nse_eval:.5f}")
            logger.info(f"  PSNR: {PSNR_eval:.5f}")
            logger.info(f"  BACC: {BACC_eval:.5f}")
            logger.info(f"  Loss: {loss_eval:.5f}")

            # 使用更好的模型保存方式，例如只保存最好的模型
            if loss_eval < best:
                count = 0
                logger.info(
                    f"Validation loss improved from {best:.5f} to {loss_eval:.5f}, saving model to {chk_path}"
                )
                self.save_model(chk_path)
                best = loss_eval
            else:
                count += 1
                logger.info(f"Validation loss did not improve for {count} epochs")

            logger.info(
                f"Time taken for epoch {epoch+1}: {time.time() - start_time:.2f}s"
            )

            if count == configs.patience and configs.early_stop:
                logger.info(
                    f"\nEarly stopping triggered. Best validation loss: {best:.5f}"
                )
                break

        logger.info("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logger.info(
            "######################## Training complete! ########################"
        )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 检查并创建 checkpoints 目录
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    trainer = Trainer()

    trainer.train(f"checkpoints/checkpoint_{configs.model}_{configs.input_length}.pt")
