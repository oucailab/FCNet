import argparse
import time
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from utils.tools import setup_logging
from dataset import SIC_dataset
from train import Trainer
from config import configs
from utils.metrics import *


def create_parser():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-st",
        "--start_time",
        type=int,
        required=True,
        help="Starting time (six digits, YYYYMMDD)",
    )
    parser.add_argument(
        "-et",
        "--end_time",
        type=int,
        required=True,
        help="Ending time (six digits, YYYYMMDD)",
    )
    parser.add_argument(
        "-save",
        "--save_result",
        action="store_true",
        help="Whether to save the test results",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    log_file = (
        f"{configs.test_results_path}/test_{configs.model}_{configs.input_length}.log"
    )
    logger = setup_logging(log_file)

    logger.info("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.info("######################## Start testing! ########################\n")

    logger.info("Model Configurations:\n")
    logger.info(configs.__dict__)

    logger.info(f"\nArguments:")
    logger.info(f"  start time: {args.start_time}")
    logger.info(f"  end time: {args.end_time}")
    logger.info(f"  output_dir: {configs.test_results_path}")
    logger.info(f"  data_paths: {configs.data_paths}")
    logger.info(f"  save_result: {args.save_result}")

    device = torch.device("cuda:0")

    dataset_test = SIC_dataset(
        configs.data_paths,
        args.start_time,
        args.end_time,
        configs.input_gap,
        configs.input_length,
        configs.pred_shift,
        configs.pred_gap,
        configs.pred_length,
        samples_gap=1,
    )

    dataloader_test = DataLoader(dataset_test, shuffle=False)

    tester = Trainer()

    tester.network.load_state_dict(
        torch.load(
            f"checkpoints/checkpoint_{configs.model}_{configs.input_length}.pt",
            weights_only=True,
            map_location=device,
        )["net"]
    )

    logger.info("\nTesting......")

    tester.network.eval()

    mask = torch.from_numpy(np.load("data/AMAP_mask.npy")).to(device)

    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            with autocast(device_type="cuda"):
                sic_pred, loss = tester.network(inputs, targets)

            mse = mse_func(sic_pred, targets, mask)
            rmse = rmse_func(sic_pred, targets, mask)
            mae = mae_func(sic_pred, targets, mask)
            nse = nse_func(sic_pred, targets, mask)
            PSNR = PSNR_func(sic_pred, targets, mask)
            BACC = BACC_func(sic_pred, targets, mask)

    logger.info(
        f"\nMetrics: mse: {mse:.5f}, rmse: {rmse:.5f}, mae: {mae:.5f}, nse: {nse:.5f}, PSNR: {PSNR:.5f}, BACC: {BACC:.5f}, loss: {loss:.5f}"
    )

    if args.save_result:

        logger.info(f"\nSaving output to {configs.test_results_path}")

        np.save(
            f"{configs.test_results_path}/sic_pred_{configs.model}.npy", sic_pred.cpu()
        )
        np.save(f"{configs.test_results_path}/inputs.npy", dataset_test.get_inputs())
        np.save(f"{configs.test_results_path}/targets.npy", dataset_test.get_targets())
        np.save(f"{configs.test_results_path}/times.npy", dataset_test.get_times())

    logger.info("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.info("######################## End of test! ########################")
