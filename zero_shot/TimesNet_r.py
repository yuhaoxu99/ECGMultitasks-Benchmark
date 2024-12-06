import torch
import random
import logging
import argparse

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_data_tsdl as load_data, ECGDataset
from models.TimesNet import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
projection = nn.Linear(1, 1).to(device)


def set_seed(seed):
    logger.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="64-shot testing on TS models")
    parser.add_argument("--file_path", type=str, default="./instruct_data/mimic_ecg.jsonl", help="Path to the JSONL file containing the dataset")
    parser.add_argument("--label", type=str, default="rr_interval", help="Column name for the label in the dataset")
    parser.add_argument("--model", type=str, default="TimesNet")
    parser.add_argument("--task_name", type=str, default="regression")
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--downsample_size", type=int, default=500)
    args = parser.parse_args()

    # Update model_save_path with dynamic components
    if not args.model_save_path:
        args.model_save_path = f"./checkpoints/{args.model}/{args.label}/best_model.pth"
    logger.info(f"Parsed arguments: {args}")
    return args


def test_model(model, dataloader, configs):
    logger.info("Starting testing...")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, label, _, _, _) in enumerate(dataloader):
            data = data.to(device)
            x_enc = data[:, :configs.seq_len, :].to(device)
            x_dec = data[:, -configs.pred_len:, :].to(device)
            label = label.to(device)

            outputs = model(x_enc, None, x_dec, None)
            # outputs = outputs.permute(0, 2, 1)
            outputs = outputs.view(outputs.shape[0], -1)
            outputs = projection(outputs)
            all_predictions.append(outputs.squeeze())
            all_labels.append(label)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_predictions, all_labels


def evaluate_metrics(predictions, labels):
    logger.info("Evaluating metrics...")
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    logger.info(f"Evaluation results - MAE: {mae}, MSE: {mse}")
    return mae, mse


def main():
    set_seed(42)
    args = parse_args()

    logger.info("Loading data...")
    features_df = load_data(args)

    unique_subject_ids = features_df['subject_id'].unique()

    num_splits = 3
    split_size = len(unique_subject_ids) // num_splits
    subject_id_splits = [unique_subject_ids[i*split_size: (i+1)*split_size] for i in range(num_splits)]

    mae_list, mse_list = [], []

    for split_idx, split_ids in enumerate(subject_id_splits):
        logger.info(f"Training on split {split_idx+1}/{num_splits}")

        test_dataset = features_df[features_df['subject_id'].isin(split_ids)]

        test_dataset = ECGDataset(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        class Configs:
            seq_len = 500
            pred_len = 1
            label_len = 168
            d_model = 16
            d_ff = 32
            num_kernels = 6
            top_k = 3
            e_layers = 3
            enc_in = 1
            c_out = 1
            embed = 'timeF'
            freq = 'h'
            dropout = 0.3
            moving_avg = 101
            weights = [1.0, 1.5]
            kernel_size = 101

        configs = Configs()
        configs.task_name = "long_term_forecast"

        model = Model(configs).to(device)
        model.load_state_dict(torch.load('/vast/yx3534/PhyFM/checkpoints/TimesNet/checkpoint.pth',
                                        map_location=device))
        
        if isinstance(model, nn.Sequential):
            model = nn.Sequential(*list(model.children())[:-1])

        logger.info("Starting model testing...")
        predictions, labels = test_model(model, test_dataloader, configs)

        mae, mse = evaluate_metrics(predictions, labels)
        mae_list.append(mae)
        mse_list.append(mse)

    logger.info(f"{args.model} - MAE: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}, MSE: {np.mean(mse_list):.2f} ± {np.std(mse_list):.2f}")


if __name__ == "__main__":
    main()
