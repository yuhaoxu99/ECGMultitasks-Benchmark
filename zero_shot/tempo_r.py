import torch
import logging
import argparse

import numpy as np
import random as random
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.TEMPO import TEMPO
from data_loader import load_data_tempo as load_data, ECGDataset_tempo as ECGDataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

projection = nn.Linear(96, 1, bias=True).to("cuda")


def set_seed(seed):
    logger.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="64-shot testing on ECG dataset with LLaMA")
    parser.add_argument("--file_path", type=str, default="./instruct_data/mimic_ecg.jsonl", help="Path to the JSONL file containing the dataset")
    parser.add_argument("--label", type=str, default="rr_interval", help="Column name for the label in the dataset")
    parser.add_argument("--task_name", type=str, default="regression")
    parser.add_argument("--downsample_size", type=int, default=336)
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args


def load_model():
    logger.info("Loading model")
    model = TEMPO.load_pretrained_model(
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        repo_id = "Melady/TEMPO",
        filename = "TEMPO-80M_v1.pth",
        cache_dir = "./checkpoints/TEMPO_checkpoints"
    )
    # model.fc = nn.Linear(1152, 1, bias=True)
    logger.info("Model loaded and moved to device")

    return model


def _evaluate(args, model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for (data, labels, trend_stamp, seasonal_stamp, resid_stamp) in dataloader:
            data, labels = data.to(device), labels.to(device)
            x_enc = data[:, :336, :].to("cuda")  # Shape: (B, seq_len, 1)
            x_dec = data[:, -1:, :].to("cuda")  # Shape: (B, pred_len, 1)

            enc_out, _ = model(x_enc, 0, trend_stamp.to("cuda"), seasonal_stamp.to("cuda"), resid_stamp.to("cuda"))
            outputs = enc_out.view(enc_out.shape[0], -1)  # Flatten to [B, T * C]
            output = projection(outputs)

            if args.task_name != "regression":
                _, preds = torch.max(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                output = output.cpu().numpy().reshape(-1)
                all_preds.extend(output)
                labels = labels.cpu().numpy().reshape(-1)
                all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    logger.info(f"Evaluation results - MAE: {mae}, MSE: {mse}")
    return mae, mse


def main():
    set_seed(42)
    args = parse_args()

    logger.info("Loading data")
    features_df = load_data(args)
    logger.info("Data loaded successfully")

    unique_subject_ids = features_df['subject_id'].unique()

    num_splits = 3
    split_size = len(unique_subject_ids) // num_splits
    subject_id_splits = [unique_subject_ids[i * split_size: (i + 1) * split_size] for i in range(num_splits)]

    mae_list, mse_list = [], []

    for split_idx, split_ids in enumerate(subject_id_splits):
        logger.info(f"Training on split {split_idx + 1}/{num_splits}")

        test_dataset = features_df[features_df['subject_id'].isin(split_ids)]

        test_dataset = ECGDataset(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = load_model()

        logger.info("Evaluating model on test set")
        mae, mse = _evaluate(args, model, test_dataloader)
        mae_list.append(mae)
        mse_list.append(mse)

    logger.info(f"MAE: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}, MSE: {np.mean(mse_list):.2f} ± {np.std(mse_list):.2f}")


if __name__ == "__main__":
    main()
