import torch
import argparse
import random
import logging

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_data_tsdl as load_data, ECGDataset
from models.DLinear import Model


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

projection = nn.Linear(1, 1).to("cuda")

def set_seed(seed):
    logger.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="64-shot testing on TS models")
    parser.add_argument("--file_path", type=str, default="/local/scratch/yxu81/PhysicialFM/instruct_data/mimic_ecg.jsonl", help="Path to the JSONL file containing the dataset")
    parser.add_argument("--label", type=str, default="rr_interval", help="Column name for the label in the dataset")
    parser.add_argument("--model", type=str, default="TimesNet")
    parser.add_argument("--shot_size", type=int, default=64)
    parser.add_argument("--task_name", type=str, default="regression")
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--downsample_size", type=int, default=500)
    args = parser.parse_args()

    # Update model_save_path with dynamic components
    if not args.model_save_path:
        args.model_save_path = f"./checkpoints/{args.model}/{args.label}/best_model.pth"
    logger.info(f"Parsed arguments: {args}")
    return args


def train_model(model, train_dataloader, configs, criterion, optimizer):
    logger.info("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.train()
    for batch_idx, (data, label) in enumerate(train_dataloader):
        data = data.to(device)
        x_enc = data[:, :configs.seq_len, :].to(device)
        x_dec = data[:, -configs.pred_len:, :].to(device)
        label = label.to(device)

        enc_out = model(x_enc, 0)
        enc_out = enc_out.permute(0, 2, 1).squeeze(-1)
        outputs = projection(enc_out)  # (batch_size, num_classes)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info(f"Loss: {loss / len(train_dataloader)}")
    
    return model


def test_model(model, dataloader, configs):
    logger.info("Starting testing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            data = data.to(device)
            x_enc = data[:, :configs.seq_len, :].to(device)
            x_dec = data[:, -configs.pred_len:, :].to(device)
            label = label.to(device)

            enc_out = model(x_enc, 0)
            enc_out = enc_out.permute(0, 2, 1).squeeze(-1)
            outputs = projection(enc_out)  # (batch_size, num_classes)
            all_predictions.append(outputs.squeeze())
            all_labels.append(label)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

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


        train_dataset = features_df[features_df['subject_id'].isin(split_ids[:args.shot_size])]
        test_dataset = features_df[features_df['subject_id'].isin(split_ids[args.shot_size:])]

        logger.info(f"TRAIN PATIENTS: {split_ids[:args.shot_size]}, TRAIN SAMPLES: {len(train_dataset)}")
        logger.info(f"TEST PATIENTS: {split_ids[args.shot_size:]}, TEST SAMPLES: {len(test_dataset)}")

        train_dataset = ECGDataset(train_dataset)
        test_dataset = ECGDataset(test_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


        class Configs:
            seq_len = 500
            pred_len = 1 
            label_len = 1
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
            moving_avg = 1001
            weights = [1.0, 1.5]
            kernel_size = 1001

        configs = Configs()
        configs.task_name = "long_term_forecast"

        device = "cuda"
        model = Model(configs).to(device)
        model.load_state_dict(torch.load('./checkpoints/DLinear/checkpoint.pth',
                                        map_location=device))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        logger.info("Starting model training...")
        model = train_model(model, train_dataloader, configs, criterion, optimizer)

        logger.info("Starting model testing...")
        predictions, labels = test_model(model, test_dataloader, configs)

        mae, mse = evaluate_metrics(predictions, labels)
        mae_list.append(mae)
        mse_list.append(mse)


    logger.info(f"{args.model} - MAE: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}, MSE: {np.mean(mse_list):.2f} ± {np.std(mse_list):.2f}")

if __name__ == "__main__":
    main()