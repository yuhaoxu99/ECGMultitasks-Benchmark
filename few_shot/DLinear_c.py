import os
import torch
import argparse
import random
import logging

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_loader import load_data_tsdl as load_data, ECGDataset

from models.DLinear import Model


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

projection = nn.Linear(1, 2).to("cuda")


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
    parser.add_argument("--num_class", type=int, default=15)
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--downsample_size", type=int, default=500)
    parser.add_argument("--pred_len", type=int, default=1)
    args = parser.parse_args()

    logger.info(f"Parsed arguments: {args}")
    return args


def train_model(model, train_dataloader, configs, criterion, optimizer):
    logger.info("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.train()
    for batch_idx, (data, label) in enumerate(train_dataloader):
        x_enc = data[:, :configs.seq_len, :].to(device)
        x_dec = data[:, -configs.pred_len:, :].to(device)
        label = label.to(device)

        enc_out = model(x_enc, 0)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        outputs = projection(output)

        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"Batch [{batch_idx+1}/{len(train_dataloader)}] - Loss: {loss.item()}")


def test_model(model, dataloader, configs):
    logger.info("Starting testing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            x_enc = data[:, :configs.seq_len, :].to(device)
            x_dec = data[:, -configs.pred_len:, :].to(device)
            label = label.to(device)

            enc_out = model(x_enc, 0)
            output = enc_out.reshape(enc_out.shape[0], -1)
            # (batch_size, num_classes)
            outputs = projection(output)
            all_predictions.append(outputs.detach())
            all_labels.append(label)

    return all_predictions, all_labels


def main():
    set_seed(42)
    args = parse_args()

    logger.info("Loading data...")
    features_df = load_data(args)

    unique_subject_ids = features_df['subject_id'].unique()

    num_splits = 3
    split_size = len(unique_subject_ids) // num_splits
    subject_id_splits = [unique_subject_ids[i*split_size: (i+1)*split_size] for i in range(num_splits)]

    acc_list, precision_list, recall_list, fscore_list = [], [], [], []

    for split_idx, split_ids in enumerate(subject_id_splits):
        logger.info(f"Training on split {split_idx+1}/{num_splits}")

        train_dataset = features_df[features_df['subject_id'].isin(split_ids[:args.shot_size])]
        test_dataset = features_df[features_df['subject_id'].isin(split_ids[args.shot_size:])]

        train_dataset = ECGDataset(train_dataset)
        test_dataset = ECGDataset(test_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
            kernel_size = 101

        configs = Configs()
        configs.task_name = args.task_name
        configs.num_class = args.num_class


        model = Model(configs, device)
        model.load_state_dict(torch.load('./checkpoints/DLinear/checkpoint.pth',
                                        map_location=device))

        class_weights = torch.tensor(configs.weights, dtype=torch.float32).to(device)
        if configs.num_class == 2:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_dataloader, configs, criterion, optimizer)

        predictions, labels = test_model(model, test_dataloader, configs)

        predictions = torch.cat(predictions, 0)
        labels = torch.cat(labels, 0)
        predictions = torch.nn.functional.softmax(predictions, dim=1)  # Specify dimension for softmax
        predictions = torch.argmax(predictions, dim=1).cpu().numpy()
        labels = labels.flatten().cpu().numpy()

        accuracy = accuracy_score(labels, predictions)
        if configs.num_class == 2:
            precision, recall, f_score, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        else:
            precision, recall, f_score, _ = precision_recall_fscore_support(labels, predictions, average='macro')

        acc_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(f_score)

    logger.info(f"{args.model} - ACC: {np.mean(acc_list):.2f} ± {np.std(acc_list):.2f}, "
                f"Precision: {np.mean(precision_list):.2f} ± {np.std(precision_list):.2f}, "
                f"Recall: {np.mean(recall_list):.2f} ± {np.std(recall_list):.2f}, "
                f"F_Score: {np.mean(fscore_list):.2f} ± {np.std(fscore_list):.2f}")


if __name__ == "__main__":
    main()
