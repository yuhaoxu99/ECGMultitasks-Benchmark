import torch
import random
import logging
import argparse

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_loader import load_data_tsdl as load_data, ECGDataset
from models.DLinear import Model


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed):
    logger.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="0-shot testing on TS models")
    parser.add_argument("--file_path", type=str, default="./instruct_data/mimic_ecg.jsonl", help="Path to the JSONL file containing the dataset")
    parser.add_argument("--label", type=str, default="rr_interval", help="Column name for the label in the dataset")
    parser.add_argument("--model", type=str, default="TimesNet")
    parser.add_argument("--num_class", type=int, default=15)
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--downsample_size", type=int, default=5000)
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args


def collate_fn(batch, max_len=None):
    seq_lengths = torch.tensor([x.size(0) for x in batch])

    if max_len is None:
        max_len = seq_lengths.max().item()

    batch_size = len(batch)
    feat_dim = batch[0].size(1)

    padded_batch_x = torch.zeros((batch_size, max_len, feat_dim))
    padding_masks = torch.zeros((batch_size, max_len))

    for i, x in enumerate(batch):
        length = x.size(0)
        padded_batch_x[i, :length, :] = x  
        padding_masks[i, :length] = 1 

    return padded_batch_x, padding_masks


def train_model(model, dataloader, configs, criterion, optimizer):
    logger.info("Starting training...")
    model.train()
    for batch_idx, (data, label) in enumerate(dataloader):
        B, T, N = data.size()
        x_enc = data[:, :configs.seq_len, :].to("cuda")
        x_dec = data[:, -configs.pred_len:, :].to("cuda")
        label = label.to("cuda")

        outputs = model(x_enc, 0)
        loss = criterion(outputs.squeeze(), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"Batch {batch_idx} - Loss: {loss.item()}")


def test_model(model, dataloader, configs):
    logger.info("Starting testing...")
    model.eval()
    all_predictions = []
    all_labels = []

    projection = nn.Linear(1, configs.num_class).to("cuda")
    
    with torch.no_grad():
        for batch_idx, (data, label, _, _, _) in enumerate(dataloader):
            B, T, N = data.size()
            x_enc = data[:, :configs.seq_len, :].to("cuda")
            x_dec = data[:, -configs.pred_len:, :].to("cuda")
            label = label.to("cuda")

            enc_out = model(x_enc, 0)
            output = enc_out.reshape(enc_out.shape[0], -1)
            # (batch_size, num_classes)
            outputs = projection(output)

            all_predictions.append(outputs.detach())
            all_labels.append(label)
    
    logger.info(f"Batch {batch_idx} - Predictions: {all_predictions} - Labels: {all_labels}")
    
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

    total_acc, total_precision, total_recall, total_f_score = [], [], [], []

    for split_idx, split_ids in enumerate(subject_id_splits):
        logger.info(f"Training on split {split_idx+1}/{num_splits}")

        test_dataset = features_df[features_df['subject_id'].isin(split_ids)]
        test_dataset = ECGDataset(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print("TEST SAMPLES:", len(test_dataset))

        class Configs:
            seq_len = 500
            pred_len = 1
            label_len = 1
            d_model = 16
            d_ff = 32
            num_kernels = 6
            top_k = 3
            e_layers = 2
            enc_in = 12
            c_out = 1
            embed = 'fixed'
            freq = 'h'
            dropout = 0.3
            moving_avg = 1001
            weights = [1.0, 1.5]
            kernel_size = 1001

        configs = Configs()
        configs.task_name = args.task_name
        configs.num_class = args.num_class

        device = "cuda"
        model = Model(configs).to(device)
        model.load_state_dict(torch.load('./checkpoints/DLinear/checkpoint.pth',
                                        map_location=device))

        logger.info("Starting model testing...")
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

        logger.info(f"{args.model} - ACC: {accuracy}, Precision: {precision}, Recall: {recall}, F_Score: {f_score}")
        total_acc.append(accuracy)
        total_precision.append(precision)
        total_recall.append(recall)
        total_f_score.append(f_score)

    logger.info(f"{args.model} - ACC: {np.mean(total_acc):.2f} ± {np.std(total_acc):.2f}, "
            f"Precision: {np.mean(total_precision):.2f} ± {np.std(total_precision):.2f}, "
            f"Recall: {np.mean(total_recall):.2f} ± {np.std(total_recall):.2f}, "
            f"F_Score: {np.mean(total_f_score):.2f} ± {np.std(total_f_score):.2f}")


if __name__ == "__main__":
    main()