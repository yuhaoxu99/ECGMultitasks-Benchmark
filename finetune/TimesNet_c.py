import torch
import random
import logging
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_loader import load_data, ECGDataset
from models.TimesNet import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

act = F.gelu
dropout = nn.Dropout(0.3)
projection = nn.Linear(32, 2).to("cuda")


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
    parser.add_argument("--shot_size", type=int, default=64)
    parser.add_argument("--num_class", type=int, default=15)
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--downsample_size", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--model_save_path", type=str, default="", help="Path to save the best model")
    parser.add_argument("--pred_len", type=int, default=0)
    args = parser.parse_args()

    # Update model_save_path with dynamic components
    if not args.model_save_path:
        args.model_save_path = f"./checkpoints/{args.model}/{args.label}/best_model.pth"

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


def train_model(model, train_dataloader, val_dataloader, configs, criterion, optimizer, num_epochs, model_save_path):
    logger.info("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, label) in enumerate(train_dataloader):
            # data, x_mark_enc = collate_fn(data, configs.seq_len)
            x_enc = data[:, :configs.seq_len, :].to(device)
            x_dec = data[:, -configs.pred_len:, :].to(device)
            label = label.to(device)

            enc_out = model(x_enc, None, x_dec, None)
            output = act(enc_out)
            output = dropout(output)
            # zero-out padding embeddings
            # output = output * x_mark_enc.unsqueeze(-1)
            # (batch_size, seq_length * d_model)
            output = output.reshape(output.shape[0], -1)
            outputs = projection(output)  # (batch_size, num_classes)

            loss = criterion(outputs.squeeze(), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}] - Loss: {loss.item()}")

        if (epoch+1) % 20 == 0:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(val_dataloader):
                    # data, x_mark_enc = collate_fn(data, configs.seq_len)
                    x_enc = data[:, :configs.seq_len, :].to(device)
                    x_dec = data[:, -configs.pred_len:, :].to(device)
                    label = label.to(device)

                    enc_out = model(x_enc, None, x_dec, None)
                    output = act(enc_out)
                    output = dropout(output)
                    # zero-out padding embeddings
                    # output = output * x_mark_enc.unsqueeze(-1)
                    # (batch_size, seq_length * d_model)
                    output = output.reshape(output.shape[0], -1)
                    outputs = projection(output)  # (batch_size, num_classes)
                    loss = criterion(outputs.squeeze(), label)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"Model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")


def test_model(model, dataloader, configs, model_save_path):
    logger.info("Starting testing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            data, x_mark_enc = collate_fn(data, configs.seq_len)
            x_enc = data[:, :configs.seq_len, :].to(device)
            x_dec = data[:, -configs.pred_len:, :].to(device)
            label = label.to(device)

            enc_out = model(x_enc, None, x_dec, None)
            output = act(enc_out)
            output = dropout(output)
            # zero-out padding embeddings
            output = output * x_mark_enc.unsqueeze(-1)
            # (batch_size, seq_length * d_model)
            output = output.reshape(output.shape[0], -1)
            outputs = projection(output)  # (batch_size, num_classes)
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

    acc_list, precision_list, recall_list, fscore_list, auroc_list = [], [], [], [], []

    for split_idx, split_ids in enumerate(subject_id_splits):
        logger.info(f"Training on split {split_idx+1}/{num_splits}")

        train_subject_ids = split_ids[:int(len(split_ids) * 0.7)]
        val_subject_ids = split_ids[int(len(split_ids) * 0.7):int(len(split_ids) * 0.9)]
        test_subject_ids = split_ids[int(len(split_ids) * 0.9):]

        train_dataset = features_df[features_df['subject_id'].isin(train_subject_ids)]
        val_dataset = features_df[features_df['subject_id'].isin(val_subject_ids)]
        test_dataset = features_df[features_df['subject_id'].isin(test_subject_ids)]

        train_dataset = ECGDataset(train_dataset)
        val_dataset = ECGDataset(val_dataset)
        test_dataset = ECGDataset(test_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        class Configs:
            seq_len = 500
            label_len = 1
            d_model = 16
            d_ff = 32
            num_kernels = 6
            top_k = 3
            e_layers = 2
            enc_in = 12
            c_out = 1
            embed = 'timeF'
            freq = 'h'
            dropout = 0.3
            weights = [1, 1]

        configs = Configs()
        configs.task_name = "long_term_forecast"
        configs.num_class = args.num_class
        configs.pred_len = args.pred_len

        model = Model(configs)
        model.load_state_dict(torch.load('/local/scratch/yxu81/PhysicialFM/checkpoints/TimesNet/checkpoint.pth',
                                        map_location="cuda"))

        if isinstance(model, nn.Sequential):
            model = nn.Sequential(*list(model.children())[:-1])

        class_weights = torch.tensor(configs.weights, dtype=torch.float32).to("cuda")
        if configs.num_class == 2:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_dataloader, val_dataloader, configs, criterion, optimizer, args.num_epochs, args.model_save_path)

        predictions, labels = test_model(model, test_dataloader, configs, args.model_save_path)

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