import torch
import logging
import argparse

import numpy as np
import random as random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
)

from data_loader import load_data_tempo as load_data, ECGDataset_tempo as ECGDataset
from models.TEMPO import TEMPO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


act = F.gelu
dropout = nn.Dropout(0.3)
projection = nn.Linear(96, 15).to(device)
# projection = nn.Linear(96, 1, bias=True).to("cuda")


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
    parser.add_argument("--use_lora", type=bool, default=False)
    parser.add_argument("--shot_size", type=int, default=64)
    parser.add_argument("--task_name", type=str, default="regression")
    parser.add_argument("--downsample_size", type=int, default=336)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
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


def finetune_model(model, train_dataloader, val_dataloader, task_name, num_epochs=20):
    logger.info("Starting fine-tuning")
    model.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # class_weights = torch.tensor([1, 1], dtype=torch.float32).to("cuda")
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, labels, trend_stamp, seasonal_stamp, resid_stamp) in enumerate(train_dataloader):
            optimizer.zero_grad()
            B, T, N = data.size()
            data, labels = data.to(device), labels.to(device)

            x_enc = data[:, :336, :].to(device)  # Shape: (B, seq_len, 1)
            x_dec = data[:, -1:, :].to(device)  # Shape: (B, pred_len, 1)

            enc_out, _ = model(x_enc, 0, trend_stamp.to(device), seasonal_stamp.to(device), resid_stamp.to(device))
            outputs = enc_out.view(enc_out.shape[0], -1)  # Flatten to [B, T * C]
            outputs = projection(outputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % 20 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (data, labels, trend_stamp, seasonal_stamp, resid_stamp) in val_dataloader:
                    B, T, N = data.size()
                    data, labels = data.to(device), labels.to(device)

                    x_enc = data[:, :336, :].to(device)  # Shape: (B, seq_len, 1)

                    enc_out, _ = model(x_enc, 0, trend_stamp.to(device), seasonal_stamp.to(device), resid_stamp.to(device))
                    outputs = enc_out.view(enc_out.shape[0], -1)  # Flatten to [B, T * C]
                    outputs = projection(outputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss

            avg_val_loss = val_loss / (len(val_dataloader))
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_model.pth")
                logger.info(f"Model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")

    logger.info("Fine-tuning completed")
    return model


def _evaluate(args, model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for (data, labels, trend_stamp, seasonal_stamp, resid_stamp) in dataloader:
            data, labels = data.to(device), labels.to(device)
            x_enc = data[:, :336, :].to(device)  # Shape: (B, seq_len, 1)

            enc_out, _ = model(x_enc, 0, trend_stamp.to(device), seasonal_stamp.to(device), resid_stamp.to(device))
            outputs = enc_out.view(enc_out.shape[0], -1)  # Flatten to [B, T * C]
            output = projection(outputs)
            _, output = torch.max(output, dim=1)

            output = output.cpu().numpy()
            all_preds.extend(output)
            labels = labels.cpu().numpy().reshape(-1)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    logger.info(f"Evaluation results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f_score}")
    return accuracy, precision, recall, f_score


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
    acc_list, precision_list, recall_list, fscore_list = [], [], [], []

    for split_idx, split_ids in enumerate(subject_id_splits):
        logger.info(f"Training on split {split_idx + 1}/{num_splits}")

        train_subject_ids = split_ids[:int(len(split_ids) * 0.7)]
        val_subject_ids = split_ids[int(len(split_ids) * 0.7):int(len(split_ids) * 0.9)]
        test_subject_ids = split_ids[int(len(split_ids) * 0.9):]

        train_dataset = features_df[features_df['subject_id'].isin(train_subject_ids)]
        val_dataset = features_df[features_df['subject_id'].isin(val_subject_ids)]
        test_dataset = features_df[features_df['subject_id'].isin(test_subject_ids)]

        train_dataset = ECGDataset(train_dataset)
        val_dataset = ECGDataset(val_dataset)
        test_dataset = ECGDataset(test_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = load_model()
        model = finetune_model(model, train_dataloader, val_dataloader, task_name=args.task_name, num_epochs=args.num_epochs).to(device)

        logger.info("Evaluating model on test set")

        accuracy, precision, recall, f_score = _evaluate(args, model, test_dataloader)
        acc_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(f_score)

    logger.info(f"Accuracy: {np.mean(acc_list):.2f} ± {np.std(acc_list):.2f}, "
                f"Precision: {np.mean(precision_list):.2f} ± {np.std(precision_list):.2f}, "
                f"Recall: {np.mean(recall_list):.2f} ± {np.std(recall_list):.2f}, "
                f"F1 Score: {np.mean(fscore_list):.2f} ± {np.std(fscore_list):.2f}")


if __name__ == "__main__":
    main()

