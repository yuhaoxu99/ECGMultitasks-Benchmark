import os
import torch
import random
import argparse
import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
)

from data_loader import load_data, ECGDataset
from fairseq_signals.models import build_model_from_checkpoint


root = os.getcwd()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


def set_seed(seed):
    logger.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot testing on ECGFM")
    parser.add_argument(
        "--file_path",
        type=str,
        default="./instruct_data/mimic_ecg.jsonl",
        help="Path to the JSONL file containing the dataset",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="label",
        help="Column name for the label in the dataset",
    )
    parser.add_argument("--shot_size", type=int, default=64)
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument(
        "--num_class", type=int, default=2, help="Number of classes for classification"
    )
    parser.add_argument("--downsample_size", type=int, default=5000)
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args


def train_model(args, model, dataloader, criterion, optimizer, projection):
    logger.info("Starting training...")
    model.train()
    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)
        output = model(source=data.permute(0, 2, 1))
        x = output["out"]

        logits = projection(x)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f"Batch {batch_idx} - Loss: {loss.item()}")


def _evaluate(args, model, dataloader, projection):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            output = model(source=data.permute(0, 2, 1))
            x = output["out"]

            logits = projection(x)

            if args.task_name == "classification":
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                output = logits.cpu().numpy()
                output = output.reshape(-1)
                all_preds.extend(output)
                labels = labels.cpu().numpy()
                labels = labels.reshape(-1)
                all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if args.task_name == "regression":
        mae = mean_absolute_error(all_labels, all_preds)
        mse = mean_squared_error(all_labels, all_preds)
        logger.info(f"Evaluation results - MAE: {mae}, MSE: {mse}")
        return mae, mse
    else:
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

    model_pretrained = build_model_from_checkpoint(
        checkpoint_path=os.path.join(root, 'ckpts/physionet_finetuned.pt')
    )
    model = model_pretrained.to(device)

    if args.task_name == "regression":
        criterion = torch.nn.MSELoss()
    else:
        if args.num_class == 2:
            class_weights = torch.tensor([1, 1], dtype=torch.float32).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    unique_subject_ids = features_df['subject_id'].unique()

    num_splits = 3
    split_size = len(unique_subject_ids) // num_splits
    subject_id_splits = [unique_subject_ids[i*split_size: (i+1)*split_size] for i in range(num_splits)]

    mae_list, mse_list = [], []
    accuracy_list, precision_list, recall_list, f_score_list = [], [], [], []

    for _, split_ids in enumerate(subject_id_splits):

        train_dataset = features_df[features_df['subject_id'].isin(split_ids[:args.shot_size])]
        test_dataset = features_df[features_df['subject_id'].isin(split_ids[args.shot_size:])]

        logger.info(f"TRAIN SAMPLES: {len(train_dataset)}, TEST SAMPLES: {len(test_dataset)}")
        
        train_dataset = ECGDataset(train_dataset)
        test_dataset = ECGDataset(test_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        projection = nn.Linear(26, args.num_class).to(device)

        train_model(args, model, train_dataloader, criterion, optimizer, projection)

        logger.info("Testing model on testing set")
        if args.task_name == 'classification':
            accuracy, precision, recall, f_score = _evaluate(args, model, test_dataloader, projection)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f_score_list.append(f_score)
        else:
            mae, mse = _evaluate(args, model, test_dataloader, projection)
            mae_list.append(mae)
            mse_list.append(mse)
    
    if args.task_name == 'classification':
        logger.info(f"ECG-FM - ACC: {np.mean(accuracy_list):.2f} ± {np.std(accuracy_list):.2f}, "
                    f"Precision: {np.mean(precision_list):.2f} ± {np.std(precision_list):.2f}, "
                    f"Recall: {np.mean(recall_list):.2f} ± {np.std(recall_list):.2f}, "
                    f"F_Score: {np.mean(f_score_list):.2f} ± {np.std(f_score_list):.2f}")
    else:
        logger.info(f"ECG-FM - MAE: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}, MSE: {np.mean(mse_list):.2f} ± {np.std(mse_list):.2f}")


if __name__ == "__main__":
    main()
