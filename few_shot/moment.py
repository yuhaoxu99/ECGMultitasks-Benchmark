import yaml
import torch
import random
import logging
import argparse

import numpy as np
from data_loader import load_data, ECGDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
)

from momentfm import MOMENTPipeline


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    parser = argparse.ArgumentParser(description="64-shot testing on ECG dataset with MOMENT")
    parser.add_argument("--file_path", type=str, default="./instruct_data/mimic_ecg.jsonl", help="Path to the JSONL file containing the dataset")
    parser.add_argument("--label", type=str, default="rr_interval", help="Column name for the label in the dataset")
    parser.add_argument("--config_path", type=str, default="./moment_config.yaml", help="Path to the config YAML file")
    parser.add_argument("--use_lora", type=bool, default=False)
    parser.add_argument("--shot_size", type=int, default=64)
    parser.add_argument("--task_name", type=str, default="regression")
    parser.add_argument("--downsample_size", type=int, default=12)
    parser.add_argument("--model", type=str, default="MOMENTPipeline", help="Model name")
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args


def load_model(config):
    logger.info("Loading model")
    # Assuming MOMENTPipeline is defined in your code or imported properly
    model = MOMENTPipeline.from_pretrained(
            config['model_path'],
            model_kwargs={
                'task_name': 'classification',
                'n_channels': config['n_channels'],
                'num_class': config['num_class'],
                'freeze_encoder': config['freeze_encoder'],
                'freeze_embedder': config['freeze_embedder']
            }
        )
    
    model.init()
    model = model.to(device)

    logger.info("Model loaded and moved to device")
    return model


def finetune_model(model, train_dataloader, output_dir="./lora_output", task_name="regression"):
    logger.info("Starting fine-tuning")

    logger.info("Trainer initialized, starting training...")
    model.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = torch.nn.MSELoss() if task_name == "regression" else torch.nn.CrossEntropyLoss()

    for data, labels in train_dataloader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data.permute(0, 2, 1))
        loss = criterion(output.logits, labels)  # Assuming output is directly the logits or replace it with correct attribute.
        loss.backward()
        optimizer.step()
        logger.info(f"Loss: {loss.item()}")
    
    logger.info("Fine-tuning completed")
    return model


def _evaluate(args, model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            output = model(data.permute(0, 2, 1))
            output = output.logits
            if args.task_name == "classification":
                _, preds = torch.max(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                output = output.cpu().numpy()
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

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    unique_subject_ids = features_df['subject_id'].unique()

    num_splits = 3
    split_size = len(unique_subject_ids) // num_splits
    subject_id_splits = [unique_subject_ids[i*split_size: (i+1)*split_size] for i in range(num_splits)]

    mae_list, mse_list = [], []
    accuracy_list, precision_list, recall_list, f_score_list = [], [], [], []

    for split_idx, split_ids in enumerate(subject_id_splits):

        train_dataset = features_df[features_df['subject_id'].isin(split_ids[:args.shot_size])]
        test_dataset = features_df[~features_df['subject_id'].isin(split_ids[args.shot_size:])]

        logger.info(f"TRAIN PATIENTS: {len(unique_subject_ids)}, TRAIN SAMPLES: {len(train_dataset)}")
        logger.info(f"TEST SAMPLES: {len(test_dataset)}")

        train_dataset = ECGDataset(train_dataset)
        test_dataset = ECGDataset(test_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        use_lora = False
        model = load_model(config)

        model = finetune_model(model, train_dataloader, task_name=args.task_name).to(device)

        logger.info("Testing model on testing set")
        if args.task_name == 'classification':
            accuracy, precision, recall, f_score = _evaluate(args, model, test_dataloader)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f_score_list.append(f_score)
        else:
            mae, mse = _evaluate(args, model, test_dataloader)
            mae_list.append(mae)
            mse_list.append(mse)
    
    if args.task_name == 'classification':
        logger.info(f"{args.model} - ACC: {np.mean(accuracy_list):.2f} ± {np.std(accuracy_list):.2f}, "
                    f"Precision: {np.mean(precision_list):.2f} ± {np.std(precision_list):.2f}, "
                    f"Recall: {np.mean(recall_list):.2f} ± {np.std(recall_list):.2f}, "
                    f"F_Score: {np.mean(f_score_list):.2f} ± {np.std(f_score_list):.2f}")
    else:
        logger.info(f"{args.model} - MAE: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}, MSE: {np.mean(mse_list):.2f} ± {np.std(mse_list):.2f}")


if __name__ == "__main__":
    main()
