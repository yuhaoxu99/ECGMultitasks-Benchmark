import yaml
import torch
import random
import logging
import argparse

import numpy as np
from momentfm import MOMENTPipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
)

from data_loader import load_data, ECGDataset


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
    parser = argparse.ArgumentParser(description="Finetune testing on ECG dataset with MOMENT")
    parser.add_argument("--file_path", type=str, default="./instruct_data/mimic_ecg.jsonl", help="Path to the JSONL file containing the dataset")
    parser.add_argument("--label", type=str, default="rr_interval", help="Column name for the label in the dataset")
    parser.add_argument("--config_path", type=str, default="./moment_config.yaml", help="Path to the config YAML file")
    parser.add_argument("--use_lora", type=bool, default=False)
    parser.add_argument("--task_name", type=str, default="regression")
    parser.add_argument("--downsample_size", type=int, default=5000)
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs for training")
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args


def load_model(config, use_lora=False):
    logger.info("Loading model")
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

    if use_lora:
        logger.info("Applying LoRA configuration to the model")
        lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
        model = get_peft_model(model, lora_config)
    
    model.to(device)
    logger.info("Model loaded and moved to device")
    return model


def finetune_model(model, train_dataloader, val_dataloader, task_name, num_epochs=20):
    logger.info("Starting fine-tuning")
    model.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    if task_name == "regression":
        criterion = torch.nn.MSELoss()
    else:
        logger.info("Load CrossEntropyLoss!")
        criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data.permute(0, 2, 1))
            loss = criterion(output.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, labels in val_dataloader:
                    data, labels = data.to(device), labels.to(device)
                    output = model(data.permute(0, 2, 1))
                    if task_name == "regression":
                        loss = criterion(output.logits, labels)
                    else:
                        loss = criterion(output.logits.float(), labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
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
    for data, labels in dataloader:
        data, labels = data.permute(0, 2, 1).to(device), labels.to(device)
        data.requires_grad = True
        output = model(data)
        output = output.logits

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
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = load_model(config, use_lora=False)

        model = finetune_model(model, train_dataloader, val_dataloader, task_name=args.task_name, num_epochs=args.num_epochs).to(device)

        logger.info("Evaluating model on test set")
        if args.task_name == 'regression':
            mae, mse = _evaluate(args, model, test_dataloader)
            mae_list.append(mae)
            mse_list.append(mse)
        else:
            accuracy, precision, recall, f_score = _evaluate(args, model, test_dataloader)
            acc_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            fscore_list.append(f_score)

    if args.task_name == 'regression':
        logger.info(f"MAE: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}, MSE: {np.mean(mse_list):.2f} ± {np.std(mse_list):.2f}")
    else:
        logger.info(f"Accuracy: {np.mean(acc_list):.2f} ± {np.std(acc_list):.2f}, "
                    f"Precision: {np.mean(precision_list):.2f} ± {np.std(precision_list):.2f}, "
                    f"Recall: {np.mean(recall_list):.2f} ± {np.std(recall_list):.2f}, "
                    f"F1 Score: {np.mean(fscore_list):.2f} ± {np.std(fscore_list):.2f}")


if __name__ == "__main__":
    main()
