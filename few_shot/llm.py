import torch
import random
import logging
import argparse

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
from peft import get_peft_model, LoraConfig
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
)
from data_loader import load_data


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
    parser = argparse.ArgumentParser(
        description="Few-shot classification testing on LLM"
    )
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
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument(
        "--use_lora",
        type=bool, 
        default=True,
        help="Whether to use LoRA for fine-tuning",
    )
    parser.add_argument("--shot_size", type=int, default=64)
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument(
        "--num_class", type=int, default=2, help="Number of classes for classification"
    )
    parser.add_argument("--downsample_size", type=int, default=128)
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args


class TextDataset(Dataset):
    def __init__(self, input_texts, labels=None):
        self.input_texts = input_texts
        self.labels = labels
        logger.info(f"Initialized TextDataset with {len(input_texts)} samples")

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        item = {"input_text": self.input_texts[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def load_model_and_tokenizer(model_name, use_lora=False):
    logger.info(f"Loading model and tokenizer for: {model_name}")
    if "gpt2" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        target_modules=["c_attn", "c_proj"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "./LLM-Research/Meta-Llama-3___1-8B-Instruct", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "./LLM-Research/Meta-Llama-3___1-8B-Instruct",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Add pad_token if it does not exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        logger.info("Applying LoRA configuration to the model")
        lora_config = LoraConfig(
            r=4, lora_alpha=32, target_modules=target_modules, lora_dropout=0.1
        )
        model = get_peft_model(model, lora_config)

    model.to(device)
    logger.info("Model loaded and moved to device")
    return model, tokenizer


def prepare_text_data(features_df, label_column):
    logger.info("Preparing text data from features dataframe")
    input_texts = []
    labels = []
    # Use tqdm to wrap the iterable
    for _, row in tqdm(features_df.iterrows(), total=features_df.shape[0], desc="Processing rows"):
        signal_text = " ".join(map(str, row['signal_data']))
        input_text = f"Signal: {signal_text} | Label:"
        label = str(row["label"])
        input_texts.append(input_text)
        labels.append(label)
    logger.info(f"Prepared {len(input_texts)} text samples")
    return input_texts, labels


def finetune_with_lora(
    model, tokenizer, input_texts, labels, output_dir="./lora_output"
):
    logger.info("Starting fine-tuning with LoRA")
    # Create full texts by appending the label to the input
    full_texts = [
        input_text + " " + label for input_text, label in zip(input_texts, labels)
    ]

    # Tokenize the full texts
    train_encodings = tokenizer(
        full_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    # Shift labels appropriately
    labels = train_encodings["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens
    train_encodings["labels"] = labels

    train_encodings = {k: v.to(device) for k, v in train_encodings.items()}

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_dir="./logs",
        save_steps=10,
        logging_steps=10,
        no_cuda=not torch.cuda.is_available(),
        overwrite_output_dir=True,
        report_to=[],  # Disable wandb logging
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=torch.utils.data.TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            train_encodings["labels"],
        ),
        data_collator=default_data_collator,
    )
    logger.info("Trainer initialized, starting training...")
    trainer.train()
    logger.info("Fine-tuning completed")
    return model


def test_model(
    model, tokenizer, input_texts, labels, batch_size=32, max_length=512
):
    logger.info("Starting model testing")
    model.eval()
    all_predictions = []
    all_labels = labels
    num_batches = len(input_texts) // batch_size + int(
        len(input_texts) % batch_size != 0
    )
    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_input_texts = input_texts[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            inputs = tokenizer(
                batch_input_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_length=max_length)
            for i in range(len(outputs)):
                decoded_output = tokenizer.decode(
                    outputs[i], skip_special_tokens=True
                )
                # Extract the generated label
                if "Label:" in decoded_output:
                    prediction = decoded_output.split("Label:")[-1].strip()
                    # Clean the prediction to remove any extra generated text
                    prediction = prediction.split()[0]  # Take the first token
                else:
                    prediction = ""
                all_predictions.append(prediction)
    logger.info("Model testing completed")
    return all_predictions, all_labels


def evaluate_classification(
    predictions, labels, num_classes, model_name, dataset_name
):
    # Map predictions to class labels if necessary
    unique_labels = set(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

    # Map true labels to indices
    mapped_labels = [label_mapping[label.strip()] for label in labels]

    # Map predictions to indices
    mapped_predictions = []
    for pred in predictions:
        pred_label = pred.strip()
        if pred_label in label_mapping:
            mapped_predictions.append(label_mapping[pred_label])
        else:
            # Handle unknown predictions
            mapped_predictions.append(-1)  # Assign an invalid index

    # Filter out invalid predictions
    valid_indices = [
        i for i, pred in enumerate(mapped_predictions) if pred != -1
    ]
    if not valid_indices:
        logger.warning("No valid predictions to evaluate.")
        print(f"{model_name} - {dataset_name} - No valid predictions to evaluate.")
        return

    final_predictions = [mapped_predictions[i] for i in valid_indices]
    final_labels = [mapped_labels[i] for i in valid_indices]

    accuracy = accuracy_score(final_labels, final_predictions)
    if num_classes == 2:
        precision, recall, f_score, _ = precision_recall_fscore_support(
            final_labels, final_predictions, average="binary"
        )
    else:
        precision, recall, f_score, _ = precision_recall_fscore_support(
            final_labels, final_predictions, average="macro"
        )

    logger.info(
        f"{model_name} - {dataset_name} - ACC: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f_score:.4f}"
    )
    print(
        f"{model_name} - {dataset_name} - ACC: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f_score:.4f}"
    )
    return accuracy, precision, recall, f_score


def main():
    set_seed(42)
    args = parse_args()

    logger.info("Loading data")
    features_df = load_data(args)
    logger.info("Data loaded successfully")

    model, tokenizer = load_model_and_tokenizer(args.model, args.use_lora)

    unique_subject_ids = features_df['subject_id'].unique()

    num_splits = 3
    split_size = len(unique_subject_ids) // num_splits
    subject_id_splits = [unique_subject_ids[i*split_size: (i+1)*split_size] for i in range(num_splits)]

    mae_list, mse_list = [], []
    accuracy_list, precision_list, recall_list, f_score_list = [], [], [], []

    for _, split_ids in enumerate(subject_id_splits):

        train_dataset = features_df[features_df['subject_id'].isin(split_ids[:args.shot_size])]
        test_dataset = features_df[~features_df['subject_id'].isin(split_ids[args.shot_size:])]

        logger.info(f"TRAIN PATIENTS: {len(unique_subject_ids)}, TRAIN SAMPLES: {len(train_dataset)}")
        logger.info(f"TEST SAMPLES: {len(test_dataset)}")

        train_input_texts, train_labels = prepare_text_data(train_dataset, args.label)
        test_input_texts, test_labels = prepare_text_data(test_dataset, args.label)

        # Determine number of classes
        num_classes = args.num_class

        if args.use_lora:
            model = finetune_with_lora(
                model, tokenizer, train_input_texts, train_labels
            ).to(device)

        logger.info("Testing model on training set")
        train_predictions, train_true_labels = test_model(
            model, tokenizer, train_input_texts, train_labels, batch_size=8
        )
        if args.task_name == "regression":
            mae, mse = evaluate_metrics(train_predictions, train_true_labels)
            print(
                f"{args.model} - MAE (train set): {mae}, MSE (train set): {mse}"
            )
        else:
            accuracy, auroc_value, precision, recall, f_score = evaluate_classification(
                train_predictions,
                train_true_labels,
                num_classes,
                args.model,
                "train set",
            )

        logger.info("Testing model on test set")
        test_predictions, test_true_labels = test_model(
            model, tokenizer, test_input_texts, test_labels, batch_size=32
        )
        if args.task_name == "regression":
            mae, mse = evaluate_metrics(test_predictions, test_true_labels)
            print(f"{args.model} - MAE (test set): {mae}, MSE (test set): {mse}")
            mae_list.append(mae)
            mse_list.append(mse)
        else:
            accuracy, auroc_value, precision, recall, f_score = evaluate_classification(
                test_predictions,
                test_true_labels,
                num_classes,
                args.model,
                "test set",
            )
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f_score_list.append(f_score)
    
    if args.task_name == 'regression':
        logger.info(f"MAE: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}, MSE: {np.mean(mse_list):.2f} ± {np.std(mse_list):.2f}")
    else:
        logger.info(f"Accuracy: {np.mean(accuracy_list):.2f} ± {np.std(accuracy_list):.2f}, "
                    f"Precision: {np.mean(precision_list):.2f} ± {np.std(precision_list):.2f}, "
                    f"Recall: {np.mean(recall_list):.2f} ± {np.std(recall_list):.2f}, "
                    f"F1 Score: {np.mean(f_score_list):.2f} ± {np.std(f_score_list):.2f}")


if __name__ == "__main__":
    main()
