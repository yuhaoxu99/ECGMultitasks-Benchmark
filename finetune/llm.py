import torch
import random
import logging
import argparse

import numpy as np
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
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
        description="Few-shot classification and regression testing on TS models"
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
        action="store_true",
        help="Whether to use LoRA for fine-tuning",
    )
    parser.add_argument("--task_name", type=str, default="classification", help="Task type: classification or regression")
    parser.add_argument(
        "--num_class",
        type=int,
        default=2,
        help="Number of classes for classification",
    )
    parser.add_argument("--downsample_size", type=int, default=5000)
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args


def split_dataset(features_df):
    unique_subject_ids = features_df['subject_id'].unique()
    random.shuffle(unique_subject_ids)

    # Split into 3 equal parts
    split_size = len(unique_subject_ids) // 3
    split_ids = [
        unique_subject_ids[i * split_size: (i + 1) * split_size]
        for i in range(3)
    ]
    datasets = []

    for ids in split_ids:
        num_train = int(0.7 * len(ids))
        num_val = int(0.2 * len(ids))

        train_ids = ids[:num_train]
        val_ids = ids[num_train:num_train + num_val]
        test_ids = ids[num_train + num_val:]

        train_dataset = features_df[features_df['subject_id'].isin(train_ids)]
        val_dataset = features_df[features_df['subject_id'].isin(val_ids)]
        test_dataset = features_df[features_df['subject_id'].isin(test_ids)]

        datasets.append((train_dataset, val_dataset, test_dataset))

    return datasets


def prepare_text_data(features_df, label_column, max_signal_length=512):
    logger.info("Preparing text data from features dataframe")

    features_df['features'] = features_df['signal_data'].apply(calculate_ecg_features)

    def features_to_text(features):
        return " | ".join([f"{key}: {value}" for key, value in features.items()])

    features_df['feature_text'] = features_df['features'].apply(features_to_text)

    features_df["input_text"] = (
        "Input the features caculated from ECG data as follows. Features: " + features_df["feature_text"] + " predict the | Label:"
    )

    features_df["label_str"] = features_df["label"].astype(str)

    input_texts = features_df["input_text"].tolist()
    labels = features_df["label_str"].tolist()

    logger.info(f"Prepared {len(input_texts)} text samples")
    return input_texts, labels


def train_llama(model, tokenizer, train_input_texts, train_labels, val_input_texts, val_labels, output_dir="./llama_output"):
    logger.info("Starting Llama 3.1 model training")

    full_train_texts = [
        input_text + " " + label
        for input_text, label in zip(train_input_texts, train_labels)
    ]

    full_val_texts = [
        input_text + " " + label
        for input_text, label in zip(val_input_texts, val_labels)
    ]

    train_encodings = tokenizer(
        full_train_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    val_encodings = tokenizer(
        full_val_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    train_encodings["labels"] = train_encodings["input_ids"].clone()
    train_encodings["labels"][
        train_encodings["input_ids"] == tokenizer.pad_token_id
    ] = -100

    val_encodings["labels"] = val_encodings["input_ids"].clone()
    val_encodings["labels"][
        val_encodings["input_ids"] == tokenizer.pad_token_id
    ] = -100

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return self.encodings["input_ids"].size(0)

        def __getitem__(self, idx):
            return {key: tensor[idx] for key, tensor in self.encodings.items()}

    train_dataset = CustomDataset(train_encodings)
    val_dataset = CustomDataset(val_encodings)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_dir="./logs",
        save_steps=10,
        logging_steps=10,
        no_cuda=not torch.cuda.is_available(),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        eval_steps=10,
        load_best_model_at_end=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    logger.info("Trainer initialized for Llama 3.1, starting training...")
    trainer.train()
    logger.info("Llama 3.1 training completed")
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


def main():
    set_seed(42)
    args = parse_args()

    logger.info("Loading data")
    features_df = load_data(args)
    logger.info("Data loaded successfully")

    datasets = split_dataset(features_df)

    metrics = []
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(datasets):
        logger.info(f"Processing split {i + 1}")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")

        train_input_texts, train_labels = prepare_text_data(train_dataset, args.label)
        val_input_texts, val_labels = prepare_text_data(val_dataset, args.label)
        test_input_texts, test_labels = prepare_text_data(test_dataset, args.label)

        if "gpt2" in args.model:
            model, tokenizer = GPT2LMHeadModel.from_pretrained(args.model), GPT2Tokenizer.from_pretrained(args.model)
            target_modules=["c_attn", "c_proj"]
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "/local/scratch/yxu81/PhysicialFM/LLM-Research/Meta-Llama-3___1-8B-Instruct",
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "/local/scratch/yxu81/PhysicialFM/LLM-Research/Meta-Llama-3___1-8B-Instruct", trust_remote_code=True
            )
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


        if args.use_lora:
            lora_config = LoraConfig(
                r=4,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_config)

        model.to(device)

        if args.model == "gpt2":
            model = train_model(model, tokenizer, train_input_texts, train_labels, val_input_texts, val_labels)
        else:
            model = train_llama(model, tokenizer, train_input_texts, train_labels, val_input_texts, val_labels)

        logger.info(f"Training and validation completed for split {i + 1}. Ready for testing.")

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
