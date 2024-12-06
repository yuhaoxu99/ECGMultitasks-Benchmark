import re
import torch
import random
import logging
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
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
    parser.add_argument("--shot_size", type=int, default=64)
    parser.add_argument(
        "--task_name",
        type=str,
        default="classification",
        help="Task type: classification or regression",
    )
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


def prepare_text_data(features_df, label_column, train):
    logger.info("Preparing text data from features dataframe")

    data_df = pd.read_csv("/local/scratch/yxu81/PhysicialFM/physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv")

    columns_to_read = ['study_id', 'rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']
    input_texts = []
    labels = []

    for _, row in features_df.iterrows():
        study_id = row['study_id']
        label_value = row['label']  

        feature_row = data_df[data_df['study_id'] == study_id]

        feature_row = feature_row.iloc[0]

        feature_text = " | ".join([
            f"{col}: {feature_row[col] if not pd.isna(feature_row[col]) else 'NaN'}" 
            for col in columns_to_read[1:]  # 忽略 study_id 列
        ])

        if label_column == "gender":
            label_text = "predict the patient's gender as a numerical value, 0: Male, 1: Female"
        elif label_column == "flag":
            label_text = "predict the patient's blood potassium as a numerical value, 0: normal, 1: abnormal"
        elif label_column == "age":
            label_text = "predict the patient's age the age between 0-91"
        elif label_column == "rr_interval":
            label_text = "predict the rr interval as a numerical value"
        else:
            label_text = ("predict the patient's Arrhythmia Detection as a numerical value, 0: Sinus rhythm, 1: Sinus bradycardia, 2: Sinus tachycardia, 3: atrial fibrillation, "
                          "4: sinus rhythm with borderline 1st degree a-v block, 5: sinus rhythm with 1st degree a-v block, "
                          "6: atrial fibrillation with rapid ventricular response, 7: sinus arrhythmia, "
                          "8: sinus rhythm with pac(s), 9: ventricular pacing, 10: consider acute st elevation mi ***, "
                          "11: warning: data quality may affect interpretation, 12: sinus rhythm with pvc(s), "
                          "13: possible ectopic atrial rhythm, 14: other")

        input_text = (
            "Given the following features calculated from ECG data, predict the label as a single numerical value.\n"
            f"Features: {feature_text}\n"
            f"| {label_text}\nPlease provide the {label_column} as a numerical value after 'Label:'.\nLabel:"
        )

        input_texts.append(input_text)
        labels.append(str(label_value))

    if input_texts:
        print("input_texts:", input_texts[0])

    logger.info(f"Prepared {len(input_texts)} text samples")
    return input_texts, labels


def load_model_and_tokenizer(model_name, use_lora=False):
    logger.info(f"Loading model and tokenizer for: {model_name}")
    if "gpt2" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
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

    # 添加 pad_token，如果不存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        logger.info("Applying LoRA configuration to the model")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)

    logger.info("Model loaded and ready")
    return model, tokenizer


def finetune_with_lora(
    model, tokenizer, input_texts, labels, output_dir="./lora_output"
):
    logger.info("Starting fine-tuning with LoRA")

    full_texts = [
        input_text + " " + label
        for input_text, label in zip(input_texts, labels)
    ]

    train_encodings = tokenizer(
        full_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    train_encodings["labels"] = train_encodings["input_ids"].clone()
    train_encodings["labels"][
        train_encodings["input_ids"] == tokenizer.pad_token_id
    ] = -100

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return self.encodings["input_ids"].size(0)

        def __getitem__(self, idx):
            return {key: tensor[idx] for key, tensor in self.encodings.items()}

    train_dataset = CustomDataset(train_encodings)

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
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    logger.info("Trainer initialized, starting training...")
    trainer.train()
    logger.info("Fine-tuning completed")
    return model


def test_model(
    model, tokenizer, input_texts, labels, batch_size=8, max_length=512
):
    logger.info("Starting model testing")
    model.eval()
    all_predictions = []
    all_labels = labels
    num_batches = len(input_texts) // batch_size + int(
        len(input_texts) % batch_size != 0
    )
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Testing batches"):
            batch_input_texts = input_texts[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            inputs = tokenizer(
                batch_input_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            for i in range(len(outputs)):
                decoded_output = tokenizer.decode(
                    outputs[i], skip_special_tokens=True
                )
                logger.info(
                    f"Decoded output at index {batch_idx * batch_size + i}: '{decoded_output}'"
                )
                # 提取生成的标签
                match = re.search(r"Label:\s*(.*)", decoded_output)
                if match:
                    prediction = match.group(1).strip()
                    try:
                        float(prediction)
                    except ValueError:
                        prediction = '0'
                else:
                    prediction = '0'
                print("prediction:", prediction)
                all_predictions.append(prediction)
    logger.info("Model testing completed")
    return all_predictions, all_labels


def evaluate_metrics(predictions, labels):
    logger.info("Evaluating regression metrics")
    predictions_float = []
    labels_float = []
    for pred, label in zip(predictions, labels):
        try:
            predictions_float.append(float(pred))
            labels_float.append(float(label))
        except ValueError:
            continue
    if not predictions_float or not labels_float:
        logger.warning("No valid predictions or labels to evaluate.")
        return None, None
    mae = mean_absolute_error(labels_float, predictions_float)
    mse = mean_squared_error(labels_float, predictions_float)
    logger.info(f"Evaluation results - MAE: {mae}, MSE: {mse}")
    return mae, mse


def evaluate_classification(
    predictions, labels, num_classes, model_name, dataset_name
):

    unique_labels = sorted(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

    mapped_labels = [label_mapping[label.strip()] for label in labels]

    mapped_predictions = []
    for pred in predictions:
        pred_label = pred.strip()
        if pred_label in label_mapping:
            mapped_predictions.append(label_mapping[pred_label])
        else:
            mapped_predictions.append(-1)

    valid_indices = [i for i, pred in enumerate(mapped_predictions) if pred != -1]
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
    return accuracy, precision, recall, f_score


def main():
    set_seed(42)
    args = parse_args()

    logger.info("Loading data")
    features_df = load_data(args)
    logger.info("Data loaded successfully")

    num_classes = args.num_class

    model, tokenizer = load_model_and_tokenizer(args.model, args.use_lora)
    unique_subject_ids = features_df['subject_id'].unique()

    num_splits = 3
    split_size = len(unique_subject_ids) // num_splits
    subject_id_splits = [unique_subject_ids[i*split_size: (i+1)*split_size] for i in range(num_splits)]

    mae_list, mse_list = [], []
    accuracy_list, precision_list, recall_list, f_score_list = [], [], [], []

    for _, split_ids in enumerate(subject_id_splits):

        test_dataset = features_df[features_df['subject_id'].isin(split_ids)]

        logger.info(f"TEST SAMPLES: {len(test_dataset)}")
        test_input_texts, test_labels = prepare_text_data(test_dataset, args.label, False)


        model.to(device)

        logger.info("Testing model on test set")
        test_predictions, test_true_labels = test_model(
            model, tokenizer, test_input_texts, test_labels, batch_size=32
        )
        if args.task_name == "regression":
            mae, mse = evaluate_metrics(test_predictions, test_true_labels)
            if mae is not None and mse is not None:
                print(f"{args.model} - MAE (test set): {mae}, MSE (test set): {mse}")
            else:
                print(f"{args.model} - No valid predictions to evaluate on the test set.")
        else:
            accuracy, precision, recall, f_score = evaluate_classification(
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