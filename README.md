# ECGMultitasks-Benchmark
Official code for "[An Electrocardiogram Multi-Task Benchmark with Comprehensive Evaluations and Insightful Findings](https://arxiv.org/abs/2512.08954)". This paper has been accepted by The 20th World Congress on Medical and Health Informatics (MedInfo 2025).

We provide a comprehensive ECG multitasks benchmark to evaluate `large language models`, `general time-series foundation models`, and `ECG foundation model` in comparison with `time-series deep learning models` across five different types of downstream tasks under `zero-shot`, `few-shot`, and `fine-tuning` settings, including `RR Interval Estimation`, `Age Estimation`, `Gender Classification`, `Potassium Abnormality Prediction` and `Arrhythmia Detection`.

## Prepare Dataset
### Prepare ECG Data
* [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
### Prepare Subset Data and Label
* [Google Drive](https://drive.google.com/file/d/1QAiLpPYi_MnLV70Vu4TQ9K73C4Nc008j/view?usp=share_link)
  
We provide `.jsonl` file subset from the MIMIC-IV-ECG, along with the corresponding labels to evaluate in different downstream tasks, including RR Interval Estimation `rr_interval`, Age Estimation `age`, Gender Classification `gender`, Potassium Abnormality Prediction `flag`, and Arrhythmia Detection `report_label`.

## Installation
The required packages can be installed by running `pip install -r requirements.txt`.

For `ECG-FM` environment please refer the link [ECG-FM](https://github.com/bowang-lab/ECG-FM) and [fairseq-signals](https://github.com/Jwoo5/fairseq-signals).

## 🚀Quick Start
In the `scripts` folder, we provide shell scripts, and you can change the `--task_name` parameter to start the evaluation.
