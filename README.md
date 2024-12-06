# ECGMultitasks-Benchmark
Official code for "Are Foundation Models Useful for Electrocardiogram Analysis? A Multi-task Benchmark with Comprehensive Evaluations and Insightful Findings". 

We provide a comprehensive benchmark to evaluate large language models, general time-series foundation models, and ECG foundation model in comparison with time-series deep learning models in five different types of downstream tasks under the zero-shot, few-shot, and fine-tuning settings, including RR Interval Estimation, Age Estimation, Gender Classification, Potassium Abnormality Prediction and Arrhythmia Detection.

## Prepare Dataset
### Prepare ECG Data
* [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
### Prepare Subset Data and Label
* [Google Drive](https://drive.google.com/drive/folders/1IkHkwa0HUbxmieBHMPd-VRdYQJbKLm3P?usp=share_link)
  
We provide `.jsonl` file subset from the MIMIC-IV-ECG, along with the corresponding labels to evaluate in different downstream tasks, including RR Interval Estimation `rr_interval`, Age Estimation `age`, Gender Classification `gender`, Potassium Abnormality Prediction `flag`, and Arrhythmia Detection `report_label`.

## Installation
The required packages can be installed by running `pip install -r requirements.txt`.

For `ECG-FM` environment please refer the link [ECG-FM](https://github.com/bowang-lab/ECG-FM) and [fairseq-signals](https://github.com/Jwoo5/fairseq-signals).

## 🚀Quick Start
In `scripts` file we provide the shell scripts and change the `--task_name` to start evluating.
