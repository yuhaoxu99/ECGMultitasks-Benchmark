# ECGMultitasks-Benchmark
Official code for "Are Foundation Models Useful for Electrocardiogram Analysis? A Multi-task Benchmark with Comprehensive Evaluations and Insightful Findings".

## Prepare Dataset
### Prepare ECG Data
* [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
### Prepare Subset Data and Label
We provide `.jsonl` file subset from the MIMIC-IV-ECG.
* [Google Drive](https://drive.google.com/drive/folders/1IkHkwa0HUbxmieBHMPd-VRdYQJbKLm3P?usp=share_link)

## Installation
The required packages can be installed by running `pip install -r requirements.txt`.

For `ECG-FM` environment please refer the link [ECG-FM](https://github.com/bowang-lab/ECG-FM) and [fairseq-signals](https://github.com/Jwoo5/fairseq-signals).

## 🚀Quick Start
In `scripts` file we provide the shell scripts and change the `--task_name` to start evluating.
