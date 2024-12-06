import os
import wfdb
import ujson
import torch
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.signal import resample
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from statsmodels.tsa.seasonal import STL


def stl_resolve(data_raw, save_stl):

    if not os.path.exists(save_stl):
        os.mkdir(save_stl)

    trend_pk = save_stl + '/trend.pk'
    seasonal_pk = save_stl + '/seasonal.pk'
    resid_pk = save_stl + '/resid.pk'
    if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
        with open(trend_pk, 'rb') as f:
            trend_stamp = pickle.load(f)
        with open(seasonal_pk, 'rb') as f:
            seasonal_stamp = pickle.load(f)
        with open(resid_pk, 'rb') as f:
            resid_stamp = pickle.load(f)
    else:
        data_raw = np.array(data_raw)
        [n, m] = data_raw.shape

        trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
        seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
        resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

        data_raw = pd.DataFrame(data_raw)
        cols = data_raw.columns

        for i, col in enumerate(cols):
            df = data_raw[col]
            res = STL(df, period=50).fit()

            trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
            seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
            resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)

        with open(trend_pk, 'wb') as f:
            pickle.dump(trend_stamp, f)
        with open(seasonal_pk, 'wb') as f:
            pickle.dump(seasonal_stamp, f)
        with open(resid_pk, 'wb') as f:
            pickle.dump(resid_stamp, f)

    return trend_stamp, seasonal_stamp, resid_stamp


def load_data_tempo(args):
    with open(args.file_path, 'r') as f:
        data = [ujson.loads(line) for line in f]

    df = pd.json_normalize(data)
    df = df[["subject_id", "ecg_path", args.label]]

    features = []
    sub_num = []

    # Step 1: Collect signals for standardization
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECG data"):
        sub_num.append(int(row['subject_id']))
        if len(list(set(sub_num))) > (30000):
            break
        ecg_path = row['ecg_path']
        if args.task_name == "classification":
            label_num = int(row[args.label])
        else:
            label_num = np.float32(row[args.label])
            if np.isclose(label_num, 29999.0):
                continue

        full_path = ecg_path

        Transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        try:
            signal_data = wfdb.rdsamp(full_path)[0]
            signal_data = Transforms(signal_data)
            signal_data = np.squeeze(signal_data, axis=0)
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply Z-score normalization using fitted scaler
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            if std != 0:
                signal_data = (signal_data - mean) / std
            else:
                signal_data = signal_data - mean

            # Select specific feature channel using feat_id
            feat_id = index % signal_data.shape[1]
            # signal_data = signal_data[:, feat_id:feat_id+1]

            if args.downsample_size is not None:
                signal_data = resample(signal_data, args.downsample_size, axis=0)

            # Perform STL decomposition
            trend_stamp, seasonal_stamp, resid_stamp = stl_resolve(signal_data, save_stl=f"./stl/{ecg_path[-8:]}")

            feature_dict = {
                "subject_id": int(row['subject_id']),
                "signal_data": signal_data[:, feat_id:feat_id+1],
                "label": label_num,
                "trend_stamp": trend_stamp[:, feat_id:feat_id+1],
                "seasonal_stamp": seasonal_stamp[:, feat_id:feat_id+1],
                "resid_stamp": resid_stamp[:, feat_id:feat_id+1]
            }

            features.append(feature_dict)

        except Exception as e:
            print(f"Error reading file at {full_path}: {e}")

    features_df = pd.DataFrame(features)
    print(features_df)

    return features_df


def load_data(args):

    with open(args.file_path, 'r') as f:
        data = [ujson.loads(line) for line in f]

    df = pd.json_normalize(data)
    df = df[["subject_id", "ecg_path", args.label]]

    features = []
    sub_num = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECG data"):
        sub_num.append(int(row['subject_id']))
        if len(list(set(sub_num))) > (10):
            break
        ecg_path = row['ecg_path']
        if args.task_name == "classification":
            label_num = int(row[args.label])
        else:
            label_num = np.float32(row[args.label])
            if np.isclose(label_num, 29999.0):
                continue

        full_path = ecg_path

        Transforms = transforms.Compose([
                            transforms.ToTensor(),
                        ])
        
        try:
            signal_data = wfdb.rdsamp(full_path)[0]
            signal_data = Transforms(signal_data)
            signal_data = np.squeeze(signal_data, axis=0)
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply Z-score normalization
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            if std != 0:
                signal_data = (signal_data - mean) / std
            else:
                signal_data = signal_data - mean
            
            if args.downsample_size is not None:
                signal_data = resample(signal_data, args.downsample_size, axis=0)

            feature_dict = {
                "subject_id": int(row['subject_id']),
                "signal_data": signal_data,
                "label": label_num,
            }

            features.append(feature_dict)
            
        except Exception as e:
            print(f"Error reading file at {full_path}: {e}")

    features_df = pd.DataFrame(features)

    print(features_df)

    return features_df


def load_data_tsdl(args):
    with open(args.file_path, 'r') as f:
        data = [ujson.loads(line) for line in f]

    df = pd.json_normalize(data)
    df = df[["subject_id", "ecg_path", args.label]]

    features = []
    sub_num = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECG data"):
        sub_num.append(int(row['subject_id']))
        if len(list(set(sub_num))) > (30000):
            break
        ecg_path = row['ecg_path']
        if args.task_name == "classification":
            label_num = int(row[args.label])
        else:
            label_num = np.float32(row[args.label])
            if np.isclose(label_num, 29999.0):
                continue

        full_path = ecg_path

        Transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        try:
            signal_data = wfdb.rdsamp(full_path)[0]
            signal_data = Transforms(signal_data)
            signal_data = np.squeeze(signal_data, axis=0)
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply Z-score normalization using fitted scaler
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            if std != 0:
                signal_data = (signal_data - mean) / std
            else:
                signal_data = signal_data - mean

            # Select specific feature channel using feat_id
            feat_id = index % signal_data.shape[1]

            if args.downsample_size is not None:
                signal_data = resample(signal_data, args.downsample_size, axis=0)

            feature_dict = {
                "subject_id": int(row['subject_id']),
                "signal_data": signal_data[:, feat_id:feat_id+1],
                "label": label_num
            }

            features.append(feature_dict)

        except Exception as e:
            print(f"Error reading file at {full_path}: {e}")

    features_df = pd.DataFrame(features)
    print(features_df)

    return features_df


class ECGDataset(Dataset):
    def __init__(self, features_df):
        self.features_df = features_df
    
    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, idx):
        row = self.features_df.iloc[idx]

        signal_data = torch.tensor(row['signal_data'], dtype=torch.float32)

        if np.issubdtype(type(row['label']), np.integer):
            label = torch.tensor(row['label'], dtype=torch.long).squeeze()
        else:
            label = torch.tensor(row['label'], dtype=torch.float32)

        return signal_data, label


class ECGDataset_tempo(Dataset):
    def __init__(self, features_df):
        self.features_df = features_df

    def __len__(self):
        return len(self.features_df)

    def __getitem__(self, idx):
        row = self.features_df.iloc[idx]

        signal_data = torch.tensor(row['signal_data'], dtype=torch.float32)

        if np.issubdtype(type(row['label']), np.integer):
            label = torch.tensor(row['label'], dtype=torch.long).squeeze()
        else:
            label = torch.tensor(row['label'], dtype=torch.float32)

        trend_stamp = torch.tensor(row['trend_stamp'], dtype=torch.float32)
        seasonal_stamp = torch.tensor(row['seasonal_stamp'], dtype=torch.float32)
        resid_stamp = torch.tensor(row['resid_stamp'], dtype=torch.float32)

        return signal_data, label, trend_stamp, seasonal_stamp, resid_stamp
