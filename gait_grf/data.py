"""数据管线：读取 -> 跨系统对齐 -> 特征/目标提取 -> 滑窗 -> 归一化 -> Dataset。"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .constants import (
    DEFAULT_MAX_LAG,
    DEFAULT_STEP,
    DEFAULT_WINDOW,
    FEATURE_COLS,
    SUBJECT_MAP,
    TARGET_COLS,
)


def read_sensor(path):
    return pd.read_csv(path)


def read_qualisys(path):
    return pd.read_csv(path)


def pressure_diff(sensor_df):
    """左右足压力差（left − right），对齐信号之一。"""
    return (
        sensor_df["left_pressure_sum"].to_numpy(dtype=float)
        - sensor_df["right_pressure_sum"].to_numpy(dtype=float)
    )


def vertical_grf_diff(qualisys_df):
    """双板垂直 GRF 差（f1_vy − f2_vy）。Qualisys 中 y 轴朝上（vy 为垂直分量）。"""
    return (
        qualisys_df["ground_force_1_vy"].to_numpy(dtype=float)
        - qualisys_df["ground_force_2_vy"].to_numpy(dtype=float)
    )


def find_lag(sensor_df, qualisys_df, max_lag=DEFAULT_MAX_LAG):
    """返回 d，使得 sensor[i] ~ qualisys[i+d]。

    d > 0 表示 sensor 超前（leads）；d < 0 表示 sensor 滞后（lags）。
    用「左右足压力差 ↔ 双板垂直 GRF 差」的归一化互相关（取绝对值最大）估计滞后；
    两足在摆动相/支撑相各自振荡，差信号比「双足之和」更适合定位。
    """
    p = pressure_diff(sensor_df)
    g = vertical_grf_diff(qualisys_df)
    p = (p - p.mean()) / (p.std() + 1e-9)
    g = (g - g.mean()) / (g.std() + 1e-9)
    best_d, best = 0, -np.inf
    for d in range(-max_lag, max_lag + 1):
        if d >= 0:
            a = p[: len(p) - d] if d > 0 else p
            b = g[d:]
        else:
            k = -d
            a = p[k:]
            b = g[: len(g) - k]
        n = min(len(a), len(b))
        if n < 10:
            continue
        score = abs(float((a[:n] * b[:n]).mean()))
        if score > best:
            best, best_d = score, d
    if abs(best_d) == max_lag:
        warnings.warn(
            f"互相关滞后卡在搜索边界 ±{max_lag}，真实滞后可能在范围外",
            stacklevel=2,
        )
    return best_d


def align(sensor_df, qualisys_df, max_lag=DEFAULT_MAX_LAG):
    """对齐 sensor 与 qualisys，返回等长的 (sensor_df, qualisys_df, lag)。"""
    d = find_lag(sensor_df, qualisys_df, max_lag)
    s = sensor_df.reset_index(drop=True)
    q = qualisys_df.reset_index(drop=True)
    if d >= 0:
        s = s.iloc[: len(s) - d] if d > 0 else s
        q = q.iloc[d:]
    else:
        k = -d
        s = s.iloc[k:]
        q = q.iloc[: len(q) - k]
    n = min(len(s), len(q))
    return s.iloc[:n].reset_index(drop=True), q.iloc[:n].reset_index(drop=True), d


def _clean_array(a):
    df = pd.DataFrame(a)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.ffill().bfill()
    return df.to_numpy(dtype=np.float32)


def extract_features(sensor_df):
    return _clean_array(sensor_df[FEATURE_COLS].to_numpy(dtype=float))


def extract_targets(qualisys_df):
    return _clean_array(qualisys_df[TARGET_COLS].to_numpy(dtype=float))


def window_trial(features, targets, window=DEFAULT_WINDOW, step=DEFAULT_STEP):
    """把单次 trial 的 (features, targets) 切分成滑窗序列（不跨 trial）。"""
    n = len(features)
    if n < window:
        return (
            np.empty((0, window, features.shape[1]), dtype=np.float32),
            np.empty((0, window, targets.shape[1]), dtype=np.float32),
        )
    count = (n - window) // step + 1
    X = np.stack([features[i * step : i * step + window] for i in range(count)])
    y = np.stack([targets[i * step : i * step + window] for i in range(count)])
    return X.astype(np.float32), y.astype(np.float32)


def discover_trial_pairs(subjectdata_root, subjects=None):
    """按受试者+序号发现 (sensor_path, qualisys_path) 配对。subjects 为 z{n} 集合。"""
    pairs = []
    for z, initials in SUBJECT_MAP.items():
        if subjects is not None and z not in subjects:
            continue
        n = z[1:]
        sensor_dir = os.path.join(subjectdata_root, "sensor", initials)
        qual_dir = os.path.join(subjectdata_root, "Qualisys", f"Z{n}_csv_calibrated")
        if not os.path.isdir(sensor_dir) or not os.path.isdir(qual_dir):
            continue
        for sf in sorted(glob.glob(os.path.join(sensor_dir, f"{initials}*.csv"))):
            code = os.path.basename(sf)[len(initials):].replace(".csv", "")
            qf = os.path.join(qual_dir, f"z{n}_{code}_mot_100Hz.csv")
            if os.path.isfile(qf):
                pairs.append((sf, qf))
    return pairs


class GRFSequenceDataset(Dataset):
    """给定 (sensor_path, qualisys_path) trial 对列表，产出对齐、滑窗、归一化的序列。"""

    def __init__(
        self,
        trial_pairs,
        window=DEFAULT_WINDOW,
        step=DEFAULT_STEP,
        max_lag=DEFAULT_MAX_LAG,
        feature_scaler=None,
        target_scaler=None,
    ):
        self.window, self.step, self.max_lag = window, step, max_lag
        Xs, ys = [], []
        for sp, qp in trial_pairs:
            sensor = read_sensor(sp)
            qual = read_qualisys(qp)
            sensor, qual, _ = align(sensor, qual, max_lag)
            f = extract_features(sensor)
            t = extract_targets(qual)
            X, y = window_trial(f, t, window, step)
            if len(X):
                Xs.append(X)
                ys.append(y)

        if not Xs:
            raise ValueError(
                f"没有产出任何序列：window={window} 大于所有 trial 的对齐后长度"
            )

        Xall = np.concatenate(Xs)
        yall = np.concatenate(ys)

        self.feature_scaler = (
            feature_scaler
            if feature_scaler is not None
            else StandardScaler().fit(Xall.reshape(-1, Xall.shape[-1]))
        )
        self.target_scaler = (
            target_scaler
            if target_scaler is not None
            else StandardScaler().fit(yall.reshape(-1, yall.shape[-1]))
        )
        self.X = (
            self.feature_scaler.transform(Xall.reshape(-1, Xall.shape[-1]))
            .reshape(Xall.shape)
            .astype(np.float32)
        )
        self.y = (
            self.target_scaler.transform(yall.reshape(-1, yall.shape[-1]))
            .reshape(yall.shape)
            .astype(np.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
