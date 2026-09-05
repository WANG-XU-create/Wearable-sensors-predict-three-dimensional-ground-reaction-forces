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
    DEFAULT_REFINE_RADIUS,
    DEFAULT_STEP,
    DEFAULT_WINDOW,
    FEATURE_COLS,
    INVALID_TRIALS,
    LEFT_FOOT_PLATE,
    SUBJECT_MAP,
)
from .features import FEATURE_MODES, derive_kinematic_features, kinematic_feature_names


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
    """双板垂直 GRF 差（f1_vy − f2_vy）。Qualisys 中 y 轴朝上（vy 为垂直分量）。

    注意：z6–z8 左脚踩板2，该差值符号与 z1–z5 相反；对齐只用 |相关| 峰，
    符号无关紧要。
    """
    return (
        qualisys_df["ground_force_1_vy"].to_numpy(dtype=float)
        - qualisys_df["ground_force_2_vy"].to_numpy(dtype=float)
    )


def _onset(arr, frac=0.2, min_run=5):
    """首个持续超过 frac*max 并保持 min_run 帧的索引；检不出返回 None。"""
    thr = frac * np.nanmax(arr)
    for i in range(len(arr) - min_run + 1):
        if (arr[i : i + min_run] > thr).all():
            return i
    return None


def _onset_anchor_lag(sensor_df, qualisys_df):
    """用「左足压力首次抬升 ↔ 首块被踩板 vy 首次抬升」估计滞后。

    采集协议：所有 trial 均左脚先运动再右脚，且左脚首次落板必踩在板上，
    因此该事件在两套系统中对应同一物理时刻，不受步态周期性影响。
    返回 d（sensor[i] ~ qualisys[i+d]）；检不出返回 None。
    """
    i_sensor = _onset(sensor_df["left_pressure_sum"].to_numpy(dtype=float))
    plate_onsets = [
        _onset(qualisys_df[f"ground_force_{p}_vy"].to_numpy(dtype=float))
        for p in ("1", "2")
    ]
    plate_onsets = [i for i in plate_onsets if i is not None]
    if i_sensor is None or not plate_onsets:
        return None
    return min(plate_onsets) - i_sensor


def _local_corr_lag(p, g, center, radius):
    """在 [center-radius, center+radius] 内做归一化互相关，取绝对值最大的滞后。"""
    p = (p - p.mean()) / (p.std() + 1e-9)
    g = (g - g.mean()) / (g.std() + 1e-9)
    best, best_d = -np.inf, center
    for d in range(center - radius, center + radius + 1):
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
    return best_d


def find_lag(sensor_df, qualisys_df, refine_radius=DEFAULT_REFINE_RADIUS):
    """返回 d，使得 sensor[i] ~ qualisys[i+d]（d > 0 表示 sensor 超前）。

    两步估计：
    1. onset 锚定：左足压力首次抬升 ↔ 首块被踩板 vy 首次抬升。步态信号周期
       ~1s，在 ~2s 的短 trial 上做宽范围互相关会在 ±1 个步态周期处出现假峰，
       因此先用首触板事件把滞后锚定住。
    2. 局部精修：在锚点 ±refine_radius 内对「左右足压力差 ↔ 双板 vy 差」做
       归一化互相关（绝对值最大），消除 onset 阈值带来的系统偏差。
    onset 检不出时按 lag=0 处理并告警。
    """
    anchor = _onset_anchor_lag(sensor_df, qualisys_df)
    if anchor is None:
        warnings.warn(
            "onset 锚定失败（未检出左足压力抬升或力板加载），按 lag=0 处理",
            stacklevel=2,
        )
        return 0
    return _local_corr_lag(
        pressure_diff(sensor_df),
        vertical_grf_diff(qualisys_df),
        anchor,
        refine_radius,
    )


def align(sensor_df, qualisys_df, refine_radius=DEFAULT_REFINE_RADIUS):
    """对齐 sensor 与 qualisys，返回等长的 (sensor_df, qualisys_df, lag)。"""
    d = find_lag(sensor_df, qualisys_df, refine_radius)
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


def extract_features(sensor_df, feature_mode="raw"):
    """按特征模式提取输入特征。

    - raw：原始 120 维（28 四元数 + 90 压力 + 2 和），即 ltc_full 基线的输入；
    - kinematic / kinematic_min：features.py 的运动学特征前端（基线相对运动
      + 压力摘要），见 features 模块 docstring。
    """
    if feature_mode == "raw":
        return _clean_array(sensor_df[FEATURE_COLS].to_numpy(dtype=float))
    if feature_mode in FEATURE_MODES:
        return derive_kinematic_features(sensor_df, mode=feature_mode)
    raise ValueError(f"未知特征模式 {feature_mode!r}，可选：{FEATURE_MODES}")


def extract_targets(qualisys_df, left_plate=1):
    """提取目标并规范化为脚语义列序：前 3 列=左脚、后 3 列=右脚。

    z1–z5 左脚踩板1（列序 ground_force_1,2 原样）；z6–z8 左脚踩板2，
    交换两组列，使输出列序与受试者无关（否则左右脚语义随受试者组翻转，
    共享模型会学到两套矛盾映射的平均）。
    """
    first, second = ("1", "2") if left_plate == 1 else ("2", "1")
    cols = [f"ground_force_{p}_{a}" for p in (first, second) for a in ("vx", "vy", "vz")]
    return _clean_array(qualisys_df[cols].to_numpy(dtype=float))


# 对齐结果缓存：LOSO 每折都要用同一批 trial，对齐是确定性的，按
# (路径对, refine_radius, left_plate, feature_mode) 缓存后 8 折只做 1 次磁盘
# 读取+对齐（224 次而不是 1792 次）。
_TRIAL_CACHE = {}


def clear_trial_cache():
    _TRIAL_CACHE.clear()


def load_aligned_trial(
    sensor_path, qualisys_path, subject, refine_radius=DEFAULT_REFINE_RADIUS, feature_mode="raw"
):
    """读取+对齐+提取单个 trial，返回 (features, targets)，带缓存。"""
    key = (
        os.path.abspath(sensor_path),
        os.path.abspath(qualisys_path),
        refine_radius,
        LEFT_FOOT_PLATE[subject],
        feature_mode,
    )
    if key not in _TRIAL_CACHE:
        sensor, qual, _ = align(
            read_sensor(sensor_path), read_qualisys(qualisys_path), refine_radius
        )
        _TRIAL_CACHE[key] = (
            extract_features(sensor, feature_mode=feature_mode),
            extract_targets(qual, left_plate=LEFT_FOOT_PLATE[subject]),
        )
    return _TRIAL_CACHE[key]


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
    """按受试者+序号发现 trial，返回 (sensor_path, qualisys_path, subject) 三元组。

    subject 供目标列的板->脚规范化使用（LEFT_FOOT_PLATE）。subjects 为 z{n} 集合。
    命中 INVALID_TRIALS（如 LQW03/04 的 IMU 全零文件）的 trial 连同其测力台
    数据一并排除，不计入训练/评估。
    """
    pairs = []
    skipped = []
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
            if (z, code) in INVALID_TRIALS:
                skipped.append((z, code))
                continue
            qf = os.path.join(qual_dir, f"z{n}_{code}_mot_100Hz.csv")
            if os.path.isfile(qf):
                pairs.append((sf, qf, z))
    if skipped:
        warnings.warn(
            f"排除无效 trial（IMU 数据缺失，见 constants.INVALID_TRIALS）：{skipped}",
            stacklevel=2,
        )
    return pairs


class GRFSequenceDataset(Dataset):
    """给定 (sensor_path, qualisys_path, subject) trial 三元组列表，产出对齐、
    滑窗、目标脚语义规范化、归一化的序列。"""

    def __init__(
        self,
        trial_pairs,
        window=DEFAULT_WINDOW,
        step=DEFAULT_STEP,
        refine_radius=DEFAULT_REFINE_RADIUS,
        feature_scaler=None,
        target_scaler=None,
        feature_mode="raw",
    ):
        self.window, self.step, self.refine_radius = window, step, refine_radius
        Xs, ys = [], []
        for sp, qp, z in trial_pairs:
            f, t = load_aligned_trial(sp, qp, z, refine_radius, feature_mode)
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
