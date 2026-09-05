"""线性探针：LOSO Ridge 快速对比特征模式的跨受试者可迁移性（ticket #7）。

LTC 全量一轮 ~4h，改特征前先用帧级线性模型验证表征可迁移性（tracer 阶段
即用此方法确认过板↔脚翻转问题）。线性模型无时序状态，直接用对齐后的逐帧
(feature, target) 样本，不做滑窗。

用法：
    python -m gait_grf.probe --data-root data/subjectdata \
        --modes raw kinematic kinematic_min [--subjects z1 z2 ...]

输出：每个特征模式的 LOSO 各折 vy/resultant Pearson r 与跨折 mean±std。
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .constants import SUBJECT_MAP, SUBJECT_WEIGHT_N
from .data import discover_trial_pairs, load_aligned_trial
from .features import FEATURE_MODES


def collect_frames(data_root, feature_mode, subjects=None):
    """读取全部 trial 的对齐帧级样本：X (N,D), y (N,6), subj (N,)。"""
    Xs, ys, zs = [], [], []
    for z in sorted(SUBJECT_MAP):
        if subjects is not None and z not in subjects:
            continue
        for sp, qp, zz in discover_trial_pairs(data_root, subjects=[z]):
            f, t = load_aligned_trial(sp, qp, zz, feature_mode=feature_mode)
            Xs.append(f)
            ys.append(t)
            zs.append(np.full(len(f), zz))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(zs)


def resultants(y):
    return np.linalg.norm(y, axis=1)


def run_probe(data_root, modes, subjects=None, alpha=1e3):
    rows = []
    for mode in modes:
        t0 = time.time()
        X, y, subj = collect_frames(data_root, mode, subjects)
        load_s = time.time() - t0
        all_subj = sorted(set(subj.tolist()))
        fold_r = []
        for test_z in all_subj:
            fit_m = subj != test_z
            scaler = StandardScaler().fit(X[fit_m])
            model = Ridge(alpha=alpha).fit(scaler.transform(X[fit_m]), y[fit_m])
            pred = model.predict(scaler.transform(X[~fit_m]))
            true = y[~fit_m]
            r = {
                col: float(np.corrcoef(pred[:, j], true[:, j])[0, 1])
                for j, col in enumerate(
                    ["L_vx", "L_vy", "L_vz", "R_vx", "R_vy", "R_vz"]
                )
            }
            r["resultant"] = float(
                np.corrcoef(resultants(pred), resultants(true))[0, 1]
            )
            r["test_subject"] = test_z
            r["n_test_frames"] = int((~fit_m).sum())
            fold_r.append(r)
        df = pd.DataFrame(fold_r)
        summary = {
            "mode": mode,
            "n_features": X.shape[1],
            "n_frames": len(X),
            "load_sec": round(load_s, 1),
            **{
                c: (float(df[c].mean()), float(df[c].std(ddof=0)))
                for c in df.columns
                if c not in ("test_subject", "n_test_frames")
            },
        }
        rows.append((summary, df))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m gait_grf.probe",
        description="LOSO 线性探针：对比特征模式的跨受试者可迁移性",
    )
    parser.add_argument("--data-root", required=True, help="subjectdata 目录")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["raw", "kinematic", "kinematic_min"],
        choices=list(FEATURE_MODES),
    )
    parser.add_argument("--subjects", nargs="+", default=None, help="受试者子集 z1..z8")
    parser.add_argument("--alpha", type=float, default=1e3, help="Ridge 正则强度")
    args = parser.parse_args(argv)

    rows = run_probe(args.data_root, args.modes, args.subjects, alpha=args.alpha)
    for summary, df in rows:
        print(
            f"\n=== {summary['mode']} ({summary['n_features']} 维, "
            f"{summary['n_frames']} 帧, 载入 {summary['load_sec']}s) ==="
        )
        with pd.option_context("display.width", 200):
            print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("跨折 mean±std：")
        for k, v in summary.items():
            if isinstance(v, tuple):
                print(f"  {k}: {v[0]:.4f}±{v[1]:.4f}")


if __name__ == "__main__":
    main()
