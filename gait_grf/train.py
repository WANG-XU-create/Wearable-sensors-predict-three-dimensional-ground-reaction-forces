"""端到端训练/评估入口（tracer bullet，ticket #3）。

单入口命令：发现 trial 对 -> LOSO（折数=受试者数，每折测试集=留出受试者的
全部 trial，训练受试者内 trial 级随机留 15% 做验证）-> 训练最小 LTC ->
逐 trial 预测 -> 每折指标 + 预测对比图。

用法：
    python -m gait_grf.train --data-root data/subjectdata --out-dir runs/tracer

输出（写入 out-dir）：
    metrics.csv            每折一行：6 个 GRF 输出列的 RMSE(N) + 合成幅值
                           RMSE(N) + Pearson r + 测试规模
    summary.json           配置、每折 trial 明细与训练历史、跨折汇总
    predictions_<z>.png    每折一张代表 trial 的预测 vs 真实对比图
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .constants import (
    DEFAULT_REFINE_RADIUS,
    DEFAULT_STEP,
    DEFAULT_WINDOW,
    LEFT_FOOT_PLATE,
    SUBJECT_MAP,
    TARGET_COLS,
)
from .data import (
    GRFSequenceDataset,
    align,
    discover_trial_pairs,
    extract_features,
    extract_targets,
    read_qualisys,
    read_sensor,
    window_trial,
)
from .models import GaitLTC

# 规范化目标名（脚语义）：ground_force_left_vx ... ground_force_right_vz
TARGET_NAMES = list(TARGET_COLS)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject_trials(data_root, subjects=None):
    """按受试者组织 (sensor, qualisys, subject) trial 三元组；只保留有 trial 的受试者。"""
    trials = {}
    for z in sorted(SUBJECT_MAP):
        if subjects is not None and z not in subjects:
            continue
        pairs = discover_trial_pairs(data_root, subjects=[z])
        if pairs:
            trials[z] = pairs
    if not trials:
        raise SystemExit(f"在 {data_root} 下没有发现任何 trial 对")
    return trials


def split_val_trials(pairs, val_frac, rng):
    """trial 级随机划分 (fit, val)；val 数量 = round(n*val_frac)，可为空。"""
    n_val = int(round(len(pairs) * val_frac))
    if n_val == 0:
        return list(pairs), []
    idx = rng.permutation(len(pairs))
    val_idx = set(idx[:n_val].tolist())
    fit = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]
    return fit, val


def _batched_forward(model, X, device, batch_size):
    """对 (N, W, F) 数组分批前向，返回拼接后的 numpy (N, W, out)。"""
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            chunk = torch.from_numpy(X[i : i + batch_size]).to(device)
            outs.append(model(chunk).cpu().numpy())
    return np.concatenate(outs)


def train_one_fold(fit_pairs, val_pairs, cfg, device):
    """训练一折：fit trial 上拟合 scaler 并训练，val trial 上监控损失。"""
    fit_ds = GRFSequenceDataset(
        fit_pairs, window=cfg["window"], step=cfg["step"]
    )
    scalers = (fit_ds.feature_scaler, fit_ds.target_scaler)
    X = torch.from_numpy(fit_ds.X)
    y = torch.from_numpy(fit_ds.y)

    val_ds = None
    if val_pairs:
        val_ds = GRFSequenceDataset(
            val_pairs,
            window=cfg["window"],
            step=cfg["step"],
            feature_scaler=scalers[0],
            target_scaler=scalers[1],
        )

    model = GaitLTC(
        input_size=X.shape[-1],
        output_size=y.shape[-1],
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.MSELoss()

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(X))
        losses = []
        for i in range(0, len(X), cfg["batch_size"]):
            idx = perm[i : i + cfg["batch_size"]]
            pred = model(X[idx].to(device))
            loss = loss_fn(pred, y[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_ds is not None:
            model.eval()
            val_pred = _batched_forward(
                model, val_ds.X, device, cfg["batch_size"]
            )
            entry["val_loss"] = float(
                np.mean((val_pred - val_ds.y) ** 2)
            )
        history.append(entry)
    return model, scalers, history


def predict_trial(model, scalers, sensor_path, qualisys_path, subject, cfg, device):
    """对单个 trial 预测：对齐 -> 滑窗 -> 逐窗预测 -> 重叠帧平均 -> 反变换回 N。

    返回 (pred_N, true_N, n_windows)，数组长度为滑窗覆盖到的帧数
    （= window + (n_windows-1)*step，trial 尾部不足一个步进的帧不参与评估）。
    """
    feature_scaler, target_scaler = scalers
    sensor, qual, _ = align(
        read_sensor(sensor_path), read_qualisys(qualisys_path), cfg["refine_radius"]
    )
    feats = extract_features(sensor)
    targets = extract_targets(qual, left_plate=LEFT_FOOT_PLATE[subject])
    Xw, _ = window_trial(feats, targets, cfg["window"], cfg["step"])
    if len(Xw) == 0:
        return None

    n, w, f = Xw.shape
    Xs = (
        feature_scaler.transform(Xw.reshape(-1, f)).reshape(Xw.shape).astype(np.float32)
    )
    model.eval()
    pred_z = _batched_forward(model, Xs, device, cfg["batch_size"])
    pred_N = target_scaler.inverse_transform(
        pred_z.reshape(-1, pred_z.shape[-1])
    ).reshape(pred_z.shape)

    # 重叠帧平均：第 k 窗覆盖帧 [k*step, k*step+window)
    t_cov = w + (n - 1) * cfg["step"]
    acc = np.zeros((t_cov, pred_N.shape[-1]), dtype=np.float64)
    cnt = np.zeros(t_cov)
    for k in range(n):
        s = k * cfg["step"]
        acc[s : s + w] += pred_N[k]
        cnt[s : s + w] += 1
    pred_frame = acc / cnt[:, None]
    return pred_frame, targets[:t_cov], n


def _pearson(pred, true):
    """逐列 Pearson r；常量列（std≈0）记 0.0，保证指标有限。"""
    ps = pred.std(axis=0)
    ts = true.std(axis=0)
    if (ps < 1e-12).any() or (ts < 1e-12).any():
        return 0.0
    return float(np.corrcoef(pred.flatten(), true.flatten())[0, 1])


def fold_metrics(preds, trues):
    """preds/trues: 列表，每个元素 (T_i, 6) 的逐帧 N 数组。返回指标 dict。"""
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    m = {}
    for j, name in enumerate(TARGET_NAMES):
        m[f"rmse_N_{name}"] = float(np.sqrt(np.mean((pred[:, j] - true[:, j]) ** 2)))
        m[f"pearson_r_{name}"] = _pearson(pred[:, j : j + 1], true[:, j : j + 1])
    # 合成幅值：每足每帧的三维合力幅值 ||F||，跨双足与帧 pooled
    mag_pred = np.concatenate(
        [np.linalg.norm(p.reshape(-1, 2, 3), axis=2).ravel() for p in preds]
    )
    mag_true = np.concatenate(
        [np.linalg.norm(t.reshape(-1, 2, 3), axis=2).ravel() for t in trues]
    )
    m["rmse_N_resultant"] = float(np.sqrt(np.mean((mag_pred - mag_true) ** 2)))
    m["pearson_r_resultant"] = _pearson(
        mag_pred[:, None], mag_true[:, None]
    )
    return m


def plot_fold_prediction(out_path, pred, true, test_subject, trial_name):
    """一张代表 trial 的对比图：2 行（左右足）× 3 列（vx/vy/vz），真实 vs 预测。

    标签用 ASCII（渲染环境无 CJK 字体，避免缺字形告警与方框）。
    """
    axes_names = ("vx", "vy", "vz")
    feet = ("left foot", "right foot")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    t = np.arange(len(true)) / 100.0
    for foot in range(2):
        for ax_i in range(3):
            j = foot * 3 + ax_i
            ax = axes[foot, ax_i]
            ax.plot(t, true[:, j], color="tab:blue", label="true", lw=1.2)
            ax.plot(t, pred[:, j], color="tab:orange", label="pred", lw=1.0, alpha=0.9)
            ax.set_title(f"{feet[foot]} {axes_names[ax_i]}", fontsize=10)
            if foot == 1:
                ax.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"LOSO held-out {test_subject} | trial {trial_name} | predicted vs true GRF")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_loso(trials, cfg, device):
    """LOSO：折数=受试者数，每折测试集=留出受试者的全部 trial。

    返回 (fold_rows, fold_details, reps)：reps 是每折代表 trial（窗口数最多）
    的 (test_subject, trial_name, pred_N, true_N)，供绘图。
    """
    subjects = sorted(trials)
    fold_rows, fold_details, reps = [], [], []
    for fold, test_z in enumerate(subjects, start=1):
        train_pairs = [p for z in subjects if z != test_z for p in trials[z]]
        fit_pairs, val_pairs = split_val_trials(train_pairs, cfg["val_frac"], cfg["rng"])

        model, scalers, history = train_one_fold(fit_pairs, val_pairs, cfg, device)

        preds, trues, trial_details = [], [], []
        for sp, qp, z in trials[test_z]:
            result = predict_trial(model, scalers, sp, qp, z, cfg, device)
            if result is None:
                raise RuntimeError(f"trial {sp} 未产出任何窗口（长度不足 window）")
            pred, true, n_windows = result
            preds.append(pred)
            trues.append(true)
            trial_details.append(
                {
                    "sensor": os.path.basename(sp),
                    "qualisys": os.path.basename(qp),
                    "n_windows": int(n_windows),
                    "n_frames_evaluated": int(len(true)),
                }
            )

        row = {
            "fold": fold,
            "test_subject": test_z,
            "n_test_trials": len(preds),
            "n_test_windows": sum(d["n_windows"] for d in trial_details),
            **fold_metrics(preds, trues),
        }
        fold_rows.append(row)
        fold_details.append(
            {
                "fold": fold,
                "test_subject": test_z,
                "test_trials": trial_details,
                "n_val_trials": len(val_pairs),
                "history": history,
            }
        )
        # 代表 trial = 窗口数最多者
        rep_i = int(np.argmax([d["n_windows"] for d in trial_details]))
        reps.append(
            (test_z, trial_details[rep_i]["sensor"], preds[rep_i], trues[rep_i])
        )
    return fold_rows, fold_details, reps


def _aggregate(fold_rows):
    """跨折 mean/std 汇总（fold/test_subject/规模列除外）。"""
    df = pd.DataFrame(fold_rows)
    agg = {}
    for col in df.columns:
        if col in ("fold", "test_subject", "n_test_trials", "n_test_windows"):
            continue
        agg[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=0)),
        }
    return agg


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m gait_grf.train",
        description="可穿戴传感器 GRF 预测：LOSO 训练/评估入口",
    )
    parser.add_argument("--data-root", required=True, help="subjectdata 目录")
    parser.add_argument("--out-dir", required=True, help="输出目录（metrics/图/摘要）")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="受试者子集（z1..z8）；默认全部有数据的受试者",
    )
    parser.add_argument(
        "--model",
        default="ltc",
        choices=["ltc"],
        help="模型选择（LSTM/TCN 基线在 ticket #5 加入）",
    )
    parser.add_argument("--hidden", type=int, default=32, help="LTC 隐层大小")
    parser.add_argument("--layers", type=int, default=1, help="LTC 层数")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout 概率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="滑窗帧数")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP, help="滑窗步进")
    parser.add_argument("--val-frac", type=float, default=0.15, help="训练内验证 trial 比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--refine-radius", type=int, default=DEFAULT_REFINE_RADIUS, help="对齐精修半径"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="设备选择，默认自动",
    )
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)
    cfg = {
        "model": args.model,
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "window": args.window,
        "step": args.step,
        "val_frac": args.val_frac,
        "refine_radius": args.refine_radius,
        "seed": args.seed,
        "rng": np.random.default_rng(args.seed),
    }

    trials = load_subject_trials(args.data_root, args.subjects)
    subjects = sorted(trials)
    print(
        f"受试者 {len(subjects)} 人（{', '.join(subjects)}），"
        f"共 {sum(len(v) for v in trials.values())} 个 trial；设备 {device}"
    )

    fold_rows, fold_details, reps = run_loso(trials, cfg, device)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    pd.DataFrame(fold_rows).to_csv(metrics_path, index=False)

    summary = {
        "config": {k: v for k, v in cfg.items() if k != "rng"},
        "data_root": os.path.abspath(args.data_root),
        "subjects": subjects,
        "folds": fold_details,
        "aggregate": _aggregate(fold_rows),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 每折一张代表 trial（窗口数最多）的预测对比图
    for test_z, trial_name, pred, true in reps:
        plot_fold_prediction(
            os.path.join(args.out_dir, f"predictions_{test_z}.png"),
            pred,
            true,
            test_z,
            trial_name,
        )

    df = pd.DataFrame(fold_rows)
    print(f"\n指标已写入 {metrics_path}")
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
