"""评估指标（ticket #6）：N 与 %BW 归一化 + 辅助指标（峰值/冲量/峰值时机）。

纯函数模块，两处复用：
- ``train.fold_metrics`` 委托本模块（训练时直接产出全量指标）
- ``evaluate`` 后置入口（从 checkpoint 重新预测算指标，无需重训）

%BW 归一化：力除以受试者体重再乘 100（N -> %BW），使 vy（~550-830N 量级）
与 vx/vz（~10-40N）以及体重 55-83kg 的不同受试者之间可公平比较。
%BW 指标的数值即百分数（如 21.8 表示 21.8 %BW）。

辅助指标逐 trial 计算后取折内平均（峰值/冲量/时机是 trial 级量）：
- 峰值误差：|max(pred) - max(true)|，%BW
- 冲量误差：|∫pred dt - ∫true dt|，%BW·s（dt = 1/fs）
- 峰值时机：|argmax(pred) - argmax(true)|，帧（100Hz 下 1 帧 = 10ms）
"""

import numpy as np

from .constants import TARGET_COLS

FS = 100  # Hz，三套采集系统一致


def _pearson(pred, true):
    """逐列 Pearson r；常量列（std≈0）记 0.0，保证指标有限。"""
    if pred.std() < 1e-12 or true.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred.flatten(), true.flatten())[0, 1])


def fold_metrics(preds, trues, weight_N):
    """一折的全量指标。

    preds/trues: 列表，每个元素 (T_i, 6) 的逐帧 N 数组（同一 trial 逐帧对齐）。
    weight_N: 该折测试受试者的体重（N），用于 %BW 归一化。
    返回 dict：池化帧级指标（RMSE N/%BW、Pearson r、合成幅值）+
    逐 trial 辅助指标（峰值/冲量/峰值时机）的折内平均。
    """
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    dt = 1.0 / FS
    m = {}
    for j, name in enumerate(TARGET_COLS):
        err = pred[:, j] - true[:, j]
        m[f"rmse_N_{name}"] = float(np.sqrt(np.mean(err ** 2)))
        m[f"rmse_pctbw_{name}"] = m[f"rmse_N_{name}"] / weight_N * 100.0
        m[f"pearson_r_{name}"] = _pearson(pred[:, j : j + 1], true[:, j : j + 1])
    # 合成幅值：每足每帧的三维合力幅值 ||F||，跨双足与帧 pooled
    mag_pred = np.concatenate(
        [np.linalg.norm(p.reshape(-1, 2, 3), axis=2).ravel() for p in preds]
    )
    mag_true = np.concatenate(
        [np.linalg.norm(t.reshape(-1, 2, 3), axis=2).ravel() for t in trues]
    )
    m["rmse_N_resultant"] = float(np.sqrt(np.mean((mag_pred - mag_true) ** 2)))
    m["rmse_pctbw_resultant"] = m["rmse_N_resultant"] / weight_N * 100.0
    m["pearson_r_resultant"] = _pearson(mag_pred[:, None], mag_true[:, None])

    # 辅助指标：逐 trial 计算后取平均
    peak_err = {n: [] for n in TARGET_COLS}
    imp_err = {n: [] for n in TARGET_COLS}
    peak_lag = {n: [] for n in TARGET_COLS}
    for p, t in zip(preds, trues):
        for j, name in enumerate(TARGET_COLS):
            peak_err[name].append(abs(p[:, j].max() - t[:, j].max()))
            imp_err[name].append(abs(p[:, j].sum() - t[:, j].sum()) * dt)
            peak_lag[name].append(abs(int(p[:, j].argmax()) - int(t[:, j].argmax())))
    for name in TARGET_COLS:
        m[f"peak_err_pctbw_{name}"] = float(np.mean(peak_err[name])) / weight_N * 100.0
        m[f"impulse_err_pctbw_s_{name}"] = (
            float(np.mean(imp_err[name])) / weight_N * 100.0
        )
        m[f"peak_lag_frames_{name}"] = float(np.mean(peak_lag[name]))
    return m
