"""后置评估入口（ticket #6）：从 run 目录的 checkpoint 重新预测并算全量指标。

对已完成的训练运行（如 #4 的 runs/ltc_full_loso8）无需重训即可补出
%BW 归一化与辅助指标（峰值/冲量/峰值时机）。逐个加载 model_fold*.pt
（内含权重 + scaler + 配置 + 测试受试者），对留出受试者的全部 trial
重新预测（与训练时同一 LOSO 划分：测试集 = 留出受试者全部 trial）。

模型输入维度优先取 checkpoint config 内的 input_size（train 保存）；
旧 checkpoint 无该字段时按 config 的 feature_mode 经 features.feature_dim
推断（曾硬编码 120，导致 kinematic 51 维 checkpoint 无法后置评估）。

用法：
    python -m gait_grf.evaluate --run-dir runs/ltc_full_loso8 \
        --data-root data/subjectdata [--stitch uniform|hann|center]

--stitch 是纯评估期的重叠窗拼接方式（默认 uniform = 训练时口径），
用于免重训对比拼接对峰值误差的影响（见 train.stitch_windows）。

输出（写入 run-dir；stitch != uniform 时文件名带后缀）：
    metrics_full[_<stitch>].csv             每折一行的全量指标（列 =
                                            metrics.csv 扩充 %BW/辅助指标）
    metrics_full[_<stitch>]_aggregate.json  跨折 mean/std 汇总
也支持部分完成的运行（逐 checkpoint 处理，缺的折跳过）。
"""

import argparse
import glob
import json
import os
import re

import pandas as pd
import torch

from .constants import SUBJECT_WEIGHT_N
from .data import discover_trial_pairs
from .features import feature_dim
from .metrics import fold_metrics
from .models import make_model
from .train import STITCH_MODES, predict_trial


def evaluate_run(run_dir, data_root, device, stitch="uniform"):
    """评估 run 目录下全部 checkpoint，返回逐折指标 dict 列表。

    stitch 为重叠窗拼接方式（train.STITCH_MODES），作为评估期参数
    覆盖进各 checkpoint 的 cfg，不影响训练产物。
    """
    ckpts = sorted(glob.glob(os.path.join(run_dir, "model_fold*.pt")))
    if not ckpts:
        raise SystemExit(f"{run_dir} 下没有 model_fold*.pt checkpoint")
    rows = []
    for ck in ckpts:
        blob = torch.load(ck, map_location=device, weights_only=False)
        cfg = dict(blob["config"])
        cfg["stitch"] = stitch
        test_z = blob["test_subject"]
        model = make_model(
            cfg["model"],
            input_size=cfg.get("input_size", feature_dim(cfg.get("feature_mode", "raw"))),
            hidden=cfg["hidden"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
            kernel=cfg.get("kernel", 5),
        ).to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        scalers = (blob["feature_scaler"], blob["target_scaler"])

        trials = discover_trial_pairs(data_root, subjects=[test_z])
        if not trials:
            print(f"跳过 {os.path.basename(ck)}：{data_root} 下没有 {test_z} 的 trial")
            continue
        preds, trues, n_windows = [], [], 0
        for sp, qp, z in trials:
            result = predict_trial(model, scalers, sp, qp, z, cfg, device)
            if result is None:
                raise RuntimeError(f"trial {sp} 未产出任何窗口（长度不足 window）")
            pred, true, nw = result
            preds.append(pred)
            trues.append(true)
            n_windows += nw

        fold = int(re.search(r"fold(\d+)", os.path.basename(ck)).group(1))
        rows.append(
            {
                "fold": fold,
                "test_subject": test_z,
                "n_test_trials": len(preds),
                "n_test_windows": n_windows,
                **fold_metrics(preds, trues, SUBJECT_WEIGHT_N[test_z]),
            }
        )
        print(f"[{os.path.basename(ck)}] test={test_z}: {len(preds)} trials 已评估",
              flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m gait_grf.evaluate",
        description="从 run 目录 checkpoint 后置评估：全量指标（N/%BW/辅助）",
    )
    parser.add_argument("--run-dir", required=True, help="含 model_fold*.pt 的运行目录")
    parser.add_argument("--data-root", required=True, help="subjectdata 目录")
    parser.add_argument(
        "--stitch",
        default="uniform",
        choices=list(STITCH_MODES),
        help="重叠窗拼接方式：uniform=简单平均（默认，训练时口径）；hann=Hann"
             "加权；center=每帧取中心最近窗（拼接削峰对比实验）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出文件名（写入 run-dir）；默认 metrics_full.csv（stitch != "
             "uniform 时为 metrics_full_<stitch>.csv）",
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

    rows = evaluate_run(args.run_dir, args.data_root, device, stitch=args.stitch)
    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    out_name = args.out or (
        "metrics_full.csv" if args.stitch == "uniform"
        else f"metrics_full_{args.stitch}.csv"
    )
    out_path = os.path.join(args.run_dir, out_name)
    df.to_csv(out_path, index=False)

    agg = {
        c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))}
        for c in df.columns
        if c not in ("fold", "test_subject", "n_test_trials", "n_test_windows")
    }
    agg_path = out_path.replace(".csv", "_aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump({"n_folds_evaluated": len(df), "stitch": args.stitch,
                   "aggregate": agg}, f, ensure_ascii=False, indent=2)

    print(f"\n指标已写入 {out_path}（汇总 {agg_path}，stitch={args.stitch}）")
    show = ["fold", "test_subject",
            "rmse_pctbw_ground_force_left_vy", "rmse_pctbw_ground_force_right_vy",
            "peak_err_pctbw_ground_force_left_vy", "peak_err_pctbw_ground_force_right_vy",
            "rmse_pctbw_resultant", "pearson_r_resultant"]
    with pd.option_context("display.width", 250):
        print(df[show].round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
