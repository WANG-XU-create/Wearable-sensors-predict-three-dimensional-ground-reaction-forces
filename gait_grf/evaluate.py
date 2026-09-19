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
        --data-root data/subjectdata [--stitch uniform|hann|center] \
        [--scaler-refit none|test] [--per-trial] \
        [--ensemble runs/other_seed1,runs/other_seed2] [--include-invalid] \
        [--plate-zero auto|on|off]

--stitch 是纯评估期的重叠窗拼接方式（默认 uniform = 训练时口径），
用于免重训对比拼接对峰值误差的影响（见 train.stitch_windows）。

--scaler-refit test 为 A5 测试受试者 transductive 自校准（无标签，部署合法）：
feature scaler 用留出受试者本人的全部对齐帧重新 fit（mean+std 全换），
target scaler 与模型权重不动——度量域偏移中「特征分布线性漂移」的可补偿部分。
已知代价：压力幅值类特征的绝对水平信息（如体重差异）会被一并归一化掉，
逐轴结果需对比原口径解读。

--per-trial 额外输出逐 trial 指标表（数据诊断用：如 z7 鞋垫故障 trial 的
残差定位、右足 vx 逐 trial 排查）。

--ensemble 逗号分隔的附加 run 目录：同名折 checkpoint 的逐 trial 预测
（N 空间、拼接后）与主 run 平均，指标算在平均预测上（多种子 ensemble 口径）。
成员的数据/架构口径字段必须与主 run 一致（feature_mode/window/step/
refine_radius/plate_zero/model/hidden/layers/input_size）。

--include-invalid 显式包含 INVALID_TRIALS（z1 的 LQW03/04、z7 的 ZWJ10–20
鞋垫故障 trial），仅供数据诊断（配合 --per-trial 定位故障 trial 残差），
不用于论文口径。

--plate-zero 覆盖评估期力板校零：auto（默认）按各 checkpoint 训练时的
config（旧 checkpoint 无该字段 -> False）；on/off 强制统一。与训练口径
不一致时告警（目标与预测的零点约定将错位）。

输出（写入 run-dir；stitch != uniform / refit != none 时文件名带后缀）：
    metrics_full[_<stitch>][_refit].csv             每折一行的全量指标（列 =
                                                    metrics.csv 扩充 %BW/辅助指标）
    metrics_full[_<stitch>][_refit]_aggregate.json  跨折 mean/std 汇总
    metrics_per_trial[_<stitch>]...csv              --per-trial 时的逐 trial 指标表
也支持部分完成的运行（逐 checkpoint 处理，缺的折跳过）。
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .constants import SUBJECT_WEIGHT_N
from .data import discover_trial_pairs, load_aligned_trial
from .features import feature_dim
from .metrics import fold_metrics
from .models import make_model
from .train import STITCH_MODES, predict_trial

# ensemble 成员必须一致的口径字段（数据 + 架构；不一致时平均无意义）
_ENSEMBLE_CFG_KEYS = (
    "feature_mode", "window", "step", "refine_radius",
    "model", "hidden", "layers", "input_size", "plate_zero",
)


def _resolved_input_size(cfg):
    return cfg.get("input_size", feature_dim(cfg.get("feature_mode", "raw")))


def _build_model(blob, device):
    """按 checkpoint 重建评估模型（eval 模式），返回 (model, cfg)。"""
    cfg = dict(blob["config"])
    model = make_model(
        cfg["model"],
        input_size=_resolved_input_size(cfg),
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        dropout=cfg["dropout"],
        kernel=cfg.get("kernel", 5),
        ode_unfolds=cfg.get("ode_unfolds", 6),
        attn_heads=cfg.get("attn_heads", 8),
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg


def evaluate_run(
    run_dir,
    data_root,
    device,
    stitch="uniform",
    scaler_refit="none",
    per_trial=False,
    ensemble_dirs=(),
    include_invalid=False,
    plate_zero="auto",
):
    """评估 run 目录下全部 checkpoint，返回 (逐折指标 rows, 逐 trial rows)。

    stitch 为重叠窗拼接方式（train.STITCH_MODES），作为评估期参数
    覆盖进各 checkpoint 的 cfg，不影响训练产物。
    scaler_refit="test" 时（A5），每折的 feature scaler 在留出受试者
    本人的全部对齐帧上重新 fit（load_aligned_trial 有进程内缓存，
    后续 predict_trial 不重复计算）。
    per_trial 额外收集逐 trial 指标行（数据诊断口径）。
    ensemble_dirs 为附加 run 目录（多种子 ensemble，见模块 docstring）。
    plate_zero: "auto" 按各 checkpoint 训练口径，"on"/"off" 强制覆盖。
    """
    ckpts = sorted(glob.glob(os.path.join(run_dir, "model_fold*.pt")))
    if not ckpts:
        raise SystemExit(f"{run_dir} 下没有 model_fold*.pt checkpoint")
    rows, trial_rows = [], []
    for ck in ckpts:
        blob = torch.load(ck, map_location=device, weights_only=False)
        cfg = dict(blob["config"])
        cfg["stitch"] = stitch
        test_z = blob["test_subject"]
        # 力板校零：训练口径缺省 False（旧 checkpoint），CLI 可强制覆盖
        trained_zero = bool(cfg.get("plate_zero", False))
        eval_zero = {"on": True, "off": False}.get(plate_zero, trained_zero)
        if eval_zero != trained_zero:
            print(
                f"[{os.path.basename(ck)}] 警告：评估期 plate_zero={eval_zero} 与"
                f"训练口径 {trained_zero} 不一致，目标零点约定将错位",
                flush=True,
            )
        cfg["plate_zero"] = eval_zero

        model, _ = _build_model(blob, device)
        feature_scaler, target_scaler = blob["feature_scaler"], blob["target_scaler"]

        trials = discover_trial_pairs(
            data_root, subjects=[test_z], include_invalid=include_invalid
        )
        if not trials:
            print(f"跳过 {os.path.basename(ck)}：{data_root} 下没有 {test_z} 的 trial")
            continue
        if scaler_refit == "test":
            feats = [
                load_aligned_trial(
                    sp, qp, z, cfg["refine_radius"], cfg.get("feature_mode", "raw"),
                    plate_zero=eval_zero,
                )[0]
                for sp, qp, z in trials
            ]
            feature_scaler = StandardScaler().fit(np.concatenate(feats, axis=0))
        members = [(model, (feature_scaler, target_scaler), cfg)]

        # ensemble 成员：同名折 checkpoint，口径字段必须与主 run 一致
        for d in ensemble_dirs:
            ck_m = os.path.join(d, os.path.basename(ck))
            if not os.path.isfile(ck_m):
                raise SystemExit(
                    f"ensemble 成员 {d} 缺少折 checkpoint {os.path.basename(ck)}"
                )
            blob_m = torch.load(ck_m, map_location=device, weights_only=False)
            if blob_m["test_subject"] != test_z:
                raise SystemExit(
                    f"ensemble 成员 {d} 的 {os.path.basename(ck_m)} 测试受试者"
                    f"({blob_m['test_subject']}) 与主 run ({test_z}) 不一致"
                )
            model_m, cfg_m = _build_model(blob_m, device)
            for k in _ENSEMBLE_CFG_KEYS:
                v_m = bool(cfg_m.get(k, False)) if k == "plate_zero" else cfg_m.get(k)
                v_0 = bool(cfg.get(k, False)) if k == "plate_zero" else cfg.get(k)
                if k == "input_size":
                    v_m, v_0 = _resolved_input_size(cfg_m), _resolved_input_size(cfg)
                if v_m != v_0:
                    raise SystemExit(
                        f"ensemble 成员 {d} 的 {k}={v_m!r} 与主 run {v_0!r} 不一致，"
                        f"平均不同口径的预测无意义"
                    )
            cfg_m["stitch"] = stitch
            cfg_m["plate_zero"] = eval_zero
            members.append(
                (model_m, (blob_m["feature_scaler"], blob_m["target_scaler"]), cfg_m)
            )

        preds, trues, n_windows = [], [], 0
        for sp, qp, z in trials:
            member_preds = []
            for model_m, scalers_m, cfg_m in members:
                result = predict_trial(model_m, scalers_m, sp, qp, z, cfg_m, device)
                if result is None:
                    raise RuntimeError(f"trial {sp} 未产出任何窗口（长度不足 window）")
                member_preds.append(result[0])
                true, nw = result[1], result[2]
            pred = (
                np.mean(member_preds, axis=0) if len(member_preds) > 1
                else member_preds[0]
            )
            preds.append(pred)
            trues.append(true)
            n_windows += nw
            if per_trial:
                fold = int(re.search(r"fold(\d+)", os.path.basename(ck)).group(1))
                trial_rows.append(
                    {
                        "fold": fold,
                        "test_subject": test_z,
                        "trial": os.path.basename(sp),
                        "n_windows": int(nw),
                        **fold_metrics([pred], [true], SUBJECT_WEIGHT_N[test_z]),
                    }
                )

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
        print(
            f"[{os.path.basename(ck)}] test={test_z}: {len(preds)} trials 已评估"
            + (f"（ensemble {len(members)} 模型平均）" if len(members) > 1 else "")
            + ("（feature scaler 已按受试者 refit）" if scaler_refit == "test" else "")
            + ("（含无效 trial）" if include_invalid else ""),
            flush=True,
        )
    return rows, trial_rows


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
        "--scaler-refit",
        default="none",
        choices=["none", "test"],
        help="A5 transductive 自校准：test=feature scaler 按留出受试者本人"
             "数据 refit（无标签，target scaler 不动）；none=训练时 scaler"
             "（默认，原口径）",
    )
    parser.add_argument(
        "--per-trial",
        action="store_true",
        help="额外输出逐 trial 指标表 metrics_per_trial*.csv（数据诊断：定位"
             "故障 trial 残差、逐 trial 排查右足 vx 等）",
    )
    parser.add_argument(
        "--ensemble",
        default=None,
        help="逗号分隔的附加 run 目录：同名折 checkpoint 的预测与主 run 平均"
             "（多种子 ensemble 指标）；成员口径字段必须与主 run 一致",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="显式包含 INVALID_TRIALS（LQW03/04、ZWJ10–20），仅数据诊断用，"
             "不用于论文口径",
    )
    parser.add_argument(
        "--plate-zero",
        default="auto",
        choices=["auto", "on", "off"],
        help="评估期力板校零覆盖：auto=按各 checkpoint 训练口径（默认）；"
             "on/off 强制统一（与训练口径不一致时告警）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出文件名（写入 run-dir）；默认 metrics_full.csv（stitch != "
             "uniform / refit != none 时带后缀）",
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

    rows, trial_rows = evaluate_run(
        args.run_dir, args.data_root, device,
        stitch=args.stitch, scaler_refit=args.scaler_refit,
        per_trial=args.per_trial,
        ensemble_dirs=tuple(d for d in args.ensemble.split(",") if d) if args.ensemble else (),
        include_invalid=args.include_invalid,
        plate_zero=args.plate_zero,
    )
    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    suffix = ""
    if args.stitch != "uniform":
        suffix += f"_{args.stitch}"
    if args.scaler_refit != "none":
        suffix += f"_refit-{args.scaler_refit}"
    if args.ensemble:
        suffix += "_ens"
    if args.include_invalid:
        suffix += "_withinvalid"
    out_name = args.out or f"metrics_full{suffix}.csv"
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
                   "scaler_refit": args.scaler_refit,
                   "ensemble_dirs": args.ensemble.split(",") if args.ensemble else [],
                   "include_invalid": args.include_invalid,
                   "aggregate": agg}, f, ensure_ascii=False, indent=2)

    if args.per_trial:
        trial_path = os.path.join(args.run_dir, f"metrics_per_trial{suffix}.csv")
        pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
        print(f"逐 trial 指标已写入 {trial_path}（{len(trial_rows)} 行）")

    print(f"\n指标已写入 {out_path}（汇总 {agg_path}，stitch={args.stitch}，"
          f"scaler_refit={args.scaler_refit}）")
    show = ["fold", "test_subject",
            "rmse_pctbw_ground_force_left_vy", "rmse_pctbw_ground_force_right_vy",
            "peak_err_pctbw_ground_force_left_vy", "peak_err_pctbw_ground_force_right_vy",
            "rmse_pctbw_resultant", "pearson_r_resultant"]
    with pd.option_context("display.width", 250):
        print(df[show].round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
