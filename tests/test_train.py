"""ticket #3 端到端测试：在约定 seam（python -m gait_grf.train 入口命令）上验证。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from gait_grf.constants import FEATURE_COLS, LEFT_FOOT_PLATE, PLATE_TARGET_COLS, TARGET_COLS
from gait_grf.data import discover_trial_pairs
from gait_grf.train import split_val_trials, train_one_fold

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 指标表约定列：6 个 GRF 输出列的 RMSE(N) 与 Pearson r，外加合成幅值两项
METRIC_COLS = (
    [f"rmse_N_{c}" for c in TARGET_COLS]
    + [f"pearson_r_{c}" for c in TARGET_COLS]
    + ["rmse_N_resultant", "pearson_r_resultant"]
)


def _gait_like(n, seed):
    # 非周期、平滑、非负的类步态信号
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = np.convolve(x, np.ones(5) / 5.0, mode="same")
    return np.maximum(x, 0.0) + 0.1


def _shifted(sig, lag, fill):
    """sig 右移 lag 帧，前 lag 帧用 fill 填充。"""
    out = np.full_like(sig, fill)
    out[lag:] = sig[: len(sig) - lag]
    return out


def _write_subject_fixture(root, z, initials, n_trials, rng, codes=None):
    """按 subjectdata 目录布局写一个受试者：左脚 vy = 左压力和 × 5（可学习映射）。

    左脚所踩板随受试者组变化（z1–z5 板1、z6–z8 板2），模拟真实采集协议，
    使跨组 LOSO 必须经受「板->脚」规范化路径。
    codes 为 trial 序号列表（默认 01..n_trials）；z1 夹具需避开
    constants.INVALID_TRIALS 中的真实无效序号（03/04），否则会被过滤。
    """
    n = int(z[1:])
    left_plate = LEFT_FOOT_PLATE[z]
    right_plate = 3 - left_plate
    sensor_dir = os.path.join(root, "sensor", initials)
    qual_dir = os.path.join(root, "Qualisys", f"Z{n}_csv_calibrated")
    os.makedirs(sensor_dir, exist_ok=True)
    os.makedirs(qual_dir, exist_ok=True)
    files = []
    for t, code in enumerate(codes or [f"{i + 1:02d}" for i in range(n_trials)], start=1):
        frames = 150 + 10 * t
        base_l = _gait_like(frames, seed=1000 * n + t)
        base_r = _shifted(base_l, 30, fill=0.1)  # 右足迟 30 帧

        sensor_cols = ["timestamp", "packet_counter"] + list(FEATURE_COLS)
        sensor = pd.DataFrame(
            rng.standard_normal((frames, len(sensor_cols))), columns=sensor_cols
        )
        sensor["left_pressure_sum"] = base_l
        sensor["right_pressure_sum"] = base_r

        qual_cols = ["time"] + list(PLATE_TARGET_COLS)
        qual = pd.DataFrame(
            rng.standard_normal((frames, len(qual_cols))), columns=qual_cols
        )
        qual["time"] = np.arange(frames) / 100.0
        qual[f"ground_force_{left_plate}_vy"] = 5.0 * base_l + 0.05 * rng.standard_normal(frames)
        qual[f"ground_force_{right_plate}_vy"] = 5.0 * base_r + 0.05 * rng.standard_normal(frames)
        # vx/vz 保留小随机幅（非常量、但不可预测）

        sp = os.path.join(sensor_dir, f"{initials}{code}.csv")
        qp = os.path.join(qual_dir, f"z{n}_{code}_mot_100Hz.csv")
        sensor.to_csv(sp, index=False)
        qual.to_csv(qp, index=False)
        files.append(os.path.basename(sp))
    return files


class TestSplitValTrials(unittest.TestCase):
    def test_val_count_and_disjoint(self):
        rng = np.random.default_rng(0)
        pairs = [(i, i) for i in range(20)]
        fit, val = split_val_trials(pairs, 0.15, rng)
        self.assertEqual(len(val), 3)  # round(20*0.15)
        self.assertEqual(len(fit), 17)
        self.assertEqual(len(set(fit) | set(val)), 20)

    def test_small_n_gives_empty_val(self):
        rng = np.random.default_rng(0)
        fit, val = split_val_trials([(0, 0), (1, 1), (2, 2)], 0.15, rng)
        self.assertEqual(val, [])
        self.assertEqual(len(fit), 3)

    def test_deterministic_given_seed(self):
        a = split_val_trials(list(range(10)), 0.2, np.random.default_rng(7))
        b = split_val_trials(list(range(10)), 0.2, np.random.default_rng(7))
        self.assertEqual(a, b)


class TestEarlyStopping(unittest.TestCase):
    def test_stops_when_val_does_not_improve(self):
        # lr=0 -> 权重不变 -> 每轮验证损失逐位相同 -> 第 patience+1 轮触发早停
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(5)
            _write_subject_fixture(d, "z1", "LQW", 4, rng,
                                   codes=["01", "02", "05", "06"])  # 避开 INVALID_TRIALS
            pairs = discover_trial_pairs(d, subjects=["z1"])
            fit, val = split_val_trials(pairs, 0.25, np.random.default_rng(0))
            cfg = {
                "model": "ltc", "window": 100, "step": 10, "hidden": 8,
                "layers": 1, "dropout": 0.0, "epochs": 50, "patience": 3,
                "batch_size": 8, "lr": 0.0,
            }
            _, _, history = train_one_fold(fit, val, cfg, torch.device("cpu"))
            self.assertEqual(len(history), 4)  # 第1轮最优，第4轮停止
            self.assertEqual(history[-1]["best_epoch"], 1)
            self.assertIn("best_val_loss", history[-1])

    def test_no_early_stop_with_zero_patience(self):
        # patience=0 表示不早停：跑满全部轮数
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(5)
            _write_subject_fixture(d, "z1", "LQW", 4, rng,
                                   codes=["01", "02", "05", "06"])  # 避开 INVALID_TRIALS
            pairs = discover_trial_pairs(d, subjects=["z1"])
            fit, val = split_val_trials(pairs, 0.25, np.random.default_rng(0))
            cfg = {
                "model": "ltc", "window": 100, "step": 10, "hidden": 8,
                "layers": 1, "dropout": 0.0, "epochs": 3, "patience": 0,
                "batch_size": 8, "lr": 0.0,
            }
            _, _, history = train_one_fold(fit, val, cfg, torch.device("cpu"))
            self.assertEqual(len(history), 3)


class TestTrainerEndToEnd(unittest.TestCase):
    """入口命令在 2 受试者 fixture 上跑通 LOSO tracer bullet。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        rng = np.random.default_rng(42)
        cls.fixture = {
            # z1 序号避开 INVALID_TRIALS（03/04 为真实数据中的 IMU 全零文件）
            "z1": _write_subject_fixture(root, "z1", "LQW", 4, rng,
                                         codes=["01", "02", "05", "06"]),
            "z3": _write_subject_fixture(root, "z3", "HYJ", 4, rng),
            # z6 左脚踩板2：跨组 LOSO 回归测试（板->脚规范化）
            "z6": _write_subject_fixture(root, "z6", "XBL", 4, rng),
        }
        cls.data_root = root
        cls.out_dir = os.path.join(root, "out")
        cls.proc = subprocess.run(
            [
                sys.executable, "-m", "gait_grf.train",
                "--data-root", root,
                "--out-dir", cls.out_dir,
                "--subjects", "z1", "z3", "z6",
                "--hidden", "16",
                "--epochs", "5",
                "--patience", "3",
                "--batch-size", "8",
                "--device", "cpu",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_exit_zero_and_outputs_written(self):
        self.assertEqual(
            self.proc.returncode, 0,
            f"stderr:\n{self.proc.stderr[-3000:]}\nstdout:\n{self.proc.stdout[-2000:]}",
        )
        for name in ("metrics.csv", "summary.json", "predictions_z1.png",
                     "predictions_z3.png", "predictions_z6.png"):
            path = os.path.join(self.out_dir, name)
            self.assertTrue(os.path.isfile(path), f"缺少输出文件 {name}")
            self.assertGreater(os.path.getsize(path), 0, f"{name} 为空文件")
        # 每折保存最优 checkpoint（权重 + scaler + 配置）
        for name in ("model_fold1_z1.pt", "model_fold2_z3.pt", "model_fold3_z6.pt"):
            path = os.path.join(self.out_dir, name)
            self.assertTrue(os.path.isfile(path), f"缺少 checkpoint {name}")
            self.assertGreater(os.path.getsize(path), 0, f"{name} 为空文件")

    def test_metrics_schema_and_finite(self):
        df = pd.read_csv(os.path.join(self.out_dir, "metrics.csv"))
        for col in ["fold", "test_subject", "n_test_trials", "n_test_windows"] + METRIC_COLS:
            self.assertIn(col, df.columns, f"指标表缺列 {col}")
        values = df[METRIC_COLS].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(values).all(), "存在非有限指标值")

    def test_loso_folds_match_subjects(self):
        df = pd.read_csv(os.path.join(self.out_dir, "metrics.csv"))
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["test_subject"]), {"z1", "z3", "z6"})
        self.assertEqual(list(df["fold"]), [1, 2, 3])

    def test_predictions_cover_all_test_trials_of_held_out_subject(self):
        with open(os.path.join(self.out_dir, "summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(len(summary["folds"]), 3)
        for fold in summary["folds"]:
            expected = set(self.fixture[fold["test_subject"]])
            covered = {t["sensor"] for t in fold["test_trials"]}
            self.assertEqual(covered, expected, "测试 trial 覆盖不完整")
            for t in fold["test_trials"]:
                self.assertGreater(t["n_windows"], 0)
                self.assertGreater(t["n_frames_evaluated"], 0)

    def test_training_loss_decreases(self):
        # vy 由压力和线性决定，最小模型也应当能压低训练损失
        with open(os.path.join(self.out_dir, "summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        for fold in summary["folds"]:
            losses = [h["train_loss"] for h in fold["history"]]
            self.assertLess(losses[-1], losses[0])


    def test_evaluate_entry_recomputes_from_checkpoints(self):
        """#6 后置评估：从 checkpoint 重新预测，产出 %BW/辅助指标（无需重训）。"""
        from gait_grf.evaluate import evaluate_run

        rows = evaluate_run(self.out_dir, self.data_root, torch.device("cpu"))
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["test_subject"] for r in rows}, {"z1", "z3", "z6"})
        for row in rows:
            self.assertGreater(row["n_test_trials"], 0)
            self.assertIn("rmse_pctbw_ground_force_left_vy", row)
            self.assertIn("impulse_err_pctbw_s_ground_force_right_vy", row)
            self.assertIn("peak_lag_frames_ground_force_left_vz", row)
            for k, v in row.items():
                if k != "test_subject":
                    self.assertTrue(np.isfinite(v), f"{k} 非有限")


if __name__ == "__main__":
    unittest.main()
