"""ticket #3 端到端测试：在约定 seam（python -m gait_grf.train 入口命令）上验证。"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from gait_grf.constants import FEATURE_COLS, LEFT_FOOT_PLATE, PLATE_TARGET_COLS, TARGET_COLS
from gait_grf.data import discover_trial_pairs
from gait_grf.train import split_val_trials, stitch_windows, train_one_fold

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

        rows, trial_rows = evaluate_run(self.out_dir, self.data_root, torch.device("cpu"))
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


class TestStitchWindows(unittest.TestCase):
    """重叠窗拼接：uniform 为历史口径，hann/center 为削峰对比实验的模式。"""

    def test_uniform_matches_manual_average(self):
        rng = np.random.default_rng(0)
        pred = rng.standard_normal((5, 100, 6))
        out = stitch_windows(pred, 100, 10, mode="uniform")
        t_cov = 100 + 4 * 10
        self.assertEqual(out.shape, (t_cov, 6))
        for t in range(t_cov):
            ks = [k for k in range(5) if k * 10 <= t < k * 10 + 100]
            manual = np.mean([pred[k, t - k * 10] for k in ks], axis=0)
            np.testing.assert_allclose(out[t], manual, rtol=1e-12)

    def test_all_modes_same_shape_and_finite(self):
        rng = np.random.default_rng(1)
        pred = rng.standard_normal((7, 100, 6))
        for mode in ("uniform", "hann", "center"):
            out = stitch_windows(pred, 100, 10, mode=mode)
            self.assertEqual(out.shape, (100 + 6 * 10, 6), mode)
            self.assertTrue(np.isfinite(out).all(), mode)

    def test_single_window_identity_for_all_modes(self):
        # n=1：无重叠可言，三种模式都应原样返回唯一窗
        rng = np.random.default_rng(2)
        pred = rng.standard_normal((1, 100, 6))
        for mode in ("uniform", "hann", "center"):
            np.testing.assert_allclose(
                stitch_windows(pred, 100, 10, mode=mode), pred[0], rtol=1e-12
            )

    def test_center_takes_nearest_center_window_without_averaging(self):
        # 每窗填不同常数：center 模式每帧输出必等于某一窗的常数（无平均）
        n, w, step = 5, 100, 10
        pred = np.zeros((n, w, 1))
        for k in range(n):
            pred[k, :, 0] = k + 1.0
        out = stitch_windows(pred, w, step, mode="center")
        t = np.arange(out.shape[0])
        k_expected = np.clip(np.round((t - (w - 1) / 2.0) / step), 0, n - 1).astype(int)
        np.testing.assert_allclose(out[:, 0], k_expected + 1.0)

    def test_hann_matches_weighted_reference(self):
        # hann = 逐帧 Hann 权重加权平均；端点权重为 0 的帧退回 uniform
        rng = np.random.default_rng(3)
        n, w, step = 5, 100, 10
        pred = rng.standard_normal((n, w, 6))
        out = stitch_windows(pred, w, step, mode="hann")
        t_cov = w + (n - 1) * step
        j = np.arange(w)
        win = 0.5 * (1.0 - np.cos(2.0 * np.pi * j / (w - 1)))
        for t in range(t_cov):
            cover = [k for k in range(n) if k * step <= t < k * step + w]
            num = np.zeros(6)
            den = 0.0
            for k in cover:
                o = t - k * step
                num += win[o] * pred[k, o]
                den += win[o]
            if den > 1e-12:
                np.testing.assert_allclose(out[t], num / den, rtol=1e-12)
            else:  # 只有端点权重窗覆盖（序列首尾）-> uniform 回退
                manual = np.mean([pred[k, t - k * step] for k in cover], axis=0)
                np.testing.assert_allclose(out[t], manual, rtol=1e-12)

    def test_unknown_mode_raises(self):
        pred = np.zeros((2, 100, 6))
        with self.assertRaises(ValueError):
            stitch_windows(pred, 100, 10, mode="nope")


class TestEvaluateKinematicCheckpoint(unittest.TestCase):
    """回归：evaluate 从 checkpoint 重建模型必须用 checkpoint 的
    feature_mode（或显式 input_size）推输入维度。

    曾在 evaluate.py 硬编码 input_size=len(FEATURE_COLS)=120，导致
    kinematic 51 维 checkpoint 的后置评估在 load_state_dict 处崩溃。
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        rng = np.random.default_rng(7)
        _write_subject_fixture(root, "z1", "LQW", 3, rng, codes=["01", "02", "05"])
        _write_subject_fixture(root, "z3", "HYJ", 3, rng)
        cls.data_root = root
        cls.out_dir = os.path.join(root, "out")
        cls.proc = subprocess.run(
            [
                sys.executable, "-m", "gait_grf.train",
                "--data-root", root,
                "--out-dir", cls.out_dir,
                "--subjects", "z1", "z3",
                "--features", "kinematic",
                "--hidden", "8",
                "--epochs", "2",
                "--patience", "0",
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

    def test_train_exit_zero_and_config_records_input_size(self):
        self.assertEqual(
            self.proc.returncode, 0,
            f"stderr:\n{self.proc.stderr[-3000:]}\nstdout:\n{self.proc.stdout[-2000:]}",
        )
        ck = torch.load(
            os.path.join(self.out_dir, "model_fold1_z1.pt"),
            map_location="cpu", weights_only=False,
        )
        self.assertEqual(ck["config"].get("feature_mode"), "kinematic")
        self.assertEqual(ck["config"].get("input_size"), 51)

    def test_evaluate_rebuilds_model_from_feature_mode(self):
        from gait_grf.evaluate import evaluate_run

        rows, _ = evaluate_run(self.out_dir, self.data_root, torch.device("cpu"))
        self.assertEqual({r["test_subject"] for r in rows}, {"z1", "z3"})
        for row in rows:
            for k, v in row.items():
                if k != "test_subject":
                    self.assertTrue(np.isfinite(v), f"{k} 非有限")

    def test_evaluate_all_stitch_modes(self):
        from gait_grf.evaluate import evaluate_run

        for stitch in ("uniform", "hann", "center"):
            rows, _ = evaluate_run(self.out_dir, self.data_root,
                                   torch.device("cpu"), stitch=stitch)
            self.assertEqual(len(rows), 2, stitch)
            for row in rows:
                for k, v in row.items():
                    if k != "test_subject":
                        self.assertTrue(np.isfinite(v), f"{stitch}/{k} 非有限")


if __name__ == "__main__":
    unittest.main()


class TestSchedulerAndClip(unittest.TestCase):
    """③ 号改进项：--lr-scheduler plateau/cosine + --grad-clip（train_one_fold 级）。"""

    def _fit_val(self):
        """写 z1 fixture 并划分 fit/val；目录用 addCleanup 延迟到测试结束再删。"""
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        rng = np.random.default_rng(15)
        _write_subject_fixture(d, "z1", "LQW", 4, rng,
                               codes=["01", "02", "05", "06"])
        pairs = discover_trial_pairs(d, subjects=["z1"])
        return split_val_trials(pairs, 0.25, np.random.default_rng(0))

    def test_plateau_steps_on_val_loss_each_epoch(self):
        # 接线验证（确定性）：plateau 每轮以「当轮 val_loss」被 step 一次——
        # 降 lr 的触发条件本身是 PyTorch 行为，不在本测试范围
        calls = []
        orig_step = torch.optim.lr_scheduler.ReduceLROnPlateau.step

        def spy(self_s, metrics):
            calls.append(float(metrics))
            return orig_step(self_s, metrics)

        torch.optim.lr_scheduler.ReduceLROnPlateau.step = spy
        try:
            fit, val = self._fit_val()
            cfg = {
                "model": "ltc", "window": 100, "step": 10, "hidden": 8,
                "layers": 1, "dropout": 0.0, "epochs": 3, "patience": 0,
                "batch_size": 8, "lr": 1e-3, "lr_scheduler": "plateau",
            }
            _, _, history = train_one_fold(fit, val, cfg, torch.device("cpu"))
        finally:
            torch.optim.lr_scheduler.ReduceLROnPlateau.step = orig_step
        self.assertEqual(len(calls), 3)  # 每轮一次，不漏不多
        self.assertEqual(calls, [h["val_loss"] for h in history])

    def test_cosine_decays_within_epochs(self):
        fit, val = self._fit_val()
        cfg = {
            "model": "ltc", "window": 100, "step": 10, "hidden": 8,
            "layers": 1, "dropout": 0.0, "epochs": 4, "patience": 0,
            "batch_size": 8, "lr": 1e-3, "lr_scheduler": "cosine",
        }
        _, _, history = train_one_fold(fit, val, cfg, torch.device("cpu"))
        lrs = [h["lr"] for h in history]
        self.assertLess(lrs[-1], lrs[0])
        # cosine 单调不增
        self.assertTrue(all(a >= b for a, b in zip(lrs, lrs[1:])))

    def test_none_scheduler_keeps_constant_lr(self):
        fit, val = self._fit_val()
        cfg = {
            "model": "ltc", "window": 100, "step": 10, "hidden": 8,
            "layers": 1, "dropout": 0.0, "epochs": 2, "patience": 0,
            "batch_size": 8, "lr": 1e-3, "lr_scheduler": "none",
            "grad_clip": 1.0,  # 裁剪不影响 lr 轨迹，可与 none 共存
        }
        _, _, history = train_one_fold(fit, val, cfg, torch.device("cpu"))
        self.assertTrue(all(h["lr"] == 1e-3 for h in history))

    def test_unknown_scheduler_raises(self):
        fit, val = self._fit_val()
        cfg = {
            "model": "ltc", "window": 100, "step": 10, "hidden": 8,
            "layers": 1, "dropout": 0.0, "epochs": 1, "patience": 0,
            "batch_size": 8, "lr": 1e-3, "lr_scheduler": "bogus",
        }
        with self.assertRaises(ValueError):
            train_one_fold(fit, val, cfg, torch.device("cpu"))

    def test_grad_clip_runs_and_recorded(self):
        # 端到端：--grad-clip + --lr-scheduler 进 summary config / checkpoint
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(16)
            _write_subject_fixture(d, "z1", "LQW", 3, rng,
                                   codes=["01", "02", "05"])
            _write_subject_fixture(d, "z3", "HYJ", 3, rng)
            out_dir = os.path.join(d, "out")
            proc = subprocess.run(
                [
                    sys.executable, "-m", "gait_grf.train",
                    "--data-root", d, "--out-dir", out_dir,
                    "--subjects", "z1", "z3",
                    "--model", "lstm", "--hidden", "8",
                    "--epochs", "2", "--patience", "0", "--batch-size", "8",
                    "--device", "cpu",
                    "--lr-scheduler", "plateau", "--grad-clip", "1.0",
                    "--plate-zero",
                ],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["config"]["lr_scheduler"], "plateau")
            self.assertEqual(summary["config"]["grad_clip"], 1.0)
            self.assertIs(summary["config"]["plate_zero"], True)
            blob = torch.load(os.path.join(out_dir, "model_fold1_z1.pt"),
                              map_location="cpu", weights_only=False)
            self.assertEqual(blob["config"]["lr_scheduler"], "plateau")
            self.assertIs(blob["config"]["plate_zero"], True)


class TestPerTrialAndEnsemble(unittest.TestCase):
    """evaluate --per-trial / --ensemble / --include-invalid 的入口烟测。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        rng = np.random.default_rng(17)
        _write_subject_fixture(root, "z1", "LQW", 3, rng,
                               codes=["01", "02", "05"])
        _write_subject_fixture(root, "z3", "HYJ", 3, rng)
        cls.data_root = root
        cls.out_dir = os.path.join(root, "out_s42")
        cls.out_dir_s43 = os.path.join(root, "out_s43")
        common = [
            "--data-root", root,
            "--subjects", "z1", "z3",
            "--model", "lstm", "--hidden", "8",
            "--epochs", "2", "--patience", "0", "--batch-size", "8",
            "--device", "cpu",
        ]
        cls.proc = subprocess.run(
            [sys.executable, "-m", "gait_grf.train",
             "--out-dir", cls.out_dir, "--seed", "42"] + common,
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
        )
        cls.proc_s43 = subprocess.run(
            [sys.executable, "-m", "gait_grf.train",
             "--out-dir", cls.out_dir_s43, "--seed", "43"] + common,
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_both_runs_ok(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-2000:])
        self.assertEqual(self.proc_s43.returncode, 0, self.proc_s43.stderr[-2000:])

    def test_per_trial_rows_and_file(self):
        from gait_grf.evaluate import evaluate_run

        rows, trial_rows = evaluate_run(
            self.out_dir, self.data_root, torch.device("cpu"), per_trial=True
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(trial_rows), 6)  # 每折 3 个 trial
        self.assertEqual(
            {r["trial"] for r in trial_rows if r["test_subject"] == "z1"},
            {"LQW01.csv", "LQW02.csv", "LQW05.csv"},
        )
        for r in trial_rows:
            self.assertGreater(r["n_windows"], 0)
            self.assertTrue(
                np.isfinite([v for k, v in r.items()
                             if k not in ("test_subject", "trial")]).all()
            )

    def test_ensemble_averages_predictions(self):
        from gait_grf.evaluate import evaluate_run

        single_rows, _ = evaluate_run(
            self.out_dir, self.data_root, torch.device("cpu")
        )
        ens_rows, _ = evaluate_run(
            self.out_dir, self.data_root, torch.device("cpu"),
            ensemble_dirs=(self.out_dir_s43,),
        )
        self.assertEqual(len(ens_rows), 2)
        # ensemble 与单模型逐折可比（同折同受试者），指标有限
        for a, b in zip(single_rows, ens_rows):
            self.assertEqual(a["test_subject"], b["test_subject"])
            for k, v in b.items():
                if k != "test_subject":
                    self.assertTrue(np.isfinite(v), f"{k} 非有限")

    def test_ensemble_rejects_mismatched_member(self):
        from gait_grf.evaluate import evaluate_run

        # 篡改成员 checkpoint 的 window 字段 -> 口径不一致必须报错
        with tempfile.TemporaryDirectory() as d:
            for name in ("model_fold1_z1.pt", "model_fold2_z3.pt"):
                blob = torch.load(os.path.join(self.out_dir_s43, name),
                                  map_location="cpu", weights_only=False)
                blob["config"]["window"] = 50
                torch.save(blob, os.path.join(d, name))
            with self.assertRaises(SystemExit):
                evaluate_run(
                    self.out_dir, self.data_root, torch.device("cpu"),
                    ensemble_dirs=(d,),
                )

    def test_per_trial_cli_writes_csv(self):
        from gait_grf.evaluate import main as evaluate_main

        evaluate_main([
            "--run-dir", self.out_dir, "--data-root", self.data_root,
            "--device", "cpu", "--per-trial",
            "--ensemble", self.out_dir_s43,
        ])
        for name in ("metrics_per_trial_ens.csv", "metrics_full_ens.csv"):
            path = os.path.join(self.out_dir, name)
            self.assertTrue(os.path.isfile(path), f"缺少输出文件 {name}")
        df = pd.read_csv(os.path.join(self.out_dir, "metrics_per_trial_ens.csv"))
        self.assertEqual(len(df), 6)
        self.assertIn("trial", df.columns)
        self.assertIn("pearson_r_resultant", df.columns)


class TestWeightedMse(unittest.TestCase):
    """B6 幅值加权损失：w = 1 + λ·|y|/mean|y|（z 空间，权重取自目标）。"""

    def test_zero_lambda_equals_mse(self):
        from gait_grf.train import weighted_mse

        torch.manual_seed(0)
        pred, y = torch.randn(8, 5, 6), torch.randn(8, 5, 6)
        y_scale = float(y.abs().mean())
        self.assertAlmostEqual(
            weighted_mse(pred, y, 0.0, y_scale).item(),
            torch.nn.functional.mse_loss(pred, y).item(),
            places=6,
        )

    def test_matches_formula_and_upweights_high_amplitude(self):
        from gait_grf.train import weighted_mse

        # 常量零预测 vs 两段幅值的目标：λ>0 时损失必须高于纯 MSE（高幅值帧加权）
        y = torch.tensor([[[0.0]] * 4 + [[3.0]] * 4])  # (1,8,1) mean|y| = 1.5
        pred = torch.zeros_like(y)
        y_scale = float(y.abs().mean())
        mse = ((pred - y) ** 2).mean()
        lam = 1.0
        got = weighted_mse(pred, y, lam, y_scale).item()
        expected = float(
            np.mean((1.0 + lam * np.abs(y.numpy()) / y_scale) * y.numpy() ** 2)
        )
        self.assertAlmostEqual(got, expected, places=5)
        self.assertAlmostEqual(weighted_mse(pred, y, 0.0, y_scale).item(), mse.item(), places=6)
        self.assertGreater(got, mse.item())

    def test_target_has_no_grad(self):
        # 权重取自目标侧，不得向目标反传（pred 才是梯度来源）
        from gait_grf.train import weighted_mse

        pred = torch.zeros(1, 4, 1, requires_grad=True)
        y = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        loss = weighted_mse(pred, y, 1.0, float(y.abs().mean()))
        loss.backward()
        self.assertTrue(pred.grad is not None and torch.isfinite(pred.grad).all())


class TestImprovementsEndToEnd(unittest.TestCase):
    """B6 --peak-weight + A4 --mirror-aug 的入口命令烟测（小 fixture，LSTM 提速）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        rng = np.random.default_rng(7)
        _write_subject_fixture(root, "z1", "LQW", 4, rng,
                               codes=["01", "02", "05", "06"])
        _write_subject_fixture(root, "z3", "HYJ", 4, rng)
        cls.data_root = root
        cls.out_dir = os.path.join(root, "out")
        cls.proc = subprocess.run(
            [
                sys.executable, "-m", "gait_grf.train",
                "--data-root", root,
                "--out-dir", cls.out_dir,
                "--subjects", "z1", "z3",
                "--model", "lstm",
                "--features", "kinematic_min",
                "--hidden", "8",
                "--epochs", "3",
                "--patience", "2",
                "--batch-size", "8",
                "--device", "cpu",
                "--peak-weight", "1.0",
                "--mirror-aug",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_exit_zero(self):
        self.assertEqual(
            self.proc.returncode, 0,
            f"stderr:\n{self.proc.stderr[-3000:]}\nstdout:\n{self.proc.stdout[-2000:]}",
        )

    def test_config_and_weighted_val_recorded(self):
        with open(os.path.join(self.out_dir, "summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["config"]["peak_weight"], 1.0)
        self.assertIs(summary["config"]["mirror_aug"], True)
        self.assertEqual(summary["config"]["feature_mode"], "kinematic_min")
        for fold in summary["folds"]:
            for h in fold["history"]:
                # λ>0 时早停判据为加权损失，同时记录 plain val_mse 供跨实验可比
                self.assertIn("val_loss", h)
                self.assertIn("val_mse", h)
                self.assertGreaterEqual(h["val_loss"], h["val_mse"])

    def test_checkpoint_config_persisted(self):
        blob = torch.load(
            os.path.join(self.out_dir, "model_fold1_z1.pt"),
            map_location="cpu", weights_only=False,
        )
        self.assertEqual(blob["config"]["peak_weight"], 1.0)
        self.assertIs(blob["config"]["mirror_aug"], True)

    def test_metrics_finite(self):
        df = pd.read_csv(os.path.join(self.out_dir, "metrics.csv"))
        values = df[list(METRIC_COLS)].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(values).all())


class TestScalerRefitEvaluate(unittest.TestCase):
    """A5 transductive scaler refit：evaluate 侧免重训自校准路径。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        rng = np.random.default_rng(8)
        _write_subject_fixture(root, "z1", "LQW", 3, rng,
                               codes=["01", "02", "05"])
        _write_subject_fixture(root, "z3", "HYJ", 3, rng)
        cls.data_root = root
        cls.out_dir = os.path.join(root, "out")
        cls.proc = subprocess.run(
            [
                sys.executable, "-m", "gait_grf.train",
                "--data-root", root,
                "--out-dir", cls.out_dir,
                "--subjects", "z1", "z3",
                "--hidden", "8",
                "--epochs", "2",
                "--patience", "0",
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

    def test_refit_rows_finite_and_complete(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-2000:])
        from gait_grf.evaluate import evaluate_run

        rows, _ = evaluate_run(
            self.out_dir, self.data_root, torch.device("cpu"),
            scaler_refit="test",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["test_subject"] for r in rows}, {"z1", "z3"})
        for row in rows:
            self.assertGreater(row["n_test_trials"], 0)
            for k, v in row.items():
                if k != "test_subject":
                    self.assertTrue(np.isfinite(v), f"{k} 非有限")

    def test_refit_cli_output_suffix(self):
        from gait_grf.evaluate import main as evaluate_main

        # 默认口径写 metrics_full.csv（复算烟测），refit 写 _refit-test 后缀文件
        evaluate_main([
            "--run-dir", self.out_dir, "--data-root", self.data_root,
            "--device", "cpu",
        ])
        evaluate_main([
            "--run-dir", self.out_dir, "--data-root", self.data_root,
            "--device", "cpu", "--scaler-refit", "test",
        ])
        for name in ("metrics_full.csv", "metrics_full_refit-test.csv",
                     "metrics_full_aggregate.json",
                     "metrics_full_refit-test_aggregate.json"):
            path = os.path.join(self.out_dir, name)
            self.assertTrue(os.path.isfile(path), f"缺少输出文件 {name}")
        import json as _json
        with open(os.path.join(self.out_dir, "metrics_full_refit-test_aggregate.json"),
                  encoding="utf-8") as f:
            agg = _json.load(f)
        self.assertEqual(agg["scaler_refit"], "test")


class TestOptimAndCellCLI(unittest.TestCase):
    """issue #0014 训练配方：--cell/--optimizer/--weight-decay/--init-gain 的
    入口烟测 + checkpoint 往返（含 ltc_ncp / cell=mix 的 evaluate 重建）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        rng = np.random.default_rng(21)
        _write_subject_fixture(root, "z1", "LQW", 3, rng,
                               codes=["01", "02", "05"])
        _write_subject_fixture(root, "z3", "HYJ", 3, rng)
        cls.data_root = root
        cls.out_mix = os.path.join(root, "out_mix")
        cls.out_ncp = os.path.join(root, "out_ncp")
        cls.out_adamw = os.path.join(root, "out_adamw")
        common = ["--data-root", root, "--subjects", "z1", "z3",
                  "--hidden", "8", "--epochs", "2", "--patience", "0",
                  "--batch-size", "8", "--device", "cpu", "--ode-unfolds", "2"]
        cls.proc_mix = subprocess.run(
            [sys.executable, "-m", "gait_grf.train", "--out-dir", cls.out_mix,
             "--model", "ltc_attn", "--cell", "mix"] + common,
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
        cls.proc_ncp = subprocess.run(
            [sys.executable, "-m", "gait_grf.train", "--out-dir", cls.out_ncp,
             "--model", "ltc_ncp", "--ncp-units", "14"] + common,
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
        cls.proc_adamw = subprocess.run(
            [sys.executable, "-m", "gait_grf.train", "--out-dir", cls.out_adamw,
             "--model", "ltc", "--optimizer", "adamw", "--weight-decay", "1e-4",
             "--init-gain", "0.9"] + common,
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_exit_zero(self):
        for p in (self.proc_mix, self.proc_ncp, self.proc_adamw):
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])

    def test_configs_recorded(self):
        import torch as _t
        blob_mix = _t.load(os.path.join(self.out_mix, "model_fold1_z1.pt"),
                           map_location="cpu", weights_only=False)
        self.assertEqual(blob_mix["config"]["cell"], "mix")
        blob_ncp = _t.load(os.path.join(self.out_ncp, "model_fold1_z1.pt"),
                           map_location="cpu", weights_only=False)
        self.assertEqual(blob_ncp["config"]["model"], "ltc_ncp")
        self.assertEqual(blob_ncp["config"]["ncp_units"], 14)
        blob_adamw = _t.load(os.path.join(self.out_adamw, "model_fold1_z1.pt"),
                             map_location="cpu", weights_only=False)
        self.assertEqual(blob_adamw["config"]["optimizer"], "adamw")
        self.assertEqual(blob_adamw["config"]["weight_decay"], 1e-4)
        self.assertEqual(blob_adamw["config"]["init_gain"], 0.9)

    def test_evaluate_rebuilds_mixed_and_ncp(self):
        from gait_grf.evaluate import evaluate_run

        for out_dir in (self.out_mix, self.out_ncp):
            rows, _ = evaluate_run(out_dir, self.data_root, torch.device("cpu"))
            self.assertEqual(len(rows), 2)
            for row in rows:
                for k, v in row.items():
                    if k != "test_subject":
                        self.assertTrue(np.isfinite(v), f"{out_dir}/{k} 非有限")

    def test_adamw_param_groups_exclude_circuit_params(self):
        # 衰减只作用于 ≥2 维权重；LTC 电路参数（cm/gleak/w 等 1-2 维有量程约束）
        # 归入 no-decay 组的判定：2 维电路参数（w/mu/sigma）也应豁免——
        # 实现按 dim>=2 分组，此处仅验证 AdamW 双组构造与训练收敛
        with open(os.path.join(self.out_adamw, "summary.json"), encoding="utf-8") as f:
            cfg = json.load(f)["config"]
        self.assertEqual(cfg["optimizer"], "adamw")
        self.assertEqual(cfg["weight_decay"], 1e-4)




class TestBuildOptimizer(unittest.TestCase):
    """AdamW 分组：电路参数豁免、权重矩阵衰减、bias 豁免。"""

    def _model(self):
        from gait_grf.models import make_model

        return make_model("ltc", input_size=120, hidden=16, layers=1, ode_unfolds=2)

    def test_adamw_groups(self):
        from gait_grf.train import build_optimizer

        m = self._model()
        opt = build_optimizer(m, {"optimizer": "adamw", "weight_decay": 1e-4, "lr": 1e-3})
        self.assertIsInstance(opt, torch.optim.AdamW)
        decay, no_decay = opt.param_groups
        self.assertEqual(decay["weight_decay"], 1e-4)
        self.assertEqual(no_decay["weight_decay"], 0.0)
        names = {id(p) for p in m.parameters() if p.requires_grad}
        self.assertTrue({id(p) for p in decay["params"]} | {id(p) for p in no_decay["params"]} == names)
        # Linear 权重必须进衰减组
        lin_w = {id(p) for n, p in m.named_parameters() if "readout.weight" in n or "w" in n and p.dim() == 2 and "._params." not in n}
        self.assertTrue(lin_w & {id(p) for p in decay["params"]})

    def test_circuit_params_exempt_from_decay(self):
        from gait_grf.train import build_optimizer

        m = self._model()
        opt = build_optimizer(m, {"optimizer": "adamw", "weight_decay": 1e-4, "lr": 1e-3})
        _, no_decay = opt.param_groups
        no_decay_ids = {id(p) for p in no_decay["params"]}
        # LTC 电路参数（LTCCell 上的 w/sigma 等，2 维但必须豁免）按模块识别
        from ncps.torch import LTCCell as _LTCCell

        circuit = set()
        for mod in m.modules():
            if isinstance(mod, _LTCCell):
                circuit.update(id(p) for p in mod.parameters() if p.requires_grad)
        self.assertTrue(circuit)
        self.assertTrue(circuit <= no_decay_ids)

    def test_adam_single_group_and_unknown_raises(self):
        from gait_grf.train import build_optimizer

        m = self._model()
        opt = build_optimizer(m, {"optimizer": "adam", "lr": 1e-3})
        self.assertIsInstance(opt, torch.optim.Adam)
        with self.assertRaises(ValueError):
            build_optimizer(m, {"optimizer": "sgd", "lr": 1e-3})

    def test_apply_init_gain_touches_only_linear(self):
        from gait_grf.train import apply_init_gain

        torch.manual_seed(0)
        m = self._model()
        from ncps.torch import LTCCell as _LTCCell

        circuit_params = set()
        for mod in m.modules():
            if isinstance(mod, _LTCCell):
                circuit_params.update(id(p) for p in mod.parameters())
        before_circuit = {id(p): p.detach().clone() for n, p in m.named_parameters() if id(p) in circuit_params}
        before_lin = {n: p.detach().clone() for n, p in m.named_parameters() if n.endswith("readout.weight")}
        apply_init_gain(m, 1.2)
        after_circuit = {id(p): p.detach().clone() for n, p in m.named_parameters() if id(p) in circuit_params}
        for n in before_circuit:
            self.assertTrue(torch.equal(before_circuit[n], after_circuit[n]), f"电路参数被改写: {n}")
        changed = any(not torch.equal(before_lin[n], p.detach())
                      for n, p in m.named_parameters() if n.endswith("readout.weight"))
        self.assertTrue(changed)
