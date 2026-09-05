import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from gait_grf.constants import (
    FEATURE_COLS,
    LEFT_FOOT_PLATE,
    PLATE_TARGET_COLS,
    PRESSURE_COLS,
    QUAT_COLS,
    SUBJECT_MAP,
    SUM_COLS,
    TARGET_COLS,
)
from gait_grf.data import (
    GRFSequenceDataset,
    align,
    discover_trial_pairs,
    extract_features,
    extract_targets,
    find_lag,
    window_trial,
)


def _make_sensor_df(n, rng):
    cols = ["timestamp", "packet_counter"] + list(FEATURE_COLS)
    df = pd.DataFrame(rng.standard_normal((n, len(cols))), columns=cols)
    df["timestamp"] = [f"{int(t // 60):02d}:{t % 60:.1f}" for t in np.arange(n) / 100.0]
    df["packet_counter"] = np.arange(n)
    return df


def _make_qualisys_df(n, rng):
    # 模拟 Qualisys 原始文件列（力板编号命名）
    cols = ["time"] + list(PLATE_TARGET_COLS)
    for p in ("1", "2"):
        cols += [f"ground_force_{p}_p{a}" for a in ("x", "y", "z")]
        cols += [f"ground_moment_{p}_m{a}" for a in ("x", "y", "z")]
    df = pd.DataFrame(rng.standard_normal((n, len(cols))), columns=cols)
    df["time"] = np.arange(n) / 100.0
    return df


def _gait_like(n):
    # 非周期、平滑、非负的类步态信号：互相关在真实滞后处有唯一峰值
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(n)
    x = np.convolve(x, np.ones(5) / 5.0, mode="same")
    return np.maximum(x, 0.0) + 0.1


class TestConstants(unittest.TestCase):
    def test_column_counts(self):
        self.assertEqual(len(QUAT_COLS), 28)
        self.assertEqual(len(PRESSURE_COLS), 90)
        self.assertEqual(len(SUM_COLS), 2)
        self.assertEqual(len(FEATURE_COLS), 120)
        self.assertEqual(len(TARGET_COLS), 6)


class TestFindLag(unittest.TestCase):
    def test_recovers_sensor_delay(self):
        # sensor pressure = base delayed by L  ->  sensor lags  ->  lag = -L
        n, L = 200, 7
        rng = np.random.default_rng(0)
        base = _gait_like(n)
        sensor = _make_sensor_df(n, rng)
        qual = _make_qualisys_df(n, rng)
        p = np.zeros(n)
        p[L:] = base[: n - L]
        sensor["left_pressure_sum"] = p
        sensor["right_pressure_sum"] = 0.0
        qual["ground_force_1_vy"] = base
        qual["ground_force_2_vy"] = 0.0
        self.assertEqual(find_lag(sensor, qual), -L)

    def test_recovers_sensor_lead(self):
        # GRF = base delayed by L  ->  sensor leads  ->  lag = +L
        n, L = 200, 5
        rng = np.random.default_rng(1)
        base = _gait_like(n)
        sensor = _make_sensor_df(n, rng)
        qual = _make_qualisys_df(n, rng)
        g = np.zeros(n)
        g[L:] = base[: n - L]
        sensor["left_pressure_sum"] = base
        sensor["right_pressure_sum"] = 0.0
        qual["ground_force_1_vy"] = g
        qual["ground_force_2_vy"] = 0.0
        self.assertEqual(find_lag(sensor, qual), L)


class TestAlign(unittest.TestCase):
    def test_align_equal_length_and_zero_residual_lag(self):
        n, L = 160, 6
        rng = np.random.default_rng(2)
        base = _gait_like(n)
        sensor = _make_sensor_df(n, rng)
        qual = _make_qualisys_df(n, rng)
        p = np.zeros(n)
        p[L:] = base[: n - L]
        sensor["left_pressure_sum"] = p
        sensor["right_pressure_sum"] = 0.0
        qual["ground_force_1_vy"] = base
        qual["ground_force_2_vy"] = 0.0

        s, q, lag = align(sensor, qual)
        self.assertEqual(lag, -L)
        self.assertEqual(len(s), len(q))
        # after alignment, residual lag should be zero
        self.assertEqual(find_lag(s, q), 0)


class TestExtract(unittest.TestCase):
    def test_feature_and_target_shapes(self):
        n = 120
        rng = np.random.default_rng(3)
        sensor = _make_sensor_df(n, rng)
        qual = _make_qualisys_df(n, rng)
        f = extract_features(sensor)
        t = extract_targets(qual)
        self.assertEqual(f.shape, (n, 120))
        self.assertEqual(t.shape, (n, 6))
        self.assertTrue(np.isfinite(f).all())
        self.assertTrue(np.isfinite(t).all())

    def test_targets_normalized_to_foot_semantics(self):
        # 目标列序与受试者无关：前 3 列恒为左脚、后 3 列恒为右脚。
        # z1–z5 左脚踩板1（列序不变）；z6–z8 左脚踩板2（交换两组列）。
        n = 60
        rng = np.random.default_rng(10)
        qual = _make_qualisys_df(n, rng)
        qual["ground_force_1_vx"] = 11.0
        qual["ground_force_1_vy"] = 12.0
        qual["ground_force_1_vz"] = 13.0
        qual["ground_force_2_vx"] = 21.0
        qual["ground_force_2_vy"] = 22.0
        qual["ground_force_2_vz"] = 23.0
        plate1_left = extract_targets(qual, left_plate=1)
        self.assertEqual(list(TARGET_COLS[:3]), ["ground_force_left_vx", "ground_force_left_vy", "ground_force_left_vz"])
        np.testing.assert_allclose(plate1_left[0], [11, 12, 13, 21, 22, 23])
        plate2_left = extract_targets(qual, left_plate=2)
        np.testing.assert_allclose(plate2_left[0], [21, 22, 23, 11, 12, 13])

    def test_left_foot_plate_mapping_covers_all_subjects(self):
        # 覆盖全部受试者；z1–z5 踩板1，z6–z8 踩板2（采集协议）
        self.assertEqual(set(LEFT_FOOT_PLATE), set(SUBJECT_MAP))
        for z in ("z1", "z2", "z3", "z4", "z5"):
            self.assertEqual(LEFT_FOOT_PLATE[z], 1)
        for z in ("z6", "z7", "z8"):
            self.assertEqual(LEFT_FOOT_PLATE[z], 2)

    def test_inf_and_nan_cleaned(self):
        n = 120
        rng = np.random.default_rng(4)
        sensor = _make_sensor_df(n, rng)
        sensor.loc[10, "right_pressure0"] = np.inf
        sensor.loc[20, "right_thigh_q1"] = np.nan
        f = extract_features(sensor)
        self.assertTrue(np.isfinite(f).all())


class TestWindow(unittest.TestCase):
    def test_window_shapes_and_count(self):
        n, W, S = 145, 100, 10
        rng = np.random.default_rng(5)
        f = rng.standard_normal((n, 120))
        t = rng.standard_normal((n, 6))
        X, y = window_trial(f, t, W, S)
        expected = (n - W) // S + 1
        self.assertEqual(X.shape, (expected, W, 120))
        self.assertEqual(y.shape, (expected, W, 6))
        self.assertTrue(np.allclose(X[0], f[0:W]))
        self.assertTrue(np.allclose(X[1], f[S : S + W]))

    def test_short_trial_yields_empty(self):
        rng = np.random.default_rng(6)
        f = rng.standard_normal((50, 120))
        t = rng.standard_normal((50, 6))
        X, y = window_trial(f, t, 100, 10)
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)


class TestDataset(unittest.TestCase):
    def _write_trial(self, root, tag, n, sentinel, rng, subject="z1"):
        sensor = _make_sensor_df(n, rng)
        qual = _make_qualisys_df(n, rng)
        base = _gait_like(n)
        sensor["left_pressure_sum"] = base
        sensor["right_pressure_sum"] = 0.0
        # 左脚所踩板随受试者组变化：z1–z5 -> 板1；z6–z8 -> 板2
        left_plate = LEFT_FOOT_PLATE[subject]
        qual[f"ground_force_{left_plate}_vy"] = base
        qual[f"ground_force_{3 - left_plate}_vy"] = 0.0
        # sentinel in a feature column to detect cross-trial mixing
        sensor["right_pressure0"] = float(sentinel)
        sp = os.path.join(root, f"sensor_{tag}.csv")
        qp = os.path.join(root, f"qual_{tag}.csv")
        sensor.to_csv(sp, index=False)
        qual.to_csv(qp, index=False)
        return sp, qp, subject

    def test_dataset_windows_do_not_cross_trials(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as d:
            t1 = self._write_trial(d, "a", 150, sentinel=1.0, rng=rng)
            t2 = self._write_trial(d, "b", 140, sentinel=2.0, rng=rng)
            ds = GRFSequenceDataset([t1, t2], window=100, step=10)
            n1 = (150 - 100) // 10 + 1
            n2 = (140 - 100) // 10 + 1
            self.assertEqual(len(ds), n1 + n2)
            # every window's pressure0 column is uniform (never mixes two trials)
            sentinels = set()
            for i in range(len(ds)):
                X, _ = ds[i]
                col = X[:, FEATURE_COLS.index("right_pressure0")]
                self.assertTrue(np.allclose(col, col[0]))
                sentinels.add(round(float(col[0]), 2))
            # 缩放后两 trial 的哨兵值不同，但窗口内恒定 → 恰好两种不同值
            self.assertEqual(len(sentinels), 2)

    def test_dataset_swaps_targets_for_plate2_left_subjects(self):
        # z6 左脚踩板2：目标 ground_force_left_vy 必须取自板2 vy（而非板1），
        # 且与左脚压力同相；ground_force_right_vy 恒 0。
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as d:
            t = self._write_trial(d, "z6", 150, sentinel=1.0, rng=rng, subject="z6")
            ds = GRFSequenceDataset([t], window=100, step=10)
            X, y = ds[0]
            yN = ds.target_scaler.inverse_transform(
                y.numpy().reshape(-1, 6)
            ).reshape(100, 6)
            left_vy, right_vy = yN[:, 1], yN[:, 4]
            lpN = X[:, FEATURE_COLS.index("left_pressure_sum")].numpy()
            lpN = lpN * ds.feature_scaler.scale_[FEATURE_COLS.index("left_pressure_sum")] \
                + ds.feature_scaler.mean_[FEATURE_COLS.index("left_pressure_sum")]
            self.assertGreater(float(np.corrcoef(left_vy, lpN)[0, 1]), 0.99)
            self.assertTrue(np.allclose(right_vy, 0.0, atol=1e-6))

    def test_dataset_no_nan_and_finite(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as d:
            t1 = self._write_trial(d, "a", 150, sentinel=1.0, rng=rng)
            ds = GRFSequenceDataset([t1], window=100, step=10)
            X, y = ds[0]
            self.assertTrue(np.isfinite(X.numpy()).all())
            self.assertTrue(np.isfinite(y.numpy()).all())
            self.assertEqual(X.shape[1], 120)
            self.assertEqual(y.shape[1], 6)

    def test_pretrained_scaler_applied(self):
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as d:
            t1 = self._write_trial(d, "a", 150, sentinel=1.0, rng=rng)
            t2 = self._write_trial(d, "b", 140, sentinel=2.0, rng=rng)
            train = GRFSequenceDataset([t1], window=100, step=10)
            val = GRFSequenceDataset(
                [t2],
                window=100,
                step=10,
                feature_scaler=train.feature_scaler,
                target_scaler=train.target_scaler,
            )
            self.assertTrue(np.isfinite(val[0][0].numpy()).all())


class TestDiscover(unittest.TestCase):
    def test_discovers_pairs_from_real_data_if_present(self):
        root = "/root/autodl-tmp/data/subjectdata"
        if not os.path.isdir(root):
            self.skipTest("subjectdata 不存在")
        pairs = discover_trial_pairs(root, subjects=["z1"])
        self.assertGreater(len(pairs), 0)
        for sp, qp, z in pairs[:3]:
            self.assertTrue(os.path.isfile(sp))
            self.assertTrue(os.path.isfile(qp))
            self.assertEqual(z, "z1")

    def test_discovers_all_224_pairs(self):
        """226 个文件对中，LQW03/04（z1）IMU 数据全零（INVALID_TRIALS），
        连同其测力台数据一并排除，有效 trial 对 = 224。"""
        root = "/root/autodl-tmp/data/subjectdata"
        if not os.path.isdir(root):
            self.skipTest("subjectdata 不存在")
        pairs = discover_trial_pairs(root)
        self.assertEqual(len(pairs), 224)
        # 被排除的文件不得出现在结果里
        codes = {os.path.basename(sp) for sp, _, _ in pairs}
        self.assertNotIn("LQW03.csv", codes)
        self.assertNotIn("LQW04.csv", codes)


class TestRealDataSmoke(unittest.TestCase):
    def test_dataset_on_two_subjects(self):
        root = "/root/autodl-tmp/data/subjectdata"
        if not os.path.isdir(root):
            self.skipTest("subjectdata 不存在")
        pairs = discover_trial_pairs(root, subjects=["z1", "z3"])
        ds = GRFSequenceDataset(pairs, window=100, step=10)
        self.assertGreater(len(ds), 0)
        X, y = ds[0]
        self.assertEqual(tuple(X.shape), (100, 120))
        self.assertEqual(tuple(y.shape), (100, 6))
        self.assertTrue(np.isfinite(ds.X).all())
        self.assertTrue(np.isfinite(ds.y).all())


if __name__ == "__main__":
    unittest.main()
