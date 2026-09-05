"""ticket #6：全量指标（%BW 归一化 + 峰值/冲量/时机）与后置评估入口测试。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gait_grf.constants import SUBJECT_WEIGHT_N, TARGET_COLS
from gait_grf.metrics import fold_metrics

W = 700.0  # 假想体重（N）


def _trial_pair(T, rng, scale=1.0, shift=0):
    """true 与 pred=scale*true+shift 的单 trial 目标对（N）。"""
    true = np.abs(rng.standard_normal((T, 6))) * 100.0
    pred = true * scale + shift
    return pred, true


class TestFoldMetrics(unittest.TestCase):
    def test_rmse_n_and_pctbw_consistent(self):
        rng = np.random.default_rng(0)
        pred, true = _trial_pair(200, rng, shift=10.0)
        m = fold_metrics([pred], [true], W)
        for name in TARGET_COLS:
            self.assertAlmostEqual(
                m[f"rmse_pctbw_{name}"], m[f"rmse_N_{name}"] / W * 100.0, places=9
            )
        self.assertAlmostEqual(
            m["rmse_pctbw_resultant"], m["rmse_N_resultant"] / W * 100.0, places=9
        )

    def test_perfect_prediction(self):
        rng = np.random.default_rng(1)
        p, t = _trial_pair(150, rng)
        m = fold_metrics([p], [t], W)
        for name in TARGET_COLS:
            self.assertAlmostEqual(m[f"rmse_N_{name}"], 0.0, places=6)
            self.assertAlmostEqual(m[f"rmse_pctbw_{name}"], 0.0, places=9)
            self.assertAlmostEqual(m[f"pearson_r_{name}"], 1.0, places=6)
            self.assertAlmostEqual(m[f"peak_err_pctbw_{name}"], 0.0, places=9)
            self.assertAlmostEqual(m[f"impulse_err_pctbw_s_{name}"], 0.0, places=9)
            self.assertAlmostEqual(m[f"peak_lag_frames_{name}"], 0.0, places=6)

    def test_constant_bias_shifts_peak_and_impulse(self):
        """恒定 +10N 偏置：RMSE=10，峰值误差=10N（%BW×100），冲量误差=10*T/100，r 不变。"""
        rng = np.random.default_rng(2)
        T = 150
        pred, true = _trial_pair(T, rng, shift=10.0)
        m = fold_metrics([pred], [true], W)
        # 左脚 vy 是第 2 列
        name = TARGET_COLS[1]
        self.assertAlmostEqual(m[f"rmse_N_{name}"], 10.0, places=6)
        self.assertAlmostEqual(m[f"peak_err_pctbw_{name}"], 10.0 / W * 100.0, places=9)
        self.assertAlmostEqual(m[f"impulse_err_pctbw_s_{name}"],
                               10.0 * T / 100 / W * 100.0, places=9)
        self.assertAlmostEqual(m[f"pearson_r_{name}"], 1.0, places=6)

    def test_peak_lag_detected(self):
        """pred 的峰比 true 晚 7 帧 -> peak_lag_frames=7。"""
        rng = np.random.default_rng(3)
        T = 120
        true = np.zeros((T, 6))
        true[40, 1] = 500.0  # 左脚 vy 单峰
        pred = np.zeros((T, 6))
        pred[47, 1] = 500.0
        m = fold_metrics([pred], [true], W)
        self.assertAlmostEqual(m["peak_lag_frames_ground_force_left_vy"], 7.0, places=6)

    def test_aux_metrics_average_across_trials(self):
        """两 trial 的峰值误差不同 -> 辅助指标取平均。"""
        rng = np.random.default_rng(4)
        p1, t1 = _trial_pair(100, rng, shift=5.0)
        p2, t2 = _trial_pair(100, rng, shift=15.0)
        m = fold_metrics([p1, p2], [t1, t2], W)
        name = TARGET_COLS[0]
        self.assertAlmostEqual(m[f"peak_err_pctbw_{name}"], 10.0 / W * 100.0, places=9)

    def test_constant_column_gives_finite_metrics(self):
        """常量列（std=0）不产生 NaN/inf。"""
        T = 100
        true = np.zeros((T, 6))
        true[:, 1] = 300.0
        pred = true.copy()
        m = fold_metrics([pred], [true], W)
        for v in m.values():
            self.assertTrue(np.isfinite(v))

    def test_weights_cover_all_subjects(self):
        self.assertEqual(len(SUBJECT_WEIGHT_N), 8)
        # 55-83 kg -> 539-814 N
        for z, w in SUBJECT_WEIGHT_N.items():
            self.assertGreater(w, 500)
            self.assertLess(w, 850)


if __name__ == "__main__":
    unittest.main()
