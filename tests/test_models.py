"""ticket #5：LSTM/TCN 基线模型测试（与 GaitLTC 同构对比）。"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gait_grf.models import GaitLSTM, GaitLTC, GaitTCN, make_model

B, T, F, OUT = 4, 50, 120, 6


class TestShapes(unittest.TestCase):
    def test_all_models_seq2seq_shape(self):
        x = torch.randn(B, T, F)
        for name in ("ltc", "lstm", "tcn"):
            with self.subTest(model=name):
                model = make_model(name, input_size=F, output_size=OUT,
                                   hidden=16, layers=2, dropout=0.3)
                model.eval()
                with torch.no_grad():
                    y = model(x)
                self.assertEqual(tuple(y.shape), (B, T, OUT))

    def test_invalid_layers_raises(self):
        for name in ("ltc", "lstm", "tcn"):
            with self.subTest(model=name):
                with self.assertRaises(ValueError):
                    make_model(name, input_size=F, hidden=8, layers=0)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            make_model("gru", input_size=F)


class TestTCNCausal(unittest.TestCase):
    def test_future_perturbation_does_not_affect_past_outputs(self):
        """因果性：扰动 t>=k 的输入，t<k 的输出必须逐位不变。"""
        torch.manual_seed(0)
        model = GaitTCN(input_size=16, output_size=OUT, hidden=24, layers=4, dropout=0.0)
        model.eval()
        x = torch.randn(2, 60, 16)
        k = 30
        x2 = x.clone()
        x2[:, k:, :] += 1.0
        with torch.no_grad():
            y1, y2 = model(x), model(x2)
        self.assertTrue(torch.allclose(y1[:, :k], y2[:, :k], atol=1e-5),
                        "未来帧的扰动影响了过去帧的输出——违反因果性")
        # 至少 k 附近应当有变化（确认扰动确实传到了输出）
        self.assertFalse(torch.allclose(y1[:, k:], y2[:, k:], atol=1e-5))


class TestBackward(unittest.TestCase):
    def test_gradients_flow_all_models(self):
        x = torch.randn(B, T, F)
        for name in ("ltc", "lstm", "tcn"):
            with self.subTest(model=name):
                model = make_model(name, input_size=F, output_size=OUT,
                                   hidden=8, layers=2, dropout=0.0)
                model.train()
                loss = torch.nn.functional.mse_loss(model(x), torch.zeros(B, T, OUT))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(len(grads) > 0)


class TestTrainerIntegration(unittest.TestCase):
    """make_model 接进 train_one_fold 的最小烟测（各模型 1 轮）。"""

    def test_train_one_fold_each_model(self):
        from gait_grf.data import discover_trial_pairs
        from gait_grf.train import split_val_trials, train_one_fold

        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(3)
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from tests.test_train import _write_subject_fixture

            _write_subject_fixture(d, "z1", "LQW", 3, rng)
            pairs = discover_trial_pairs(d, subjects=["z1"])
            fit, val = split_val_trials(pairs, 0.25, np.random.default_rng(0))
            for name in ("lstm", "tcn"):
                with self.subTest(model=name):
                    cfg = {
                        "model": name, "window": 100, "step": 10, "hidden": 8,
                        "layers": 1, "dropout": 0.0, "epochs": 1, "patience": 0,
                        "batch_size": 8, "lr": 1e-3,
                    }
                    model, _, history = train_one_fold(fit, val, cfg, torch.device("cpu"))
                    self.assertEqual(len(history), 1)
                    self.assertTrue(np.isfinite(history[0]["train_loss"]))


if __name__ == "__main__":
    unittest.main()
