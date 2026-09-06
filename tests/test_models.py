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
        for name in ("ltc", "cfc", "lstm", "tcn"):
            with self.subTest(model=name):
                model = make_model(name, input_size=F, output_size=OUT,
                                   hidden=16, layers=2, dropout=0.3)
                model.eval()
                with torch.no_grad():
                    y = model(x)
                self.assertEqual(tuple(y.shape), (B, T, OUT))

    def test_invalid_layers_raises(self):
        for name in ("ltc", "cfc", "lstm", "tcn"):
            with self.subTest(model=name):
                with self.assertRaises(ValueError):
                    make_model(name, input_size=F, hidden=8, layers=0)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            make_model("gru", input_size=F)

    def test_ode_unfolds_passthrough(self):
        # ode_unfolds 只影响 LTC 前向展开次数，不引入参数：参数量与 state_dict 键不变
        # （checkpoint 兼容：旧 checkpoint 重建时默认 6，新配置降参无需重写权重）
        m6 = make_model("ltc", input_size=16, hidden=24, layers=2, ode_unfolds=6)
        m2 = make_model("ltc", input_size=16, hidden=24, layers=2, ode_unfolds=2)
        p = lambda m: sum(x.numel() for x in m.parameters())
        self.assertEqual(p(m6), p(m2))
        self.assertEqual(list(m6.state_dict()), list(m2.state_dict()))
        # state_dict 互载成功 = 权重结构完全一致
        m2.load_state_dict(m6.state_dict())
        # 其他模型忽略该参数
        n = make_model("cfc", input_size=16, hidden=24, layers=2, ode_unfolds=2)
        ref = make_model("cfc", input_size=16, hidden=24, layers=2)
        self.assertEqual(p(n), p(ref))

    def test_tcn_kernel_passthrough(self):
        # kernel 透传到 GaitTCN：核宽变化应改变参数量；非 TCN 模型忽略 kernel
        m5 = make_model("tcn", input_size=16, hidden=24, layers=2, kernel=5)
        m3 = make_model("tcn", input_size=16, hidden=24, layers=2, kernel=3)
        p = lambda m: sum(x.numel() for x in m.parameters())
        self.assertNotEqual(p(m5), p(m3))
        n = make_model("lstm", input_size=16, hidden=24, layers=2, kernel=3)
        ref = make_model("lstm", input_size=16, hidden=24, layers=2)
        self.assertEqual(p(n), p(ref))


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
        for name in ("ltc", "cfc", "lstm", "tcn"):
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
            for name in ("cfc", "lstm", "tcn"):
                with self.subTest(model=name):
                    cfg = {
                        "model": name, "window": 100, "step": 10, "hidden": 8,
                        "layers": 1, "dropout": 0.0, "epochs": 1, "patience": 0,
                        "batch_size": 8, "lr": 1e-3,
                    }
                    model, _, history = train_one_fold(fit, val, cfg, torch.device("cpu"))
                    self.assertEqual(len(history), 1)
                    self.assertTrue(np.isfinite(history[0]["train_loss"]))

    def test_train_one_fold_ltc_low_ode_unfolds(self):
        # C12：ode_unfolds=2 经 cfg 透传训练一折（跑通即可，不看精度）
        from gait_grf.data import discover_trial_pairs
        from gait_grf.train import split_val_trials, train_one_fold

        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(4)
            from tests.test_train import _write_subject_fixture

            _write_subject_fixture(d, "z1", "LQW", 3, rng)
            pairs = discover_trial_pairs(d, subjects=["z1"])
            fit, val = split_val_trials(pairs, 0.25, np.random.default_rng(0))
            cfg = {
                "model": "ltc", "ode_unfolds": 2, "window": 100, "step": 10,
                "hidden": 8, "layers": 1, "dropout": 0.0, "epochs": 1,
                "patience": 0, "batch_size": 8, "lr": 1e-3,
            }
            model, _, history = train_one_fold(fit, val, cfg, torch.device("cpu"))
            self.assertTrue(np.isfinite(history[0]["train_loss"]))




class TestLTCAttn(unittest.TestCase):
    """issue #0011：借鉴 main.py 的 LTC+卷积前端+双向注意力混合架构。"""

    def test_seq2seq_shape_and_grad(self):
        x = torch.randn(B, T, F)
        model = make_model("ltc_attn", input_size=F, output_size=OUT,
                           hidden=16, layers=2, dropout=0.3)
        model.train()
        loss = torch.nn.functional.mse_loss(model(x), torch.zeros(B, T, OUT))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        model.eval()
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (B, T, OUT))

    def test_rel_pos_bias_slicing(self):
        # L == max_len 与 L < max_len 两种前向；L > max_len 报错
        model = make_model("ltc_attn", input_size=16, output_size=OUT,
                           hidden=16, layers=1)
        for L in (model.self_attn.max_len, 40):
            x = torch.randn(2, L, 16)
            with torch.no_grad():
                y = model(x)
            self.assertEqual(tuple(y.shape), (2, L, OUT))
        with self.assertRaises(ValueError):
            model(torch.randn(2, model.self_attn.max_len + 1, 16))

    def test_attention_is_bidirectional(self):
        """双向性：扰动未来帧，过去帧输出必须变化（非因果上界的锁定断言，
        与 TestTCNCausal 的因果断言相反）。"""
        torch.manual_seed(1)
        model = make_model("ltc_attn", input_size=16, output_size=OUT,
                           hidden=16, layers=1)
        model.eval()
        x = torch.randn(2, 60, 16)
        k = 30
        x2 = x.clone()
        x2[:, k:, :] += 1.0
        with torch.no_grad():
            y1, y2 = model(x), model(x2)
        self.assertFalse(torch.allclose(y1[:, :k], y2[:, :k], atol=1e-5),
                         "未来帧扰动未影响过去帧——注意力退化成了因果的")
        self.assertFalse(torch.allclose(y1[:, k:], y2[:, k:], atol=1e-5))

    def test_rezero_skip_starts_closed(self):
        # ReZero 跳连零初始化：初始时跳连分支贡献为 0（主干先学的语义）
        m1 = make_model("ltc_attn", input_size=16, output_size=OUT, hidden=16, layers=1)
        self.assertTrue(torch.all(m1.skip_scale == 0.0))

    def test_train_one_fold_ltc_attn(self):
        from gait_grf.data import discover_trial_pairs
        from gait_grf.train import split_val_trials, train_one_fold

        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(5)
            from tests.test_train import _write_subject_fixture

            _write_subject_fixture(d, "z1", "LQW", 3, rng)
            pairs = discover_trial_pairs(d, subjects=["z1"])
            fit, val = split_val_trials(pairs, 0.25, np.random.default_rng(0))
            cfg = {
                "model": "ltc_attn", "ode_unfolds": 2, "attn_heads": 4,
                "window": 100, "step": 10, "hidden": 8, "layers": 1,
                "dropout": 0.0, "epochs": 1, "patience": 0,
                "batch_size": 8, "lr": 1e-3,
            }
            model, _, history = train_one_fold(fit, val, cfg, torch.device("cpu"))
            self.assertTrue(np.isfinite(history[0]["train_loss"]))


if __name__ == "__main__":
    unittest.main()
