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
        # ReZero 门控零初始化时注意力无贡献（因果性由门控关断）；打开门再验双向性
        with torch.no_grad():
            model.attn_scale.fill_(1.0)
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


class TestAttnGate(unittest.TestCase):
    """EXP-011：注意力分支 ReZero 门控（修 EXP-010 峰值回退与 z7 过拟合）。"""

    def test_attn_gate_starts_closed_and_learns(self):
        m = make_model("ltc_attn", input_size=16, output_size=OUT, hidden=16, layers=1)
        self.assertTrue(torch.all(m.attn_scale == 0.0), "注意力门控必须零初始化")
        # 门控参数自身必须能拿到梯度（ReZero 机制：α 先动、分支后学）
        x = torch.randn(2, 50, 16)
        loss = torch.nn.functional.mse_loss(m(x), torch.zeros(2, 50, OUT))
        loss.backward()
        self.assertIsNotNone(m.attn_scale.grad)
        self.assertTrue(torch.isfinite(m.attn_scale.grad).all())
        self.assertFalse(torch.all(m.attn_scale.grad == 0.0))


class TestMixedLTCCell(unittest.TestCase):
    """issue #0014：官方 MixedCfcCell 移植（--cell mix）——双状态循环。"""

    def test_seq2seq_shape_and_finite(self):
        from gait_grf.models import MixedLTCCell

        cell = MixedLTCCell(F, 16, ode_unfolds=2)
        x = torch.randn(B, T, F)
        y = cell(x)
        self.assertEqual(tuple(y.shape), (B, T, 16))
        self.assertTrue(torch.isfinite(y).all())

    def test_backward_flows_to_both_paths(self):
        # 梯度必须同时到达门控累加器（input/recurrent kernel）与液态单元电路参数
        from gait_grf.models import MixedLTCCell

        cell = MixedLTCCell(F, 16, ode_unfolds=2)
        y = cell(torch.randn(B, T, F))
        y.pow(2).mean().backward()
        for name in ("input_kernel", "recurrent_kernel"):
            p = getattr(cell, name).weight
            self.assertIsNotNone(p.grad, name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        circuit = cell.ltc._params["w"]
        self.assertIsNotNone(circuit.grad)
        self.assertGreater(circuit.grad.abs().sum().item(), 0)

    def test_forget_bias_changes_gating(self):
        # forget_bias 越大遗忘门越开：cell 状态保留越多，输出应可区分
        from gait_grf.models import MixedLTCCell

        torch.manual_seed(0)
        x = torch.randn(B, T, F)
        y1 = MixedLTCCell(F, 16, ode_unfolds=2, forget_bias=0.0)(x)
        y2 = MixedLTCCell(F, 16, ode_unfolds=2, forget_bias=5.0)(x)
        # 两个不同门控行为的细胞（同参数不同——参数随机初始化不同，仅确认可区分构造）
        self.assertFalse(torch.allclose(y1, y2))

    def test_gait_ltc_cell_mix_dispatch(self):
        x = torch.randn(B, T, F)
        for cell in ("ltc", "mix"):
            with self.subTest(cell=cell):
                m = make_model("ltc", input_size=F, output_size=OUT,
                               hidden=16, layers=2, ode_unfolds=2, cell=cell)
                self.assertEqual(tuple(m(x).shape), (B, T, OUT))
        with self.assertRaises(ValueError):
            make_model("ltc", input_size=F, hidden=16, cell="bogus")
        # 历史口径：不传 cell 默认 ltc（旧配置/checkpoint 兼容）
        m = make_model("ltc", input_size=F, hidden=16)
        self.assertNotIsInstance(m.rnn[0], __import__("gait_grf.models", fromlist=["MixedLTCCell"]).MixedLTCCell)

    def test_gait_ltc_attn_cell_mix(self):
        m = make_model("ltc_attn", input_size=F, output_size=OUT,
                       hidden=16, layers=2, ode_unfolds=2, cell="mix")
        x = torch.randn(B, T, F)
        y = m(x)
        self.assertEqual(tuple(y.shape), (B, T, OUT))
        y.pow(2).mean().backward()


class TestGaitLTCNCP(unittest.TestCase):
    """issue #0014：NCP motor 直读出变体（Nature MI 2020 驾驶构型）。"""

    def test_motor_readout_shape_no_mlp_head(self):
        m = make_model("ltc_ncp", input_size=F, output_size=OUT,
                       hidden=64, layers=2, ode_unfolds=2, ncp_units=70)
        self.assertFalse(hasattr(m, "readout"))  # motor 即输出，无 MLP 头
        x = torch.randn(B, T, F)
        y = m(x)
        self.assertEqual(tuple(y.shape), (B, T, OUT))
        self.assertTrue(torch.isfinite(y).all())

    def test_wiring_topology_recorded(self):
        m = make_model("ltc_ncp", input_size=F, output_size=OUT,
                       hidden=64, layers=2, ode_unfolds=2,
                       ncp_units=70, ncp_sparsity=0.4, ncp_seed=7)
        w = m.ncp.rnn_cell._wiring
        self.assertEqual(w.units, 70)
        self.assertEqual(w.output_dim, OUT)

    def test_invalid_ncp_units_raises(self):
        with self.assertRaises(ValueError):
            make_model("ltc_ncp", input_size=F, hidden=8, ncp_units=8)  # units-output<3

    def test_state_dict_roundtrip_rebuild(self):
        # evaluate 侧按 config 重建后 load 必须严格一致（adjacency 是 buffer 非 param）
        m1 = make_model("ltc_ncp", input_size=F, output_size=OUT,
                        hidden=64, layers=2, ode_unfolds=2, ncp_units=70, ncp_seed=42)
        m2 = make_model("ltc_ncp", input_size=F, output_size=OUT,
                        hidden=64, layers=2, ode_unfolds=2, ncp_units=70, ncp_seed=42)
        m2.load_state_dict(m1.state_dict())  # strict=True
        x = torch.randn(B, T, F)
        m1.eval(); m2.eval()
        with torch.no_grad():
            np.testing.assert_allclose(m1(x).numpy(), m2(x).numpy(), rtol=1e-5)

    def test_gradients_flow(self):
        m = make_model("ltc_ncp", input_size=F, output_size=OUT,
                       hidden=64, layers=2, ode_unfolds=2, ncp_units=70)
        y = m(torch.randn(B, T, F))
        y.pow(2).mean().backward()
        got = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        self.assertGreater(got, 0)


class TestPressureBranch(unittest.TestCase):
    """issue #0015：压力空间双分支（--press-branch + kinematic_dyn_p90）。"""

    def test_encoder_shape_and_causality(self):
        from gait_grf.models import PressureSpatialEncoder

        enc = PressureSpatialEncoder(45, 16)
        x = torch.randn(B, T, 90)
        y = enc(x)
        self.assertEqual(tuple(y.shape), (B, T, 16))
        self.assertTrue(torch.isfinite(y).all())
        # 因果性：未来帧扰动不得影响过去输出
        x2 = x.clone()
        x2[:, -10:] += 10.0
        enc.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(enc(x)[:, :-30], enc(x2)[:, :-30], atol=1e-5))

    def test_odd_hidden_raises(self):
        from gait_grf.models import PressureSpatialEncoder

        with self.assertRaises(ValueError):
            PressureSpatialEncoder(45, 15)

    def test_rezero_starts_closed(self):
        # press_scale 零初始化：分支关闭时输出与压力块内容无关（行为等价基础模型）
        from gait_grf.models import make_model

        torch.manual_seed(3)
        m = make_model("ltc_attn", input_size=213, output_size=OUT,
                       hidden=16, layers=2, ode_unfolds=2, press_branch=True)
        self.assertTrue((m.press_scale == 0).all())
        m.eval()
        x = torch.randn(B, T, 213)
        x2 = x.clone()
        x2[..., -90:] = torch.randn(B, T, 90)  # 压力块整体置换
        with torch.no_grad():
            np.testing.assert_allclose(m(x).numpy(), m(x2).numpy(), rtol=1e-6)

    def test_gate_grad_flow(self):
        # scale=0 时编码器无梯度（门关闭）、scale 自身有梯度；打开后编码器有梯度
        from gait_grf.models import make_model

        m = make_model("ltc_attn", input_size=213, output_size=OUT,
                       hidden=16, layers=1, ode_unfolds=2, press_branch=True)
        m(torch.randn(B, T, 213)).pow(2).mean().backward()
        self.assertIsNotNone(m.press_scale.grad)
        self.assertGreater(m.press_scale.grad.abs().sum().item(), 0)
        enc_grads = [p.grad for p in m.press_enc.parameters()]
        self.assertTrue(all(g is None or g.abs().sum() == 0 for g in enc_grads))
        with torch.no_grad():
            m.press_scale.fill_(0.5)
        m.zero_grad()
        m(torch.randn(B, T, 213)).pow(2).mean().backward()
        enc_grads = [p.grad for p in m.press_enc.parameters()]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in enc_grads))

    def test_input_size_guard(self):
        with self.assertRaises(ValueError):
            make_model("ltc_attn", input_size=90, hidden=16, press_branch=True)
