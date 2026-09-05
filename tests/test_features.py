"""features.py 单测：四元数数学 + 参考系消去不变性（ticket #7）。"""

import numpy as np
import pandas as pd
import pytest

from gait_grf.constants import INVALID_TRIALS
from gait_grf.features import (
    _SENSOR_QIDX,
    kinematic_feature_names,
    load_quat_wxyz,
    quat_conj,
    quat_multiply,
    quat_normalize,
    quat_to_rotvec,
    static_baseline_quat,
    static_selfrel,
)


def axis_angle_quat(axis, angle):
    """绕 axis 旋转 angle（rad）的四元数 (w,x,y,z)。angle 为标量或 (N,) 数组。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = np.asarray(angle, dtype=float) / 2.0
    half = np.atleast_1d(half)
    q = np.stack(
        [np.cos(half)] + [np.sin(half) * a for a in axis], axis=-1
    )  # (N,4)
    return q if np.ndim(angle) else q[0]


def random_unit_quats(rng, n):
    q = rng.normal(size=(n, 4))
    return quat_normalize(q)


def make_sensor_df(quats_by_sensor, pressure_scale=1.0):
    """按 features 期望的列构造传感器 DataFrame。"""
    n = len(next(iter(quats_by_sensor.values())))
    data = {}
    for sensor, idx in _SENSOR_QIDX.items():
        for j, i in enumerate(idx):
            data[f"{sensor}_q{i}"] = quats_by_sensor[sensor][:, j]
    rng = np.random.default_rng(0)
    for foot in ("right", "left"):
        p = rng.random((n, 45)) * pressure_scale
        for i in range(45):
            data[f"{foot}_pressure{i}"] = p[:, i]
        data[f"{foot}_pressure_sum"] = p.sum(axis=1)
    return pd.DataFrame(data)


def test_quat_multiply_known_rotations():
    qz90 = axis_angle_quat([0, 0, 1], np.pi / 2)
    qy90 = axis_angle_quat([0, 1, 0], np.pi / 2)
    # 单位元
    assert np.allclose(
        quat_multiply(qz90, np.array([1.0, 0, 0, 0])), qz90, atol=1e-12
    )
    # 同轴复合：90° + 90° = 180°（绕 z）
    q = quat_multiply(qz90, qz90)
    rv = quat_to_rotvec(q)
    assert np.isclose(np.linalg.norm(rv), np.pi, atol=1e-9)
    assert np.allclose(rv / np.linalg.norm(rv), [0, 0, 1], atol=1e-9)
    # 共轭复合：q ⊗ q⁻¹ = 单位元
    assert np.allclose(
        quat_multiply(qy90, quat_conj(qy90)), [1.0, 0, 0, 0], atol=1e-12
    )


def test_quat_to_rotvec_identity_and_known():
    assert np.allclose(quat_to_rotvec(np.array([1.0, 0, 0, 0])), 0.0)
    q = axis_angle_quat([1, 2, 3], 0.7)
    rv = quat_to_rotvec(q)
    assert np.isclose(np.linalg.norm(rv), 0.7, atol=1e-9)
    assert np.allclose(rv / np.linalg.norm(rv), np.array([1, 2, 3]) / np.sqrt(14), atol=1e-9)


def test_selfrel_cancels_poweron_reference():
    """核心性质：上电参考系（任意常量旋转）在基线相对运动中严格相消。

    构造「参考系 A / 参考系 B」两套观测：B = C ⊗ A（C 为任意常量参考旋转），
    两套观测推导的 selfrel rotvec 必须逐帧一致。
    """
    rng = np.random.default_rng(42)
    n = 40
    # 真实运动：大腿绕 x 摆动 + 绕 y 小量
    motion = quat_multiply(
        axis_angle_quat([1, 0, 0], np.linspace(0, 0.6, n)),
        axis_angle_quat([0, 1, 0], np.linspace(0.2, 0, n)),
    )
    ref_a = axis_angle_quat([0.3, -0.5, 0.8], 1.1)  # 上电姿态 A
    ref_b = axis_angle_quat([-0.9, 0.2, 0.4], 2.3)  # 上电姿态 B（不同受试者/会话）
    qa = quat_multiply(quat_conj(ref_a), motion)  # 观测 = ref⁻¹ ⊗ 真实
    qb = quat_multiply(quat_conj(ref_b), motion)

    from gait_grf.features import static_selfrel

    ra = quat_to_rotvec(static_selfrel(qa))
    rb = quat_to_rotvec(static_selfrel(qb))
    assert np.allclose(ra, rb, atol=1e-9)
    # 且等于真实运动相对「前 8 帧平均姿态」（即 static_selfrel 的基线定义）的 rotvec
    truth_base = quat_normalize(motion[:8].mean(axis=0))
    truth = quat_to_rotvec(quat_multiply(quat_conj(truth_base), motion))
    assert np.allclose(ra, truth, atol=1e-9)


def test_jointrel_zero_for_rigid_common_motion():
    """相邻环节同动（无关节运动）时 jointrel rotvec 应为 0。"""
    rng = np.random.default_rng(1)
    n = 30
    ref = axis_angle_quat([0.1, 0.9, -0.4], 1.7)
    motion = quat_multiply(
        axis_angle_quat([0, 1, 0], np.linspace(0, 0.4, n)),
        axis_angle_quat([1, 0, 0], np.linspace(0.3, 0, n)),
    )
    q_parent = quat_multiply(quat_conj(ref), motion)
    q_child = quat_multiply(quat_conj(ref), motion)  # 与 parent 完全同动
    from gait_grf.features import static_selfrel

    dp = static_selfrel(q_parent)
    dc = static_selfrel(q_child)
    joint = quat_to_rotvec(quat_multiply(quat_conj(dp), dc))
    assert np.allclose(joint, 0.0, atol=1e-9)


def test_zero_quat_filled_with_identity():
    n = 12
    q = random_unit_quats(np.random.default_rng(3), n)
    q[:4] = 0.0  # 模拟 LQW03/04 的全零帧
    df = make_sensor_df({s: q for s in _SENSOR_QIDX})
    loaded = load_quat_wxyz(df, "trunk")
    assert np.allclose(loaded[:4], [1.0, 0, 0, 0])
    assert np.allclose(np.linalg.norm(loaded, axis=1), 1.0)


def test_derive_shapes_and_finiteness():
    rng = np.random.default_rng(5)
    quats = {s: random_unit_quats(rng, 50) for s in _SENSOR_QIDX}
    df = make_sensor_df(quats)
    from gait_grf.features import derive_kinematic_features

    for mode in ("kinematic", "kinematic_min"):
        f = derive_kinematic_features(df, mode=mode)
        assert f.shape == (50, len(kinematic_feature_names(mode)))
        assert np.isfinite(f).all()
        assert f.dtype == np.float32


def test_invalid_trials_registered():
    assert INVALID_TRIALS == {("z1", "03"), ("z1", "04")}
