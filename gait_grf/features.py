"""运动学特征前端：从传感器四元数推导跨受试者可迁移的特征（ticket #7）。

数据约定（2026-09-05 实证结论）：
- 7 个 IMU 的四元数均为「相对各自上电时刻姿态」的朝向序列——单位范数、平滑、
  各环节摆幅层级符合人体先验（躯干 < 大腿 ≈ 小腿 < 足），但四种循环分量读法下
  静态旋转轴均不与任何世界轴对齐，即不存在共享重力对齐世界系。
- 由此，真解剖关节角不可恢复；但「相对自身静态基线的运动」中上电参考系在
  乘法 ΔM(t) = M(t0)⁻¹ M(t) 中严格相消，剩下纯环节运动（以自身绑定系表达），
  语义为「相对站立位的运动」，跨受试者成立。分量读法（w 位置）是全局固定
  变换，对迁移无影响，统一按 S0（列序即 wxyz）处理。
- 全零四元数帧（LQW03/04 整文件，见 constants.INVALID_TRIALS）用单位四元数
  填充，特征退化为常数，管线不中断。

特征块（feature_mode="kinematic"，共 51 维）：
1. selfrel：逐传感器相对静态基线的 rotvec（7×3 = 21 维）。角度部分干净，
   轴受绑扎旋转影响（受试者间近似一致但不保证）。
2. jointrel：相邻环节相对运动 ΔM_p⁻¹ ⊗ ΔM_c（6 关节×3 = 18 维），上电参考
   系与共模晃动相消。
3. 压力摘要（每足 6 维 = 总和/峰值/接触面积/三个索引区段和，共 12 维）。区段
   划分是启发式（无鞋垫布局坐标），线性模型可自行取舍。

feature_mode="kinematic_min"（共 25 维）：只留角度量（selfrel 幅值 7 +
jointrel 幅值 6）+ 压力摘要，完全不含受绑扎旋转污染的轴向信息。
"""

import numpy as np

from .constants import STATIC_BASELINE_FRAMES

# 传感器 -> 四元数列号（大腿/小腿/躯干原始列名 q1–q4，双足 q0–q3，
# 见 data/subjectdata/docs/subject_info.md），加载后统一按 (w,x,y,z) 处理。
_SENSOR_QIDX = {
    "right_thigh": (1, 2, 3, 4),
    "right_calf": (1, 2, 3, 4),
    "left_thigh": (1, 2, 3, 4),
    "left_calf": (1, 2, 3, 4),
    "trunk": (1, 2, 3, 4),
    "right_foot": (0, 1, 2, 3),
    "left_foot": (0, 1, 2, 3),
}

# 关节定义：(关节名, parent, child)，特征 = child 相对 parent 的基线相对运动。
_JOINT_PAIRS = (
    ("right_hip", "trunk", "right_thigh"),
    ("right_knee", "right_thigh", "right_calf"),
    ("right_ankle", "right_calf", "right_foot"),
    ("left_hip", "trunk", "left_thigh"),
    ("left_knee", "left_thigh", "left_calf"),
    ("left_ankle", "left_calf", "left_foot"),
)

# 压力索引区段（启发式三等分，无鞋垫布局坐标，仅作线性特征冗余备份）
_PRESSURE_REGIONS = ((0, 15), (15, 30), (30, 45))

# 支持的特征模式：raw=原始 120 维；kinematic=运动学前端全量（51 维）；
# kinematic_min=仅角度量+压力摘要（25 维）
FEATURE_MODES = ("raw", "kinematic", "kinematic_min")

_SELFREL_COLS = [f"{s}_selfrel_r{a}" for s in _SENSOR_QIDX for a in ("x", "y", "z")]
_JOINTREL_COLS = [f"{j}_rel_r{a}" for j, _, _ in _JOINT_PAIRS for a in ("x", "y", "z")]
_PRESS_SUMMARY_COLS = [
    f"{foot}_{stat}"
    for foot in ("right", "left")
    for stat in ("sum", "max", "area", "region1", "region2", "region3")
]
_SELFREL_ANGLE_COLS = [f"{s}_selfrel_angle" for s in _SENSOR_QIDX]
_JOINTREL_ANGLE_COLS = [f"{j}_rel_angle" for j, _, _ in _JOINT_PAIRS]


def kinematic_feature_names(mode="kinematic"):
    """按模式返回运动学特征列名（顺序与 derive_kinematic_features 输出一致）。"""
    if mode == "kinematic_min":
        return _SELFREL_ANGLE_COLS + _JOINTREL_ANGLE_COLS + _PRESS_SUMMARY_COLS
    return _SELFREL_COLS + _JOINTREL_COLS + _PRESS_SUMMARY_COLS


def load_quat_wxyz(sensor_df, sensor):
    """读取单传感器四元数为内部 (w,x,y,z) (N,4)；全零帧填单位四元数。

    统一按 S0 读法：列序即 (w,x,y,z)。其余循环读法与 S0 差一个全局固定共轭
    变换，对跨受试者迁移无影响（见模块 docstring）。
    """
    idx = _SENSOR_QIDX[sensor]
    a = sensor_df[[f"{sensor}_q{i}" for i in idx]].to_numpy(dtype=float)
    bad = np.linalg.norm(a, axis=1) < 1e-8
    if bad.any():
        a = a.copy()
        a[bad] = (1.0, 0.0, 0.0, 0.0)
    return a


def quat_multiply(q1, q2):
    """Hamilton 积，(w,x,y,z) 约定，支持广播 (...,4)。"""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conj(q):
    """共轭（=逆，单位四元数）。"""
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_rotvec(q):
    """单位四元数 -> 旋转向量（axis*angle, rad）。数值稳定写法。"""
    q = quat_normalize(q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    vec = q[..., 1:]
    norm = np.linalg.norm(vec, axis=-1)
    angle = 2.0 * np.arctan2(norm, w)
    safe_norm = np.where(norm < 1e-12, 1.0, norm)
    return vec * (angle / safe_norm)[..., None]


def static_baseline_quat(q, n_frames=STATIC_BASELINE_FRAMES):
    """前 n_frames 帧的平均旋转（单位四元数）。步态 trial 均以站立静止开始。"""
    return quat_normalize(q[:n_frames].mean(axis=0))


def static_selfrel(q, n_frames=STATIC_BASELINE_FRAMES):
    """基线相对运动四元数序列：ΔM(t) = M(t0)⁻¹ ⊗ M(t)。

    上电参考系在乘法中严格相消（见模块 docstring），返回单位四元数 (N,4)。
    """
    return quat_multiply(quat_conj(static_baseline_quat(q, n_frames)), q)


def derive_kinematic_features(sensor_df, mode="kinematic"):
    """传感器 DataFrame -> 运动学特征 (N, D) float32，列序见 kinematic_feature_names。"""
    quats = {s: load_quat_wxyz(sensor_df, s) for s in _SENSOR_QIDX}
    # ΔM(t) = M(t0)⁻¹ ⊗ M(t)：上电参考系在乘法中严格相消
    selfrel_q = {s: static_selfrel(quats[s]) for s in _SENSOR_QIDX}

    blocks = []
    if mode == "kinematic_min":
        selfrel_block = np.linalg.norm(
            np.concatenate([selfrel_q[s] for s in _SENSOR_QIDX], axis=0), axis=1
        )
        n = len(quats["trunk"])
        blocks.append(selfrel_block.reshape(7, n).T)  # (N,7) 幅值
    else:
        blocks.append(
            np.concatenate([quat_to_rotvec(selfrel_q[s]) for s in _SENSOR_QIDX], axis=1)
        )

    # 关节相对运动：ΔM_p⁻¹ ⊗ ΔM_c（上电参考系 + 共模晃动相消）
    joint_blocks = []
    for _, parent, child in _JOINT_PAIRS:
        rv = quat_to_rotvec(quat_multiply(quat_conj(selfrel_q[parent]), selfrel_q[child]))
        if mode == "kinematic_min":
            joint_blocks.append(np.linalg.norm(rv, axis=1, keepdims=True))
        else:
            joint_blocks.append(rv)
    blocks.append(np.concatenate(joint_blocks, axis=1))

    # 压力摘要：每足 总和/峰值/接触面积/三个区段和
    press = []
    for foot in ("right", "left"):
        p = sensor_df[[f"{foot}_pressure{i}" for i in range(45)]].to_numpy(dtype=float)
        total = p.sum(axis=1, keepdims=True)
        peak = p.max(axis=1, keepdims=True)
        area = (p > 0.05 * np.maximum(peak, 1e-9)).sum(axis=1, keepdims=True).astype(float)
        regions = [p[:, lo:hi].sum(axis=1, keepdims=True) for lo, hi in _PRESSURE_REGIONS]
        press.append(np.concatenate([total, peak, area] + regions, axis=1))
    blocks.append(np.concatenate(press, axis=1))

    feats = np.concatenate(blocks, axis=1)
    expected = len(kinematic_feature_names(mode))
    if feats.shape[1] != expected:
        raise AssertionError(f"特征维数 {feats.shape[1]} 与列名数 {expected} 不一致")
    return feats.astype(np.float32)
