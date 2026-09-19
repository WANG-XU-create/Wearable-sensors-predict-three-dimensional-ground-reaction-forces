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

动力学扩展（v2 分析 §4 A-2，2026-09-05）——rotvec 的 trial 内数值差分
（np.gradient 中心差分、边缘二阶单侧；本函数按单 trial 调用，天然不跨边界），
语义近似环节角速度/角加速度（GRF ≈ m·a_COM，加速度信息直接相关）：
- feature_mode="kinematic_vel"（72 维）：kinematic + selfrel rotvec 一阶差分；
- feature_mode="kinematic_acc"（93 维）：+ selfrel rotvec 二阶差分（角加速度）；
- feature_mode="kinematic_dyn"（123 维）：+ jointrel rotvec 一阶差分 +
  压力摘要一阶差分（加载率）。
三档阶梯用于探针消融：姿态 -> +速度 -> +加速度 -> +关节速度/压力变化率。

mounting 规整（v2 分析 §4 B-7，2026-09-06）——A5 负结果与 EXP-010 的 z7 退化
共同指向「特征在传感器系、目标在实验室系」的残余错配（IMU 相对上电姿态、
无重力对齐，实验室航向不可观测，只能从运动统计近似）：
- feature_mode="kinematic_dyn_pca"（123 维，同 dyn 布局）：逐传感器/逐关节把
  rotvec（含差分块）旋转到该 trial 运动主轴基上（trial 级 PCA，无标签，部署
  对应「上电后采集几步做自校准」）；轴语义从传感器轴变为「主摆动/次摆动/
  面外」方向，消绑扎旋转偏差。符号约定见 _pca_align_blocks。
"""

import numpy as np

from .constants import FEATURE_COLS, SAMPLE_RATE_HZ, STATIC_BASELINE_FRAMES

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
# kinematic_min=仅角度量+压力摘要（25 维）；kinematic_vel/acc/dyn=动力学
# 扩展阶梯（72/93/123 维，见模块 docstring）；kinematic_dyn_pca=dyn 的
# mounting 规整版（B-7，逐块 PCA 对齐，123 维）
FEATURE_MODES = (
    "raw",
    "kinematic",
    "kinematic_min",
    "kinematic_vel",
    "kinematic_acc",
    "kinematic_dyn",
    "kinematic_dyn_pca",
)

_SELFREL_COLS = [f"{s}_selfrel_r{a}" for s in _SENSOR_QIDX for a in ("x", "y", "z")]
_JOINTREL_COLS = [f"{j}_rel_r{a}" for j, _, _ in _JOINT_PAIRS for a in ("x", "y", "z")]
_PRESS_SUMMARY_COLS = [
    f"{foot}_{stat}"
    for foot in ("right", "left")
    for stat in ("sum", "max", "area", "region1", "region2", "region3")
]
_SELFREL_ANGLE_COLS = [f"{s}_selfrel_angle" for s in _SENSOR_QIDX]
_JOINTREL_ANGLE_COLS = [f"{j}_rel_angle" for j, _, _ in _JOINT_PAIRS]
# 动力学块：_d1 = 一阶差分（角速度/变化率），_d2 = 二阶差分（角加速度）
_SELFREL_VEL_COLS = [f"{s}_selfrel_r{a}_d1" for s in _SENSOR_QIDX for a in ("x", "y", "z")]
_SELFREL_ACC_COLS = [f"{s}_selfrel_r{a}_d2" for s in _SENSOR_QIDX for a in ("x", "y", "z")]
_JOINTREL_VEL_COLS = [f"{j}_rel_r{a}_d1" for j, _, _ in _JOINT_PAIRS for a in ("x", "y", "z")]
_PRESS_VEL_COLS = [
    f"{foot}_{stat}_d1"
    for foot in ("right", "left")
    for stat in ("sum", "max", "area", "region1", "region2", "region3")
]


def kinematic_feature_names(mode="kinematic"):
    """按模式返回运动学特征列名（顺序与 derive_kinematic_features 输出一致）。"""
    if mode == "kinematic_min":
        return _SELFREL_ANGLE_COLS + _JOINTREL_ANGLE_COLS + _PRESS_SUMMARY_COLS
    if mode == "kinematic":
        return _SELFREL_COLS + _JOINTREL_COLS + _PRESS_SUMMARY_COLS
    if mode == "kinematic_vel":
        return _SELFREL_COLS + _SELFREL_VEL_COLS + _JOINTREL_COLS + _PRESS_SUMMARY_COLS
    if mode == "kinematic_acc":
        return (
            _SELFREL_COLS + _SELFREL_VEL_COLS + _SELFREL_ACC_COLS
            + _JOINTREL_COLS + _PRESS_SUMMARY_COLS
        )
    if mode == "kinematic_dyn":
        return (
            _SELFREL_COLS + _SELFREL_VEL_COLS + _SELFREL_ACC_COLS
            + _JOINTREL_COLS + _JOINTREL_VEL_COLS
            + _PRESS_SUMMARY_COLS + _PRESS_VEL_COLS
        )
    if mode == "kinematic_dyn_pca":
        # 布局与 dyn 完全一致（列名同语义槽位），仅 rotvec 块被旋转到本 trial 主轴基
        return kinematic_feature_names("kinematic_dyn")
    raise ValueError(f"未知特征模式 {mode!r}，可选：{FEATURE_MODES}")


def feature_dim(mode="raw"):
    """按特征模式返回输入维数（evaluate.py 从 checkpoint config 重建模型用）。

    raw -> len(FEATURE_COLS)（120）；其余模式 -> 对应特征列数。
    优先级低于 checkpoint 内显式保存的 input_size（train 侧写入，见
    train.train_one_fold），仅在旧 checkpoint 缺该字段时作为回退。
    """
    if mode == "raw":
        return len(FEATURE_COLS)
    if mode in FEATURE_MODES:
        return len(kinematic_feature_names(mode))
    raise ValueError(f"未知特征模式 {mode!r}，可选：{FEATURE_MODES}")


def mirror_feature_perm(feature_mode="kinematic_dyn"):
    """A4 左右镜像增广的特征列置换（对合置换）：right_↔left_ 整块互换，trunk_ 原位。

    口径为朴素块交换（v2 分析 §4-A4）：步态以矢状面摆动为主，绕内外侧轴的
    pitch 分量在镜像下不变，故块交换 ≈ 真 sagittal 镜像的主导分量；roll/yaw
    分量的符号校正需要 per-sensor mounting 姿态（上电参考系无重力对齐，不可
    得，属 B7 mounting 规整范畴），不做。已知风险：若某侧信号本身弱（右足 vx），
    镜像会把弱侧复制到对侧。列置换与逐列时间差分可交换（d(xP)/dt = (dx/dt)·P），
    对 *_d1/_d2 动力学块同样成立，故差分块无需特殊处理。
    """
    names = (
        list(FEATURE_COLS) if feature_mode == "raw"
        else kinematic_feature_names(feature_mode)
    )
    idx = {n: i for i, n in enumerate(names)}
    perm = np.empty(len(names), dtype=int)
    for i, n in enumerate(names):
        if n.startswith("right_"):
            perm[i] = idx["left_" + n[len("right_"):]]
        elif n.startswith("left_"):
            perm[i] = idx["right_" + n[len("left_"):]]
        else:  # trunk_*（无对侧）等原位
            perm[i] = i
    if sorted(perm.tolist()) != list(range(len(names))) or not all(
        perm[perm[i]] == i for i in range(len(names))
    ):
        raise AssertionError(f"{feature_mode} 的镜像置换不是对合双射")
    return perm


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


def _pca_align_blocks(a, n_blocks):
    """B-7 mounting 规整：把 (N, 3*n_blocks) 的逐块 rotvec 旋转到各块本 trial 主轴基。

    每块 (N,3) 对中心化协方差做特征分解，轴按方差降序；符号约定消 PCA 二义性：
    u1 取三阶矩 Σ((r−μ)·u1)³ > 0 的方向（步态屈伸不对称给出一致符号），u3 =
    u1×u2 保证右手系。只旋转不平移（偏移交给下游 StandardScaler）。退化为常数
    方差的块（如全零四元数填充段）返回单位阵。差分与常量旋转可交换
    （d(Rr)/dt = R·dr/dt），故对旋转后的 rotvec 统一做差分即可，差分块无需单独
    处理。基由 trial 自身运动统计决定（无标签），部署对应「上电后采几步自校准」。
    """
    out = np.empty_like(a)
    eps = 1e-12
    for b in range(n_blocks):
        r = a[:, 3 * b:3 * b + 3]
        mu = r.mean(axis=0)
        X = r - mu
        cov = X.T @ X / max(len(r), 1)
        w, V = np.linalg.eigh(cov)  # 升序
        order = np.argsort(w)[::-1]  # 方差降序
        w, V = w[order], V[:, order]
        if w[0] < eps:  # 常数块：无主轴可定，原样保留
            out[:, 3 * b:3 * b + 3] = r
            continue
        u1 = V[:, 0]
        # 符号约定：主轴三阶矩为正（步态摆动屈/伸不对称性提供一致方向）
        skew = np.sum((X @ u1) ** 3)
        if skew < 0:
            u1 = -u1
        u2 = V[:, 1]
        u3 = np.cross(u1, u2)
        u2 = np.cross(u3, u1)  # 右手系正交基
        R = np.stack([u1, u2, u3])  # (3,3)，行 = 主轴
        out[:, 3 * b:3 * b + 3] = r @ R.T
    return out


def derive_kinematic_features(sensor_df, mode="kinematic"):
    """传感器 DataFrame -> 运动学特征 (N, D) float32，列序见 kinematic_feature_names。"""
    quats = {s: load_quat_wxyz(sensor_df, s) for s in _SENSOR_QIDX}
    # ΔM(t) = M(t0)⁻¹ ⊗ M(t)：上电参考系在乘法中严格相消
    selfrel_q = {s: static_selfrel(quats[s]) for s in _SENSOR_QIDX}

    if mode == "kinematic_min":
        blocks = [
            np.linalg.norm(
                np.concatenate([selfrel_q[s] for s in _SENSOR_QIDX], axis=0), axis=1
            ).reshape(7, len(quats["trunk"])).T  # (N,7) 幅值
        ]
        blocks.append(np.concatenate(
            [np.linalg.norm(rv, axis=1, keepdims=True) for rv in _joint_rotvecs(selfrel_q)],
            axis=1,
        ))
        blocks.append(_pressure_summary(sensor_df))
        feats = np.concatenate(blocks, axis=1)
    else:
        # pca 模式与 dyn 共享块结构（仅 rotvec 块被旋转），差分条件按 dyn 走
        base_mode = "kinematic_dyn" if mode == "kinematic_dyn_pca" else mode
        selfrel = np.concatenate(
            [quat_to_rotvec(selfrel_q[s]) for s in _SENSOR_QIDX], axis=1
        )  # (N,21) 基线相对姿态
        jointrel = np.concatenate(_joint_rotvecs(selfrel_q), axis=1)  # (N,18)
        if mode == "kinematic_dyn_pca":
            # B-7：逐传感器/逐关节旋转到本 trial 主轴基；差分在旋转后统一计算
            # （d(Rr)/dt = R·dr/dt，与先差分后旋转等价）
            selfrel = _pca_align_blocks(selfrel, len(_SENSOR_QIDX))
            jointrel = _pca_align_blocks(jointrel, len(_JOINT_PAIRS))
        press = _pressure_summary(sensor_df)  # (N,12)

        blocks = [selfrel]
        if base_mode in ("kinematic_vel", "kinematic_acc", "kinematic_dyn"):
            blocks.append(_diff(selfrel))  # 环节角速度
        if base_mode in ("kinematic_acc", "kinematic_dyn"):
            blocks.append(_diff(_diff(selfrel)))  # 环节角加速度
        blocks.append(jointrel)
        if base_mode == "kinematic_dyn":
            blocks.append(_diff(jointrel))  # 关节角速度
        blocks.append(press)
        if base_mode == "kinematic_dyn":
            blocks.append(_diff(press))  # 压力变化率（加载率）
        feats = np.concatenate(blocks, axis=1)

    expected = len(kinematic_feature_names(mode))
    if feats.shape[1] != expected:
        raise AssertionError(f"特征维数 {feats.shape[1]} 与列名数 {expected} 不一致")
    return feats.astype(np.float32)


def _diff(a):
    """trial 内数值差分（中心差分，边缘二阶单侧），按采样率换算为物理单位。

    返回 d/dt（如 selfrel rotvec 的差分 = 环节角速度 rad/s）。调用方保证 a
    属于单一 trial（本模块按 trial 调用，不跨边界差分）。极短序列安全退化：
    <2 帧返回全零，<3 帧用一阶边缘差分。
    """
    dt = 1.0 / SAMPLE_RATE_HZ
    if len(a) < 2:
        return np.zeros_like(a)
    if len(a) < 3:
        return np.gradient(a, dt, axis=0, edge_order=1)
    return np.gradient(a, dt, axis=0, edge_order=2)


def _joint_rotvecs(selfrel_q):
    """关节相对运动 rotvec 列表：ΔM_p⁻¹ ⊗ ΔM_c（上电参考系 + 共模晃动相消）。"""
    return [
        quat_to_rotvec(quat_multiply(quat_conj(selfrel_q[parent]), selfrel_q[child]))
        for _, parent, child in _JOINT_PAIRS
    ]


def _pressure_summary(sensor_df):
    """压力摘要 (N,12)：每足 总和/峰值/接触面积/三个区段和（列序 _PRESS_SUMMARY_COLS）。"""
    press = []
    for foot in ("right", "left"):
        p = sensor_df[[f"{foot}_pressure{i}" for i in range(45)]].to_numpy(dtype=float)
        total = p.sum(axis=1, keepdims=True)
        peak = p.max(axis=1, keepdims=True)
        area = (p > 0.05 * np.maximum(peak, 1e-9)).sum(axis=1, keepdims=True).astype(float)
        regions = [p[:, lo:hi].sum(axis=1, keepdims=True) for lo, hi in _PRESSURE_REGIONS]
        press.append(np.concatenate([total, peak, area] + regions, axis=1))
    return np.concatenate(press, axis=1)
