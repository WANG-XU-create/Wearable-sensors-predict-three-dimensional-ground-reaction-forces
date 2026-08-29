"""本项目共享常量：传感器/Qualisys 列名、受试者映射、默认超参。"""

# --- 传感器输入特征列 ---
# 7 个 IMU 四元数（每个 4 分量）：大腿/小腿/躯干用 q1–q4，双足用 q0–q3
QUAT_COLS = (
    [f"right_thigh_q{i}" for i in (1, 2, 3, 4)]
    + [f"right_calf_q{i}" for i in (1, 2, 3, 4)]
    + [f"left_thigh_q{i}" for i in (1, 2, 3, 4)]
    + [f"left_calf_q{i}" for i in (1, 2, 3, 4)]
    + [f"trunk_q{i}" for i in (1, 2, 3, 4)]
    + [f"right_foot_q{i}" for i in (0, 1, 2, 3)]
    + [f"left_foot_q{i}" for i in (0, 1, 2, 3)]
)  # 28

# 双足各 45 通道压力（共 90）
PRESSURE_COLS = [f"right_pressure{i}" for i in range(45)] + [
    f"left_pressure{i}" for i in range(45)
]

# 双足压力之和（共 2）
SUM_COLS = ["right_pressure_sum", "left_pressure_sum"]

# 完整输入特征列（120 维）
FEATURE_COLS = list(QUAT_COLS) + PRESSURE_COLS + SUM_COLS

# Qualisys 目标列：左右足各 3 维力（vx/vy/vz，共 6 维）
TARGET_COLS = [f"ground_force_{p}_{a}" for p in ("1", "2") for a in ("vx", "vy", "vz")]

# Qualisys 受试者编号（z1–z8）-> sensor 姓名缩写
SUBJECT_MAP = {
    "z1": "LQW",
    "z2": "WY",
    "z3": "HYJ",
    "z4": "SXL",
    "z5": "WX",
    "z6": "XBL",
    "z7": "ZWJ",
    "z8": "YZJ",
}

# 默认滑窗/对齐超参
DEFAULT_WINDOW = 100  # 帧（1s @100Hz）
DEFAULT_STEP = 10  # 帧
DEFAULT_MAX_LAG = 100  # 帧（对齐搜索范围）
