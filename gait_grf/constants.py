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

# Qualisys 原始文件中的力板目标列（板编号命名）
PLATE_TARGET_COLS = [f"ground_force_{p}_{a}" for p in ("1", "2") for a in ("vx", "vy", "vz")]

# 规范化目标列（脚语义命名）：组1=左脚、组2=右脚，各 3 维力（vx/vy/vz，共 6 维）
# 采集时 z1–z5 左脚先踩板1、z6–z8 左脚先踩板2，因此「板编号 -> 左右脚」随受试者
# 组翻转；数据管线在读列时统一规范化为脚语义（见 data.extract_targets）。
TARGET_COLS = [f"ground_force_{f}_{a}" for f in ("left", "right") for a in ("vx", "vy", "vz")]

# 各受试者左脚先踏上的测力板编号（决定 ground_force_1/2 -> 左右脚的映射）
LEFT_FOOT_PLATE = {
    "z1": 1,
    "z2": 1,
    "z3": 1,
    "z4": 1,
    "z5": 1,
    "z6": 2,
    "z7": 2,
    "z8": 2,
}

# 各受试者体重（N）= kg × g。来源：subject_info.md 的体重表（79.5/60.0/75.5/
# 68.5/73.0/60.0/55.0/83.0 kg，已与各 Qualisys calibration_report.csv 核对一致）。
# 用于 %BW 归一化指标（跨受试者公平比较）。
G = 9.80665
SUBJECT_WEIGHT_N = {
    "z1": 79.5 * G,
    "z2": 60.0 * G,
    "z3": 75.5 * G,
    "z4": 68.5 * G,
    "z5": 73.0 * G,
    "z6": 60.0 * G,
    "z7": 55.0 * G,
    "z8": 83.0 * G,
}

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
DEFAULT_REFINE_RADIUS = 10  # 帧（onset 锚点附近的互相关精修半径）
