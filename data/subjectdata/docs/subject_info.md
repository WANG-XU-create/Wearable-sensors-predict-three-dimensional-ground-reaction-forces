# 受试者信息文档

## 概述

本项目涉及三套数据采集系统，需将不同系统中的受试者编号进行对应：

| 系统 | 数据内容 | 格式 |
|------|----------|------|
| **Noraxon** | 运动学数据：下肢关节角度（deg）、各环节加速度（mG）、环节旋转 | `.csv` |
| **Qualisys** | 测力台数据（已标定）：2 块力板的 3 维力 / 压力中心 / 力矩 | `.csv` |
| **Sensor** | 可穿戴传感器：7 组 IMU 四元数 + 双足各 45 点足底压力 | `.csv` |

> 三套系统均为 **100 Hz** 采样，同一受试者同一序号的文件一一对应，且时长一致（如 z1.01 三系统均约 2.04 s）。
> 注意：三套系统的时间列基准不同（各自导出的时间，未清零、未互相同步），跨系统对齐需另行处理。

---

## 受试者基本信息

| Noraxon | Qualisys | Sensor | 身高 (cm) | 体重 (kg) | 脚尺码 (EU) |
|---------|----------|--------|-----------|-----------|-------------|
| z1 | Z1 | LQW | 181 | 79.5 | 41.5 |
| z2 | Z2 | WY | 175 | 60.0 | 42 |
| z3 | Z3 | HYJ | 177 | 75.5 | 41 |
| z4 | Z4 | SXL | 175 | 68.5 | 41 |
| z5 | Z5 | WX | 173 | 73.0 | 41 |
| z6 | Z6 | XBL | 170 | 60.0 | 40.5 |
| z7 | Z7 | ZWJ | 172 | 55.0 | 39.5 |
| z8 | Z8 | YZJ | 180 | 83.0 | 42 |

> 体重已与 Qualisys 各受试者 `calibration_report.csv` 中的 `weight_kg` 逐个核对一致。

---

## 数据序列命名规则

### 命名格式

| 系统 | 目录 | 文件名格式 | 示例 |
|------|------|-----------|------|
| **Noraxon** | `Noraxon_csv/z{N}/` | `z{N}.{code}.csv` | `z1.01.csv` |
| **Qualisys** | `Qualisys/Z{N}_csv_calibrated/` | `z{N}_{code}_mot_100Hz.csv` | `z1_01_mot_100Hz.csv` |
| **Sensor** | `sensor/{INITIALS}/` | `{INITIALS}{code}.csv` | `LQW01.csv` |

> `{N}` 为受试者编号（1–8），`{code}` 为 2 位补零序列号（`01`–`31`），`{INITIALS}` 为受试者姓名首字母缩写。

### 以 z1 / Z1 / LQW 为例

| 序号 | Noraxon | Qualisys | Sensor |
|------|---------|----------|--------|
| 1 | `z1.01.csv` | `z1_01_mot_100Hz.csv` | `LQW01.csv` |
| 2 | `z1.02.csv` | `z1_02_mot_100Hz.csv` | `LQW02.csv` |
| ... | ... | ... | ... |
| 28 | `z1.28.csv` | `z1_28_mot_100Hz.csv` | `LQW28.csv` |

### 对应关系总结

三系统间的数据序列通过 **受试者编号** + **序列号** 唯一确定：

```
Noraxon   Noraxon_csv/z{N}/z{N}.{code}.csv
Qualisys  Qualisys/Z{N}_csv_calibrated/z{N}_{code}_mot_100Hz.csv
Sensor    sensor/{INITIALS}/{INITIALS}{code}.csv
```

---

## 各系统数据文件格式

### Noraxon（`z{N}.{code}.csv`，71 列，100 Hz）

- `Time`：时间（s），此外还有 `Activities`、`Activity Names` 两列标记列
- 关节角度（deg，LT/RT 左右侧）：`Hip Flexion`、`Hip Abduction`、`Hip Rotation Ext`、`Knee Flexion`、`Ankle Dorsiflexion`、`Ankle Inversion`、`Ankle Abduction`、`Pelvic Tilt`、`Pelvic Obliquity`、`Pelvic Rotation`、`Foot Pitch Down`、`Foot Roll Med`、`Foot Rotation Ext`
- 加速度传感器（mG）：`Pelvis Accel Sensor X/Y/Z`、`Thigh/Shank/Foot Accel Sensor X/Y/Z`（LT/RT）
- 环节旋转：`Pelvis Rot X/Y/Z`、`LT/RT Thigh/Shank/Foot Rot X/Y/Z`

### Qualisys（`z{N}_{code}_mot_100Hz.csv`，19 列，100 Hz）

测力台数据（已标定），2 块力板 × 9 列：

- `time`：时间（s）
- `ground_force_{1,2}_vx/vy/vz`：力板的 3 维力（N）
- `ground_force_{1,2}_px/py/pz`：压力中心位置（m）
- `ground_moment_{1,2}_mx/my/mz`：力矩

每个受试者目录下另有一个 `calibration_report.csv`，记录标定参数：

| 列 | 含义 |
|----|------|
| `file` / `subject` | 文件名 / 受试者 |
| `weight_kg` / `weight_N` | 受试者体重 |
| `offset_vx/vy/vz` | 各方向力偏移去除量 |
| `scale_vy` / `scale_used` / `subject_median_scale` | 垂直力按体重校准的缩放系数 |
| `n_frames` / `n_cross_plate` | 帧数 / 过板次数 |
| `force_error_N` / `warnings` | 力误差 / 警告（当前数据均为空） |

### Sensor（`{INITIALS}{code}.csv`，122 列，约 100 Hz）

- `timestamp`：时间，格式 `mm:ss.s`（如 `36:14.4`）；`packet_counter`：帧计数
- IMU 四元数（各 4 列）：`right_thigh_q1–q4`、`right_calf_q1–q4`、`left_thigh_q1–q4`、`left_calf_q1–q4`、`trunk_q1–q4`、`right_foot_q0–q3`、`left_foot_q0–q3`
  > 注意：大腿/小腿/躯干四元数列为 `q1–q4`，双足为 `q0–q3`，命名不一致但均为 4 分量四元数。
- 足底压力：`right_pressure0–44`、`left_pressure0–44`（每足 45 个传感点），以及 `right_pressure_sum` / `left_pressure_sum` 合计列

---

## 各受试者数据文件清单

三套系统的序列号完全对齐、连续无缺口：

| 受试者 | 序号范围 | Noraxon | Qualisys | Sensor |
|--------|----------|---------|----------|--------|
| z1 / Z1 / LQW | 01–28 | 28 | 28 | 28 |
| z2 / Z2 / WY | 01–20 | 20 | 20 | 20 |
| z3 / Z3 / HYJ | 01–31 | 31 | 31 | 31 |
| z4 / Z4 / SXL | 01–29 | 29 | 29 | 29 |
| z5 / Z5 / WX | 01–31 | 31 | 31 | 31 |
| z6 / Z6 / XBL | 01–30 | 30 | 30 | 30 |
| z7 / Z7 / ZWJ | 01–29 | 29 | 29 | 29 |
| z8 / Z8 / YZJ | 01–28 | 28 | 28 | 28 |
| **合计** | | **226** | **226**（另 8 个标定报告） | **226** |

> 每个受试者的最大序号即其文件数（z1、z8 到 28；z2 到 20；z3、z5 到 31；z4、z7 到 29；z6 到 30），不存在编号内部缺口，也没有某系统单独缺失某个序号的情况。

---

## 数据质量说明

- **z1 / LQW 尾部空行已清理（2026-08-29）**：该受试者原有 18 个文件在末尾带有全逗号空行（不影响有效数据，仅干扰行数统计），已全部删除，共计 5343 行：
  - Qualisys：`z1_01` ~ `z1_04`、`z1_06` ~ `z1_10` 共 9 个文件，删除 1722 行
  - Sensor：`LQW01`、`LQW03` ~ `LQW10` 共 9 个文件，删除 3621 行
  - 空行原均位于文件**末尾**，中间无缺失帧；清理时仅移除尾部空行，数据内容、CRLF 行尾均保持原样
- 清理后已全量复查：三套数据所有 CSV 文件均无空行、编号连续无缺口。
- 编码：所有数据文件均为无 BOM 的 UTF-8，可直接读取；仅 Qualisys 的 `calibration_report.csv` 带 BOM（UTF-8-sig），读取该文件时需 `encoding="utf-8-sig"`。

---

## 数据目录结构

```
/root/autodl-tmp/data/subjectdata/
├── Noraxon_csv/                  # Noraxon 运动学 CSV 数据（按受试者分子目录）
│   ├── z1/                       #   z1.01.csv ~ z1.28.csv
│   ├── z2/                       #   z2.01.csv ~ z2.20.csv
│   ├── z3/                       #   z3.01.csv ~ z3.31.csv
│   ├── z4/                       #   z4.01.csv ~ z4.29.csv
│   ├── z5/                       #   z5.01.csv ~ z5.31.csv
│   ├── z6/                       #   z6.01.csv ~ z6.30.csv
│   ├── z7/                       #   z7.01.csv ~ z7.29.csv
│   └── z8/                       #   z8.01.csv ~ z8.28.csv
│
├── Qualisys/                     # Qualisys 测力台 CSV 数据（已标定，按受试者分子目录）
│   ├── Z1_csv_calibrated/        #   z1_01_mot_100Hz.csv ~ z1_28_mot_100Hz.csv + calibration_report.csv
│   ├── Z2_csv_calibrated/        #   z2_01 ~ z2_20（同上格式）+ calibration_report.csv
│   ├── Z3_csv_calibrated/        #   z3_01 ~ z3_31 + calibration_report.csv
│   ├── Z4_csv_calibrated/        #   z4_01 ~ z4_29 + calibration_report.csv
│   ├── Z5_csv_calibrated/        #   z5_01 ~ z5_31 + calibration_report.csv
│   ├── Z6_csv_calibrated/        #   z6_01 ~ z6_30 + calibration_report.csv
│   ├── Z7_csv_calibrated/        #   z7_01 ~ z7_29 + calibration_report.csv
│   └── Z8_csv_calibrated/        #   z8_01 ~ z8_28 + calibration_report.csv
│
├── sensor/                       # 可穿戴传感器数据（按受试者分子目录）
│   ├── HYJ/                      #   HYJ01.csv ~ HYJ31.csv
│   ├── LQW/                      #   LQW01.csv ~ LQW28.csv
│   ├── SXL/                      #   SXL01.csv ~ SXL29.csv
│   ├── WX/                       #   WX01.csv ~ WX31.csv
│   ├── WY/                       #   WY01.csv ~ WY20.csv
│   ├── XBL/                      #   XBL01.csv ~ XBL30.csv
│   ├── YZJ/                      #   YZJ01.csv ~ YZJ28.csv
│   └── ZWJ/                      #   ZWJ01.csv ~ ZWJ29.csv
│
└── docs/                         # 文档
    └── subject_info.md           # 本文件
```

---

## 批量处理时的命名转换伪代码

```python
# 受试者映射
SUBJECT_MAP = {
    "z1": "LQW", "z2": "WY",  "z3": "HYJ", "z4": "SXL",
    "z5": "WX",  "z6": "XBL", "z7": "ZWJ", "z8": "YZJ",
}

# Noraxon -> Qualisys CSV
def noraxon_to_qualisys_csv(noraxon_name: str) -> str:
    # z{N}.{code} -> z{N}_{code}_mot_100Hz.csv
    # 例: z1.01 -> z1_01_mot_100Hz.csv
    subject_num, code = noraxon_name.split(".")
    return f"{subject_num}_{code}_mot_100Hz.csv"

# Noraxon -> Sensor
def noraxon_to_sensor(noraxon_name: str) -> str:
    # z{N}.{code} -> {INITIALS}{code}.csv
    # 例: z1.01 -> LQW01.csv
    subject_num, code = noraxon_name.split(".")
    initials = SUBJECT_MAP[subject_num]
    return f"{initials}{code}.csv"
```
