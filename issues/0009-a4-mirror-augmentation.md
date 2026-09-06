---
id: 0009
title: "A4 左右镜像增广：--mirror-aug 特征/目标整块互换"
status: open          # open | closed
labels: [ready-for-agent]
assignee: claude
blocked_by: []
part_of: ""           # 若为 wayfinder child，填 map 的编号
created: 2026-09-06
updated: 2026-09-06
---

# 0009 — A4 左右镜像增广

## Description

v2 §4-A4：交换左右 sensor 列 + 压力块 + target 前后半 → 数据翻倍 + 双边对称先验；8 人数据量下很值。

**口径**（朴素块交换，v2 文档原文）：步态以矢状面摆动为主，绕内外侧轴的 pitch 分量在镜像下不变，块交换 ≈ 真 sagittal 镜像的主导分量；roll/yaw 分量符号校正需 per-sensor mounting 姿态（不可得，属 B7 范畴），不做。target 纯块交换不做 vx 符号翻转，与特征侧同一约定。列置换与逐列时间差分可交换（d(xP)/dt = (dx/dt)·P），对 *_d1/_d2 动力学块同样成立。

**实现**（已完成 2026-09-06）：

- `gait_grf/features.py`：`mirror_feature_perm(feature_mode)`——按特征列名 right_↔left_ 前缀整块互换、trunk_ 原位；断言对合双射；raw 与全部 kinematic* 模式覆盖
- `gait_grf/data.py`：`TARGET_MIRROR_PERM = [3,4,5,0,1,2]`；`GRFSequenceDataset(..., mirror_aug=False)` 每 trial 复制镜像窗（训练窗翻倍，scaler fit 在原始+镜像全体上）
- `gait_grf/train.py`：CLI `--mirror-aug`；只对 fit_ds 生效，val 永不增广

**已知风险**（v2 原文警示）：若右足信号本身弱（R_vx 数据质量问题，5 次独立佐证），镜像会把弱侧复制到左足——逐轴结果需对比 EXP-003 判读。

**验收**：`--features kinematic_dyn --hidden 128 --layers 2 --dropout 0.3 --epochs 100 --mirror-aug` 全量 LOSO（EXP-007，注意每 epoch 窗数翻倍、墙钟约 ×1.5–2）。

## Comments

### @claude — 2026-09-06

实现完成，测试全绿（置换对合性/块映射/差分可交换 + 数据集翻倍烟测 + 入口烟测）。待用户在 screen 跑 EXP-007。

### @claude — 2026-09-06

方向预检完成（EXP-008，TCN + mirror，~2 min）：**合成幅值持平（r 0.9333 vs 0.9370，p=0.52），无增益**；且 v2 警示精确兑现——**L_vx 8/8 折全部下降（−0.021）**，右足 vx 弱信号经镜像污染左足；z7 −0.039。机制模型无关。**建议 LTC 正式实验降级/跳过**，GPU 让给 B6（EXP-009）。票保持 open 等用户拍板是否还跑 LTC 版。
