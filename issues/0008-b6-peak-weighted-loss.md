---
id: 0008
title: "B6 峰值加权损失：--peak-weight 幅值加权 MSE"
status: closed         # open | closed
labels: [ready-for-agent]
assignee: claude
blocked_by: []
part_of: ""           # 若为 wayfinder child，填 map 的编号
created: 2026-09-06
updated: 2026-09-06
---

# 0008 — B6 峰值加权损失

## Description

v2 §4-B6（优先级上调）：峰值误差归因收敛到模型侧（MSE 回归偏差 + 域偏移），本项为峰值精度主要手段。2026-09-06 用户拍板采用**幅值加权**形式：

```
w = 1 + λ·|y|/mean|y|   （z 空间逐元素，权重取自目标、detach；λ 起步 1.0）
```

**实现**（已完成 2026-09-06）：

- `gait_grf/train.py`：`weighted_mse` + CLI `--peak-weight`（默认 0.0 = 纯 MSE，逐字节向后兼容）
- λ>0 时训练与早停判据均用加权损失；history 额外记录 `val_mse`（plain）供跨实验可比
- λ/开关随 checkpoint config 与 summary.json 持久化

**验收**：`--features kinematic_dyn --hidden 128 --layers 2 --dropout 0.3 --epochs 100 --peak-weight 1.0` 全量 LOSO（EXP-006），主看 vy 峰值误差（EXP-003：L 18.73 / R 16.76 %BW），次看 resultant r 不明显回退。

**注意**：z7 部分峰值崩坏属域偏移，B6 预期对其无效（v2 §2.3），勿据 z7 单折否定本项。

## Comments

### @claude — 2026-09-06

实现完成，测试全绿（weighted_mse 单测 ×3 + 入口烟测）。待用户在 screen 跑 EXP-006。

### @claude — 2026-09-06

EXP-009 完成并登记（用户手动跑，λ=1.0 + ou=2，严格对照 EXP-007）：**主指标达标**——L_vy 峰值误差 18.86 → 16.77（−11%，6/8 折，z7 −6.6 最大），R_vy −0.25；**无回退**——resultant r 持平（p=0.78）、RMSE +1.7% 噪声级、L_vy 冲量误差还改善 8%。λ=1.0 采纳进默认配方；λ=2 sweep 列为可选低优先。验收达成，关单。
