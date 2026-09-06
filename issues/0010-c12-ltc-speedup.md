---
id: 0010
title: "C12 LTC 提速基建：CfC 分支 + ode_unfolds 透传"
status: closed         # open | closed
labels: [ready-for-agent]
assignee: claude
blocked_by: []
part_of: ""           # 若为 wayfinder child，填 map 的编号
created: 2026-09-06
updated: 2026-09-06
---

# 0010 — C12 LTC 提速基建（v2 §3.3）

## Description

LTC 每折 ~29 min（GPU 利用率 ~17%，瓶颈为 Python 时间循环 × ode_unfolds=6 × (B,128,128) 中间张量），拖慢所有实验迭代。按 v2 §3.3 三手段落地前两项（torch.compile 侵入训练循环，暂缓）：

**实现**（已完成 2026-09-06）：

- `gait_grf/models.py`：`GaitLTC` 增加 `ode_unfolds` 构造参数（只影响前向展开次数，不引入参数，checkpoint 兼容）；新增 `GaitCfC`（`ncps.torch.CfC` 闭式解，drop-in，default mode）；`make_model(..., ode_unfolds=6)`
- `gait_grf/train.py`：CLI `--model cfc`、`--ode-unfolds`（默认 6 = 原行为）
- `gait_grf/evaluate.py`：重建模型按 checkpoint config 恢复 ode_unfolds（旧 checkpoint 缺省回退 6）

**基准**（4 折×3 epochs，z1–z4，kinematic_dyn 128×2，RTX 4090）：

| 变体 | 墙钟 | 加速比 | 3-epoch val_loss |
|---|---|---|---|
| LTC ou=6 | 226s | 1× | 0.992 / 0.952 / 0.939 / 0.971 |
| LTC ou=3 | 131s | 1.7× | 逐位几乎一致 |
| LTC ou=2 | 98s | **2.3×** | 逐位几乎一致（第 4 位小数） |
| CfC | 56s | **4.0×** | **0.117 / 0.100 / 0.114 / 0.118**（远低于 LTC 同期） |

**关键观察**：① ode_unfolds 6→2 精度零损，纯赚 2.3×；② CfC 3 epochs 即收敛到 LTC 全程早停水平（EXP-003 best_val 0.109–0.166），闭式解早期收敛显著更快。

**验收**：全量 8 折 LOSO 精度验证（`runs/cfc_kinematic_dyn/` 与 `runs/ltc_kinematic_dyn_ou2/`，已启动）对比 EXP-003，确认 CfC r ≥ ~LTC 或明确差距后关单；若 CfC 达标，后续实验可切换主力模型。

## Comments

### @claude — 2026-09-06

实现完成，测试 80 全绿；基准如上。全量验证运行中（CfC → LTC ou=2 串行后台）。

### @claude — 2026-09-06

EXP-006（CfC 全量）完成并登记：r 0.9153 ± 0.0767，**低于 LTC 0.9300（5/8 折负向，p=0.109）**，不能替换主模型；但 8 折仅 31 min（7.6×）、早停 19–42 轮，定位收敛为快速近似参照。模型排序定局：TCN 0.9370 > LTC 0.9300 > LSTM 0.9162 ≈ CfC 0.9153。LTC ou=2 全量验证运行中（EXP-007，26+ min 处 fold 1/8 完成），完成后关单。

### @claude — 2026-09-06

EXP-007（LTC ou=2 全量）完成：**精度零损确认**——r 0.9298 vs ou=6 的 0.9300（p=0.92，3/8 折略升），RMSE +0.8% 噪声级；每 epoch 2.45×，正常早停 ~1.7h/轮（vs 3h55m）。**LTC 正式口径切换为 --ode-unfolds 2**。C12 收官：ou=2 白拿的提速，CfC 有代价（EXP-006，r −0.015）。EXP-006/007 已登记 experiment-log.md，关单。
