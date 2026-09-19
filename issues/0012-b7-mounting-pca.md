---
id: 0012
title: "B7 mounting 规整：kinematic_dyn_pca（探针否定）"
status: closed         # open | closed
labels: [ready-for-agent]
assignee: claude
blocked_by: []
part_of: ""
created: 2026-09-06
updated: 2026-09-06
---

# 0012 — B7 逐传感器 PCA 姿态规整（v2 §4-B7）

## Description

动机：A5 负结果与 EXP-010 的 z7 退化共同指向「特征在传感器系、目标在实验室系」的残余错配。v2 B7 方案：把各传感器/关节 rotvec（含差分块）旋转到该 trial 运动主轴基（trial 级 PCA，无标签，部署对应上电自校准），消绑扎旋转偏差。

**实现**（已完成 2026-09-06）：`gait_grf/features.py` 新增 `kinematic_dyn_pca`（123 维，同 dyn 布局）+ `_pca_align_blocks`（方差降序主轴 + 三阶矩符号约定 + 右手系；常数块恒等退化；差分与常量旋转可交换故旋转后统一差分）。数学验证：不同绑扎旋转的同源信号对齐后逐位一致（允许轴符号独立判定）。

**探针结果**（LOSO Ridge 帧级，224 trial，详见 experiment-log 数据备忘）：

| 模式 | resultant r | R_vx | z7 |
|---|---|---|---|
| kinematic_dyn | **0.6838 ± 0.0883** | 0.665 | 0.468 |
| kinematic_dyn_pca | 0.5782 ± 0.0601 | 0.616 | 0.459 |

**结论：否定，全量 EXP 不跑。** PCA 对齐抹掉了「轴与实验室系方向」的绝对信息——跨受试者绑扎大体一致（A-2 时期已知），这个共性本身是可迁移特征的一部分；PCA 只剩纯形状信息，线性模型无法再把分量映射回力分量；z7 亦未改善（其偏移不是 PCA 可捕捉的刚体旋转）。

## Comments

### @claude — 2026-09-06

探针否定（−0.106，远超噪声），按探针决策门惯例不烧全量。z7 的 mounting 问题需要别的路径（EXP-011 门控版看注意力是否是 z7 退化主因；或数据层逐受试者绑扎复核）。关单。
