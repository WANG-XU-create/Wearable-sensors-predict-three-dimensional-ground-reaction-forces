---
id: 0007
title: "A5 测试受试者 transductive scaler refit（evaluate 免重训自校准）"
status: closed         # open | closed
labels: [ready-for-agent]
assignee: claude
blocked_by: []
part_of: ""           # 若为 wayfinder child，填 map 的编号
created: 2026-09-06
updated: 2026-09-06
---

# 0007 — A5 测试受试者 transductive scaler refit

## Description

v2 分析 §4-A5：部署场景下可穿戴自校准（无标签）合法；直接针对 z7 类域偏移。LOSO 每折的 feature scaler 在留出受试者本人的全部对齐帧上重新 fit（mean+std 全换），target scaler 与模型权重不动，度量域偏移中「特征分布线性漂移」的可补偿部分。

**实现**（已完成 2026-09-06）：

- `gait_grf/evaluate.py`：`evaluate_run(..., scaler_refit="none")`；`"test"` 时逐折 refit feature scaler（`load_aligned_trial` 进程内缓存，predict_trial 不重复计算特征）
- CLI `--scaler-refit {none,test}`；refit 输出文件名带 `_refit-test` 后缀，不覆盖原口径结果

**验收**：`runs/ltc_kinematic_dyn` checkpoint 免重训 refit 评估，逐折对比 EXP-003 metrics.csv（重点 z7/z6）；结果按「数据备忘」条目登记 experiment-log.md。

**已知代价**：压力幅值类特征的绝对水平信息（体重差异）会被一并归一化，逐轴结果需对比原口径解读。

## Comments

### @claude — 2026-09-06

实现完成，测试全绿（新增 refit 路径烟测 ×2）。待在真实 EXP-003 checkpoint 上跑评估并记录。

### @claude — 2026-09-06

实测完成（EXP-003 checkpoint 免重训）：**负结果**。r 0.9300 → 0.9171（配对 t=−2.73，p=0.029），6/8 折下降，逐轴 6 轴全负、R_vx −0.074 最重。结论：模型依赖特征绝对水平（压力幅值 ↔ 力尺度），全量 refit 得不偿失；z7 域偏移非线性漂移主导 → B7 mounting 规整提级。详见 experiment-log.md 数据备忘条目。验收达成（跑完+记录），关单。
