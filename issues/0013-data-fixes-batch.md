---
id: 0013
title: "数据层修复批次：z7 鞋垫故障排除（D1）+ 力板校零（D2）+ 训练侧 scheduler/clip（③）+ evaluate per-trial/ensemble（M3）"
status: open         # open | closed
labels: [ready-for-human]   # 实验命令待用户手动执行
assignee: claude
blocked_by: []
part_of: ""
created: 2026-09-13
updated: 2026-09-13
---

# 0013 — 数据层修复批次（D1/D2）+ ③ 号训练侧改进 + evaluate 扩展

## Description

**动机**（2026-09-13 数据诊断，两项一直排队未做的排查一次做完）：

1. **z7 崩坏根因 = 左鞋垫硬件故障**。z7（ZWJ）trial 10–20（11/29）左足压力峰
   15.6→11.5 unit（−26%），而测力台左足 vy 峰 603→598 N 纹丝不动（步态未变，
   仅鞋垫增益中途掉了又恢复，典型移位/失接触）；右足鞋垫全程稳定。这 11 个
   trial 的「低压力↔正常力」矛盾样本同时污染训练（其余 7 折）与评估（z7 折
   左 vy 崩坏，实验日志记录峰值 594→280 N）。
2. **全数据集力板 vx/vz 通道未校零**。8 人 × 双板无接触期（vy≈0）vx 恒读
   −25~−38 N、vz 恒读 −3~−11 N（帧间 std 0.1–0.6 N，纯常数）；vy 通道仅
   −0.1~−2.4 N。所有 vx/vz 目标带每人不同的恒定偏置，模型只能学到群体均值，
   残余 ±5 N 不可预测误差进 RMSE/峰值误差，且「摆动相预测 −32 N 剪切力」
   物理上错误。

**实施**（代码+测试已完成，105 全绿）：

- **D1** `constants.INVALID_TRIALS` 登记ZWJ10–20（同 LQW03/04 先例，用户
  2026-09-13 拍板整 trial 排除）；数据集 224→213；`discover_trial_pairs`
  加 `include_invalid` 诊断开关（evaluate `--include-invalid` 透传）。
- **D2** `data.extract_targets(plate_zero=True)`：逐 trial 逐足以无接触帧
  （|vy|<10N）三轴中位数扣零漂；`train --plate-zero`（默认开，
  `--no-plate-zero` 恢复历史口径）；贯穿 `load_aligned_trial` 缓存键 /
  `GRFSequenceDataset` / `predict_trial` / evaluate `--plate-zero auto|on|off`
  （auto=按 checkpoint 训练口径，旧 checkpoint 回退 False，覆盖不一致时告警）。
- **③** `train --lr-scheduler none|plateau|cosine`（plateau =
  ReduceLROnPlateau factor 0.5 / patience 3，适配早停节奏）+
  `--grad-clip`（默认 0 关）；history 逐轮记录 lr；进 checkpoint config。
- **M3** `evaluate --per-trial`（逐 trial 指标表，数据诊断基建）+
  `--ensemble DIR[,..]`（多种子 checkpoint 逐 trial 预测平均，成员口径字段
  强校验）；输出文件名带 `_ens` / `_withinvalid` 后缀。

**验收**：重跑 LTC 主配方在修复后数据上的全量 LOSO（EXP-012），对照
EXP-011（0.9365）看 z7 折与整体提升；vy 峰值误差在 λ=2 下是否回收。

## Comments

### @claude — 2026-09-13

诊断脚本结论（均为无标签分析，只读传感器/测力台原始数据）：

| 诊断项 | 结论 |
|---|---|
| z7 左鞋垫增益 | trial 10–20 掉 26%，21 起恢复；力峰全程稳定 → 硬件故障实锤 |
| 力板 vx 零漂 | 8 人双板全部 −25~−38 N（无接触期常数） |
| 力板 vz 零漂 | −3~−11 N；vy ≤2.4 N（可忽略但一并校零） |
| 步态节律/压力-体重比 | z7 与全员无异常（排除步态突变解释） |
| 右足 vx | 偏置不可预测部分仅 ~5 N，不解释 r 0.774 的弱势（信号本身弱 + 域异质），维持数据层后续排查 |

z6 折高 RMSE（19.7 %BW）高 r（0.966）指向幅值回归问题（60 kg 轻体重 ÷
小体重放大绝对误差），列为后续观察项，不在本批次。

预期：z7 折 r 0.78→~0.9 区间；整体 mean r 0.9365→0.95+；vx/vz RMSE 与
峰值误差小幅改善；vy 峰值看 λ=2。
