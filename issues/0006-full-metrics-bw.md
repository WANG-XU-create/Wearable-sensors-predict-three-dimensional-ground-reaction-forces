---
id: 0006
title: "完整指标：%BW 归一化 RMSE + 峰值/冲量/峰值时机"
status: open          # open | closed
labels: [ready-for-agent]
assignee: ""          # 空 = 未分配；claim 时填名字
blocked_by: []        # 被哪些 issue 阻塞
part_of: ""           # 若为 wayfinder child，填 map 的编号
created: 2026-08-30
updated: 2026-08-30
---

# 0006 — 完整指标

## Description

当前指标（`train.py` 的 `fold_metrics`）输出 N 单位 RMSE + Pearson r。问题是 vy（垂直）被体重量级（~550–830 N）主导，vx/vz 是 AP/ML 分量（~10–40 N），三轴 RMSE 无法横向可比，也无法跨受试者（体重 55–83 kg）公平比较。需按计划补齐：

1. **%BW 归一化**：力除以受试者体重（N → %BW）。体重来源：
   - `data/subjectdata/docs/subject_info.md` 表（z1–z8：79.5 / 60.0 / 75.5 / 68.5 / 73.0 / 60.0 / 55.0 / 83.0 kg），或
   - 各 `Qualisys/Z{n}_csv_calibrated/calibration_report.csv` 的 `weight_kg`（注意该文件带 UTF-8-sig BOM，读取需 `encoding="utf-8-sig"`）。

   建议在 `constants.py` 加 `SUBJECT_WEIGHT_N` 映射（kg × 9.80665）。
2. **主指标**：合成幅值 RMSE(%BW) + 各轴 RMSE(%BW) + Pearson r（r 与单位无关，保持现状即可）。
3. **辅助指标**：峰值（N/%BW）误差、冲量（力·时间积分）误差、峰值时机（帧偏移）。

**依赖**：实现本身可独立进行（可先在 tracer 输出上验证）；正式对比需要 #4、#5 的 LOSO 结果（软依赖，非硬阻塞）。

## Comments
