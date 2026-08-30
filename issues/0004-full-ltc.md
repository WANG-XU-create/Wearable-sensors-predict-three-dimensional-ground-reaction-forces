---
id: 0004
title: "完整 LTC 训练：hidden 128 × 2 层、dropout 0.3、LOSO 8 折"
status: open          # open | closed
labels: []            # 正在后台训练，非 triage 状态
assignee: ""          # 空 = 未分配；claim 时填名字
blocked_by: []        # 被哪些 issue 阻塞
part_of: ""           # 若为 wayfinder child，填 map 的编号
created: 2026-08-30
updated: 2026-08-30
---

# 0004 — 完整 LTC 训练

## Description

tracer（#3，hidden 32×1）已跑通 LOSO，本 ticket 把 LTC 扩到既定规模并产出正式结果。

- 模型：`GaitLTC(hidden=128, layers=2, dropout=0.3)`
- 训练：LOSO 8 折，每折训练 7 人内留 15% trial 做验证；验证损失早停（patience=10）+ 恢复最优权重
- 数据：`window=100`、`step=10`、对齐精修半径 10；特征 StandardScaler，目标 z-score，评估反变换回 N
- 输出（`runs/ltc_full_loso8/`）：`metrics.csv`、`summary.json`、8 张预测图、8 个 checkpoint

**当前状态（2026-08-30）**：正在后台训练：

```
python -m gait_grf.train --data-root data/subjectdata --out-dir runs/ltc_full_loso8 \
    --hidden 128 --layers 2 --dropout 0.3 --epochs 100 --patience 10
```

fold1（约 29 分钟）已完成，8 折预计共 3.5–4 小时；`metrics.csv` / `summary.json` 要等全部折结束才写出。

**验收**：`metrics.csv` 8 行齐全、无 NaN；对比 tracer 基线（合成幅值 r 0.70–0.91，z7 最弱），确认 hidden128×2 是否带来提升。

## Comments
