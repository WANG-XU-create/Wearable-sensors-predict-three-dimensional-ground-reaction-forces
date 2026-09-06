---
id: 0005
title: "LSTM/TCN 基线：接入 train.py --model，与 LTC 公平对比"
status: closed         # open | closed
labels: [ready-for-agent]
assignee: claude       # 空 = 未分配；claim 时填名字
blocked_by: []        # 被哪些 issue 阻塞
part_of: ""           # 若为 wayfinder child，填 map 的编号
created: 2026-08-30
updated: 2026-08-30
---

# 0005 — LSTM/TCN 基线

## Description

`models.py` 已实现 `GaitLSTM`（nn.LSTM 堆叠）与 `GaitTCN`（因果空洞卷积残差），统一为 `seq2seq (B,T,F)->(B,T,6)` 接口，与 `GaitLTC` 同构（hidden/层数/dropout 参数一致），用于公平对比。但尚未接入训练入口：

- `train.py:120` 仍硬编码 `GaitLTC`
- `train.py:373` `--model` 仍 `choices=["ltc"]`
- TCN 的 `kernel` 参数（默认 5）需经 `make_model` 透传

**待办**：

1. `train.py` 改用 `make_model(name, input_size, output_size, hidden, layers, dropout)`，`--model` 加 `lstm` / `tcn`
2. TCN `kernel` 经 CLI 透传
3. 三模型同超参（hidden128×2、dropout0.3、epochs100、patience10）各跑一轮 LOSO，产出对比指标

**注意**：GPU 正在跑 #4，LSTM/TCN 的完整 LOSO 应在 #4 结束后再排队（避免抢 GPU）；接入代码本身现在就可以做，不受影响。

## Comments

### @claude - 2026-08-30

已 claim。待办 1 已在 f077caf 完成：`train.py` 改用 `make_model`，`--model` 已支持 `ltc/lstm/tcn`，`models.py` 同提交（GaitLSTM/GaitTCN + 因果性测试，33 测试全绿）。待办 2（TCN kernel CLI 透传）实现中；待办 3（基线 LOSO 运行）按票面注意排在 #4 结束后。

### @claude — 2026-09-06

v2 分析 §4-B8 拍板：基线运行沿用 **kinematic_dyn（123 维）** 特征模式（与 EXP-003 同构，公平对比），而非本票原写的 kinematic。代码侧（待办 1/2）早已就绪，待办 3 的运行命令：

    python -m gait_grf.train --model lstm --features kinematic_dyn --hidden 128 --layers 2 --dropout 0.3 --epochs 100 --out-dir runs/lstm_kinematic_dyn
    python -m gait_grf.train --model tcn  --features kinematic_dyn --hidden 128 --layers 2 --dropout 0.3 --epochs 100 --out-dir runs/tcn_kinematic_dyn

由 agent 代跑（远快于 LTC），完成后按 EXP-004/005 登记 experiment-log.md 并关单。

### @claude — 2026-09-06

待办 3 完成（EXP-004/005，已登记 experiment-log.md）：**LSTM r 0.9162 弱于 LTC 0.9300；TCN r 0.9370 ≥ LTC（6/8 折正向，p=0.067），RMSE −3.0%，R_vx +0.057，z7 0.801 历史新高**，且 8 折仅 ~1 min。要点：短感受野（130 ms）因果卷积不输 ODE 递归；R_vx 并非纯数据质量问题；TCN 成为并列候选主模型 + 部署友好。关单。
