---
id: 0014
title: "GitHub LNN 项目架构移植批次：MixedLTC 细胞（--cell mix）+ ltc_ncp 直读出 + AdamW/init-gain 训练配方"
status: open         # open | closed
labels: [ready-for-human]   # 实验待用户手动执行
assignee: claude
blocked_by: []
part_of: ""
created: 2026-09-19
updated: 2026-09-19
---

# 0014 — GitHub 液态神经网络项目架构移植（研究→落地）

## Description

**研究来源**（2026-09-19 克隆精读三仓，/root/lnn_research/）：

| 仓库 | 定位 | 提炼出的模式 |
|---|---|---|
| `raminmh/CfC`（1055★，官方） | Nature MI 2022 CfC 论文基准代码 | ① **mixed 模式**：门控累加器（cell state）与液态单元双状态循环——LSTM 部分的循环矩阵吃液态隐状态，累加器门控输出作为液态单元输入（HAR 基准获胜配置 hidden 256 + backbone 128×2）；② AdamW + weight decay 4e-5~2e-4；③ xavier init gain 0.67–1.35；④ 指数 LR 衰减 |
| `makramchahine/drone_causality`（93★，Harvard 真机） | 无人机视觉飞行，RNN 细胞对撞（ctrnn/ltc/cfc/mixedcfc/ncp/wiredcfc/tcn/lstm） | ① **mixedcfc 是默认细胞**；② **NCP wiring（18 inter→12 cmd→4 motor）与 NCP-wired CfC 是一等候选**；③ clipnorm=1 默认；④ AdamW wd 1e-6；⑤ 双流 trunk（CNN+MLP）后 concat |
| `mlech26l/ordinary_neural_circuits`（42★） | 2024 ONC（脉冲电路生成，RL 控制） | 与监督回归不相关，不采纳 |

**移植实现**（models.py + train.py，测试 123 全绿 +18）：

1. **`MixedLTCCell`**（`--cell mix`，作用于 `ltc`/`ltc_attn`）：官方 MixedCfcCell 的
   torch+LTC 逐时间步移植。双状态循环：`z = x·Win + h_ode·Wrec`；
   `new_cell = cell·σ(fg+forget_bias) + tanh(i)·σ(ig)`；`ode_input = tanh(new_cell)·σ(og)`；
   `ode_out, h_ode' = LTCCell(ode_input, h_ode)`（ncps 全连接 wiring，稠密语义）。
   关键接缝：液态隐状态调制门控行为、累加器输出驱动液态动力学——与现有
   「LTC 堆叠」本质不同（现状是各层独立液态循环）。
2. **`GaitLTCNCP`**（`--model ltc_ncp`，`--ncp-units`/`--ncp-sparsity`）：稠密 LTC 层
   堆叠后末层换 NCP wiring（AutoNCP(units, 6)，inter+command ≈ hidden，**motor
   状态即 6 维 GRF 输出、无 MLP 头**）——Nature MI 2020 驾驶的「可审计」构型，
   计算成本中性（区别于 2026-09-05 被否的 AutoNCP 提速论证：当时只否定了
   FLOP 节省，表征价值未测）。wiring 拓扑随 config 保存，checkpoint 重建一致。
3. **训练配方**（官方两仓一致、本项目此前缺失）：`--optimizer adamw
   --weight-decay λ`（衰减只作用于 Linear/Conv 权重矩阵；LTCCell/CfCCell 电路
   参数与 bias 全豁免——电路参数有正性量程约束，衰减会破坏）；`--init-gain g`
   （xavier_uniform 仅作用于 Linear 权重）。

**不采纳**：irregular-timespans/mask 缺失特征（本项目 100Hz 均匀采样无缺失）；
ONC 脉冲电路；NCP 中间层（AutoNCP(2H,H)：稀疏 mask 是稠密乘法，FLOP 与状态
×4 增大，成本恶性）。

## Comments

### @claude — 2026-09-19

实验阶梯（单因素纪律，全部在新数据口径 213-trial + plate_zero 上，对照组 =
EXP-012 新口径基线）：

| 实验 | 命令差异 vs 基线 | 验证假设 |
|---|---|---|
| EXP-013 | `--model ltc_attn --cell mix` | 混合细胞在完整配方上的增益（官方旗舰模式） |
| EXP-014 | `--model ltc --cell mix`（无 attn） | 混合细胞 vs EXP-007 纯 LTC 的独立效果 |
| EXP-015 | `--model ltc_ncp` | NCP 直读出 vs EXP-007（成本中性、可解释性叙事） |
| EXP-016 | `--optimizer adamw --weight-decay 1e-4 --init-gain 0.84` | 官方训练配方在共享模型上的增益（可用 ltc 快扫） |

优先级建议：EXP-013 > EXP-016（分钟级可先用 TCN 快扫 wd/gain 方向）> EXP-015 > EXP-014。
