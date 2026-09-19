---
id: 0011
title: "GaitLTCAttn：借鉴 main.py 混合架构（卷积前端+双向注意力+ReZero 门控）"
status: closed         # open | closed
labels: [ready-for-agent]
assignee: claude
blocked_by: []
part_of: ""           # 若为 wayfinder child，填 map 的编号
created: 2026-09-06
updated: 2026-09-06
---

# 0011 — GaitLTCAttn 混合架构（借鉴 main.py，docs/main_model.md）

## Description

动机：TCN（0.9370）已超 LTC（0.9300），纯递归栈不是最优表征；main.py 原型是"LTC+多尺度因果卷积前端+双向自注意力+门控跳连"混合架构，文档结论"结构有想法，问题在实现层面"。

**模块借用映射**（docs/main_model.md 分析 × gait_grf/models.py）：

| main.py 模块 | 处置 | 适配说明 |
|---|---|---|
| CausalInputProj 多尺度因果卷积 | **借** | 不做 delta 拼接（kinematic_dyn 已显式含 d1/d2）；pointwise 出 hidden 而非 4H（ncps LTC 第 0 层按 hidden 逐帧消费，4H 技巧只适用其自制 cell） |
| 自制 LTCCell（leaky-LSTM 门控） | **不借** | 保持 ncps LTC（项目身份；CfC 实验已示连续时间变体需谨慎） |
| 层间 LayerNorm+残差 | **借** | 改为 seq 级残差，不碰 LTC 内部状态（避免 main.py "norm 后状态传下一步破坏 ODE 语义"问题） |
| 双向自注意力 + rel-pos 偏置 | **借** | 用户拍板双向（main.py 原样），兼作 v2 C10 非因果上界；偏置 cast 到 q dtype（main.py 问题 3 修复）；max_len=100=DEFAULT_WINDOW |
| 门控跳连 | **借（修复版）** | 文档推荐 ReZero（逐通道 zero-init 无 sigmoid，修 sigmoid(0)=0.5 半开问题）；conv 已出 hidden 维免 skip_proj |
| 3 层 MLP 头 | **借（简化）** | 两层 128→128→6 |
| dropout 0.5 / torch.compile / 数据管线 | **不借** | 单因子纪律保持 0.3；C12 暂缓 compile；我方管线本就逐 trial 滑窗 + LOSO 无泄漏（main.py 问题 5/6 不存在） |

**实现**（已完成 2026-09-06，测试 85 全绿）：`gait_grf/models.py` 新增 `CausalMultiScaleProj` / `RelPosSelfAttention` / `GaitLTCAttn`，`--model ltc_attn`、`--attn-heads`（默认 8）；evaluate 按 checkpoint config 恢复。测试含双向性锁定断言（扰动未来帧过去输出必须变化——与 TCN 因果断言相反）。

**验收**：EXP-010 `runs/ltc_attn_kinematic_dyn/`（vs EXP-009 严格单因子=架构）：主看 resultant r vs 0.9291；次看峰值误差保持 ~16.77、z7/R_vx 是否受益于全局上下文。因果性注意：结果与全因果历史对比需注明非因果。

## Comments

### @claude — 2026-09-06

实现完成，测试 85 全绿。EXP-010 后台运行中（~2h）。

### @claude — 2026-09-06

EXP-010 完成（用户手动跑）并登记 experiment-log：**均值持平（r 0.9306 vs 0.9291，p=0.88）但两极分化**——6/8 折升（z4 0.980 单折历史新高），z7 −0.058 / z1 −0.019（双向全局上下文放大域偏移）；**R_vx 0.700→0.766（8/8 折升，首个系统性移动 R_vx 的模型侧手段，超 TCN 0.740）**；L_vy 峰值 16.77→21.47 回退（注意力平滑吃掉 B6 收益）。净判断：方向有价值、配方不达标；候选后续 ①注意力分支 ReZero 门控保峰值 ②与 B7 组合救 z7 ③注意力放 TCN 上 ④暂缓。票保持 open 等拍板。

### @claude — 2026-09-06（EXP-011 关单）

门控版（EXP-011，`attn_scale` 逐通道零初始化）达成设计目标三项全中：**r 0.9365 = LTC 系历史最高**（vs 基线 0.9291 p=0.078，7/8 折 ≥ 基线，≈ TCN 0.9370）；z7 −0.057→−0.009、L_vy 峰值 21.47→16.83 完全恢复、**R_vx 0.774 保留并扩大**；逐轴 6/6 全升、RMSE 18.07 最优。门控开度 0.14–0.32（适度打开，可解释）。**新默认配方：ltc_attn 门控版 + kinematic_dyn + ou=2 + pw1**。后续候选：3 种子显著性 / λ=2 / 因果版注意力 / 数据层排查。关单。
