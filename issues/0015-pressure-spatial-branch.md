---
id: 0015
title: "压力空间双分支架构（--press-branch + kinematic_dyn_p90）：针对 vx/vz 剪切轴的 r>0.95 目标"
status: open         # open | closed
labels: [ready-for-human]
assignee: claude
blocked_by: []
part_of: ""
created: 2026-09-19
updated: 2026-09-19
---

# 0015 — 压力空间双分支架构（用户目标：6 分量各自 r>0.95）

## Description

**目标重申**（用户 2026-09-19）：左右足各分量 r>0.95；优先寻找高精度架构，
基线/种子重复押后到架构定稿。

**差距**（EXP-013 逐轴 r）：L_vy 0.966 / R_vy 0.967 已达标；缺口 =
R_vx 0.789、L_vz 0.872、R_vz 0.872、L_vx 0.905——全是剪切/内外侧轴。

**物理假设**：剪切力由足底压力中心（CoP）动态驱动；当前 kinematic_dyn 特征
把 90 通道压力压成每足 12 个标量，空间分布信息全部丢弃（subject_info.md
确认鞋垫无布局坐标，精确 CoP 不可算，但空间模式可由卷积隐式学习）。

**探针实证**（LOSO Ridge 帧级，213 trial，2026-09-19）：

| 模式 | 维数 | L_vx | R_vx | L_vz | R_vz | resultant |
|---|---|---|---|---|---|---|
| kinematic_dyn（现行） | 123 | 0.849 | 0.677 | 0.856 | 0.833 | 0.733 |
| kinematic_dyn_p90（+原始90通道） | 213 | 0.788 | **0.732** | 0.828 | 0.823 | 0.712 |

R_vx **+0.055，α∈[1e2,1e4] 三档稳健**——原始压力场里确有右足 vx 信息；
线性模型对 L_vx/L_vz 被冗余稀释（−0.03~−0.06）→ 信息在、需要非线性编码器。

**实现**（models.py/train.py/evaluate.py，测试 129 全绿 +6）：

- `PressureSpatialEncoder`：每足 45 通道 depthwise 多尺度因果时序卷积
  （3/1、7/2、15/3，对称 CausalMultiScaleProj）+ pointwise 融合，每足出
  hidden/2；严格因果（测试覆盖未来扰动不变性）。
- `GaitLTCAttn --press-branch`：特征末 90 维（右45+左45）进压力分支，
  运动学前 123 维走原卷积投影；**ReZero 零初始化门控 `press_scale`**
  （初始严格关闭=EXP-011 行为，测试验证输出与压力块内容无关；与
  attn_scale 同构，防弱信号无差别污染——A4 镜像的教训）。
- `--features kinematic_dyn_p90`（213 维 = dyn 123 + 原始 90）；
  checkpoint config 记录 press_branch，evaluate 重建，ensemble 口径校验。

## Comments

### @claude — 2026-09-19

EXP-017（用户手动跑，~2h）：= EXP-011 配方 + 压力分支，单因素。验收看
逐轴 r（重点 R_vx/L_vz/R_vz）与 |press_scale| 开度（可解释性）。
若 R_vx 显著抬升 → 与 λ=2 组合做正式配方；若门开度≈0 且无增益 → 假设否定，
线性稀释结论即终局。
