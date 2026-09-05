# LTC 模型架构审读与改进路线 v2

- 日期：2026-09-05（第二轮，承接 `ltc-improvement-analysis.md` §6 之后）
- 依据代码：`gait_grf/` 全部 9 个模块（1496 行）+ `ncps-master/`（上游 mlech26l/ncps v1.0.1 源码）
- 依据结果：`runs/ltc_kinematic/`（EXP-002，kinematic 特征 + LTC 128×2）
- 结论一句话：**表征改造（EXP-002，r 0.869→0.920）验证了 v1 的主推方向；本轮全量代码审读发现 1 个阻塞性 bug（evaluate.py 硬编码输入维度）、1 个被误判的数据层问题（right_vx 全员偏弱而非 z8 特有）、1 个错误的技术判断（AutoNCP 不省算力），并给出下一阶段的优先级路线。**
- 追记（同日）：evaluate.py bug 已修（测试 60 全绿）；拼接方式实验已完成并**否定** §2.3 削峰假设；A-2 动力学特征探针已完成且**正向**（resultant r 0.638→0.684）；EXP-003（LTC + kinematic_dyn）已完成：**r 0.9300±0.0578**（vs EXP-002 +0.010，7/8 折提升；增益低于探针预期，LTC 隐式微分吸收了部分价值）。详见 §4 与 `experiment-log.md`。

---

## 1. 现状基线（EXP-002）

| 运行 | 特征 | 模型 | LOSO 合成幅值 r | RMSE %BW |
|---|---|---|---|---|
| ltc_full（EXP-001） | raw 120 维 | LTC 128×2 | 0.8689 ± 0.0718 | 23.5 |
| **ltc_kinematic（EXP-002）** | **kinematic 51 维** | **LTC 128×2** | **0.9202 ± 0.0599** | **19.86（−15.6%）** |

- 8/8 折全升，最大提升在 z7（0.681→0.764）；vy 各折 r 0.92+。
- v1 的判断被完整证实：**瓶颈在表征，不在容量**（tracer 32×1 五轮 0.8525 → 128×2 早停 0.8689 → 换表征 0.9202）。
- 遗留：z7 仍离群（0.764）；vy 峰值误差 5–44 %BW（预测偏平滑）；每折仍 ~30 分钟。

当前架构全貌：

```
传感器CSV ──对齐──> 特征前端 ──滑窗──> 标准化 ──> 2×LTC(128) ──线性读出──> 6维GRF
 (onset锚定+互相关)  (kinematic 51维)  (100/10)   (z-score)  +dropout0.3     (z空间)
```

各环节落点：对齐 `data.py:102`；板→脚语义规范化 `data.py:166`；特征前端 `features.py:142`；
模型堆叠 `models.py:12`；训练循环（Adam 1e-3 + MSE + 早停）`train.py:97`；
重叠窗拼回 `train.py:203`；指标 `metrics.py:31`。

---

## 2. 本轮代码审读新发现

### 2.1 🔴 bug：evaluate.py 对 kinematic checkpoint 必崩（阻塞后置评估）

`evaluate.py:47`：

```python
model = make_model(
    cfg["model"],
    input_size=len(FEATURE_COLS),   # ← 硬编码 120（raw 维数）
```

checkpoint 的 `config` 里存了 `feature_mode: 'kinematic'`（`predict_trial` 也确实用它加载 51 维特征），但模型构造时忽略它、硬用 120。对 `runs/ltc_kinematic/` 的任何 checkpoint 跑 `python -m gait_grf.evaluate`，`load_state_dict` 处必然形状不匹配。

**修法**（二选一，推荐前者）：训练保存 checkpoint 时把 `input_size` 直接写进 config（防未来再加特征模式）；或 evaluate.py 从 `cfg["feature_mode"]` 经 `features.kinematic_feature_names()` 推维度。目前 EXP-002 指标由 train 直接产出故未触发，属潜伏 bug。

### 2.2 🟡 right_vx 是全员系统性偏弱，不是 z8 特有（修正 v1 §1 的归因）

v1 与 EXP-001 观察（"z8 右足 vx 0.303 崩坏，指向受试者特异绑扎"）在 EXP-002 逐轴数据下需要修正——从 `runs/ltc_kinematic/metrics.csv` 提取：

| 逐轴 Pearson r | left_vx | **right_vx** | left_vy | right_vy |
|---|---|---|---|---|
| 8 人范围 | 0.69–0.94 | **0.48–0.82** | 0.79–0.97 | 0.77–0.97 |
| 8 人均值 | ~0.84 | **~0.68** | ~0.93 | ~0.93 |

**8/8 受试者 right_vx 都明显弱于 left_vx**，z8（0.476）只是最差个例而非孤例。既然左右足共用同一模型、同侧输入结构对称，全员一致的偏侧弱项指向**数据层而非模型层**：

- 右足切向力信号本身弱（试验协议左脚先上板，右足落点可能更偏板缘，剪切分量截断）；
- 或所在板的剪切方向标定问题；
- 或全组系统性行走偏斜。

**待办**：对比原始数据左右足 vx 幅值分布；右足预测图人工复核从 z8 扩到全员。

### 2.3 🟡 峰值低估有两个来源，损失只是其一

vy 峰值误差 5–44 %BW（z7 达 44 %BW）。除 MSE 偏好均值外，`train.py:203` 的**重叠窗 uniform 平均本身是 box filter**：window=100、step=10 时每个中间帧最多混合 10 个不同时间上下文的预测，进一步削峰。

两来源可分离验证——拼接方式改动**无需重训**，直接用现有 8 个 checkpoint 重评（见 §4 A-1）。这决定了"峰值加权损失"值不值得做：若拼接改 Hann/三角权重后峰值误差大幅回收，损失改造的优先级可以下调。

> **实验结论（2026-09-05，拼接方式实验，见 `experiment-log.md`）：上述假设被否定。**
> uniform/hann/center 三种拼接指标无差异（最大逐折差 0.14 %BW，噪声水平）。根因是跨窗预测高度一致——重叠帧上各窗预测的跨窗 std 中位仅 0.00–0.09 N（帧级残差中位 4–247 N），仅 ~5 帧热身的窗与热身 95 帧的窗预测亦一致，预测主要由当前帧压力输入驱动。峰值误差全部来自模型预测本身：z4 右 vy 峰值甚至高估（775→808 N），z7 左 vy 为大范围域偏移崩坏（残差中位 247 N）而非削峰。→ 峰值问题的手段收敛到 B-6（峰值/导数加权损失）与域偏移对策（A-5/B-7），拼接保持 uniform。

### 2.4 训练配置缺口（承接 v1 §2.3）

无梯度裁剪、无 LR scheduler、单种子。v1 已列，本轮确认仍未做；补一句：LOSO 的 mean±std 是跨受试者方差，不是种子方差，论文口径需要 3 种子。

---

## 3. ncps-master 核验结果

### 3.1 身份：上游纯参考副本

`ncps-master` = 上游 mlech26l/ncps **v1.0.1**，与 site-packages 已安装版逐文件 diff 相同（验证过 `torch/ltc.py`、`torch/ltc_cell.py`、`wirings/wirings.py`），**无任何本地改动**。import 走 site-packages，不冲突；留着完整克隆只为读源码。附带：`/root/autodl-tmp` 根目录的旧原型 `LTCs.py`（TF1 风格 `tf.nn.rnn_cell`，不可运行）、`wiring.py`（ncps 拷贝）、`main.py`/`main_loso.py` 建议归档到 `legacy/` 或删除，避免与 `gait_grf` 包混淆。

### 3.2 ⚠️ 修正：AutoNCP 不省算力

v1 §2.2 的隐含假设与本项目 EXP-002 追记中"提速靠 AutoNCP/减层"的说法**是错的**。查证 `ncps-master/ncps/torch/ltc_cell.py:230-247`：torch 实现的稀疏性是**稠密参数上的逐元素 mask 乘法**：

```python
w_activation = w_param * self._sigmoid(v_pre, mu, sigma)   # (B,H,H) 稠密逐元素
w_activation = w_activation * self._params["sparsity_mask"] # mask 只置零，不减计算
```

FLOP 与显存流量完全不减少。稀疏 wiring（NCP/AutoNCP/Random）只有**正则化/可解释性**价值，没有速度收益。

### 3.3 慢的真正根源与提速正解

每折 ~30 分钟的构成：`ltc.py:175` 的 **Python 时间循环**（100 步串行）× 每步 `ode_unfolds=6` 次半隐式迭代 × 每层物化多个 `(B,128,128)` 中间张量并留在 autograd 图中（GPU 利用率仅 ~17%，kernel 启动开销主导）。

| 提速手段 | 原理 | 预期 | 成本 |
|---|---|---|---|
| `ode_unfolds` 6→2/3 | 构造参数（`LTC(..., ode_unfolds=)`），半隐式解算器平滑动力学下收敛快 | ~2–3× | 一行，需 sweep 验证精度 |
| **换 `ncps.torch.CfC`** | 闭式连续时间解，无内层展开循环，纯 Linear 运算；接口与 LTC drop-in（`out, hn = rnn(x)`） | ~5–10× | `models.py` 加一个分支 |
| `torch.compile` / CUDA graph | 摊薄小 kernel 启动开销 | 中等 | 侵入训练循环 |

### 3.4 其他库内可用件

- `mixed_memory=True`（LTC+LSTM 混合记忆）：利于长程依赖——本项目窗口仅 1s（100 帧），**暂不需要**；
- 库内另有 `Random`/`NCP`/`AutoNCP` wiring、Keras/TF/Paddle 后端（用不到）；
- `examples/bidirectional.py` 为 TF 旧版，仅思路参考（与 §4 C-10 相关）。

### 3.5 当前模型的真实身份

`models.py:24` 传 `units=128`（int）→ `FullyConnected` 稠密 wiring（`ltc.py:85`）。**现在训练的是稠密 LTC RNN，不是 Lenz et al. 的稀疏 NCP**。这本身不是问题（v1 §2.2 已分析过参数量），但写论文/报告时的术语要准确。

---

## 4. 改进路线（按优先级）

### A. 高收益低成本（先做）

| # | 内容 | 成本 | 说明 |
|---|---|---|---|
| 0 | ~~修 evaluate.py bug~~ **已完成（2026-09-05）** | 10 分钟 | 按 checkpoint `input_size`（新）/`feature_mode`（旧，经 `features.feature_dim`）推断；uniform 复算与训练口径逐折一致 |
| 1 | ~~拼接方式实验~~ **已完成（2026-09-05，否定）** | 免重训 | uniform/hann/center 无差异（§2.3 实验结论）：跨窗预测 std 中位 0.00–0.09 N，拼接不构成信息损失，保持 uniform |
| 2 | ~~动力学特征：rotvec 一/二阶差分~~ **已完成（2026-09-05，正向）** | 探针分钟级 | 新增 kinematic_vel(72)/acc(93)/dyn(123) 三档阶梯；探针 resultant r 0.638→0.675(vel)/0.684(dyn)，α 三档稳健，7/8 折正向（配对 t≈4.2）。**增益主体是一阶差分（角速度 +0.038）**，acc/关节速度/压力变化率边际 +0.008；vz 获益最大、R_vx 微降（再证右足 vx 是数据质量问题）。建议 EXP-003 用 kinematic_dyn（详见 experiment-log 探针条目） |
| 3 | **grad clip + LR scheduler** | 几行 | v1 §2.3 遗留；clip norm 1.0 + cosine 或 plateau |
| 4 | **左右镜像增广** | 低 | 交换左右 sensor 列+压力块+target 前后半 → 数据翻倍 + 双边对称先验；8 人数据量下很值；注意与 §2.2 的右足 vx 排查联动（若右足信号本身弱，镜像会把弱侧复制到左足） |
| 5 | **测试受试者 transductive scaler refit** | 低 | 部署场景可穿戴自校准（无标签）合法；直接针对 z7 类域偏移 |

### B. 中等投入

6. **峰值/导数加权损失**（**优先级上调**，2026-09-05）：拼接假设已被否定（§2.3），峰值误差全部来自模型侧（MSE 回归偏差 + 域偏移）——本项成为峰值精度的主要手段。
7. **逐传感器 PCA 姿态规整**：把各传感器 rotvec 表达到其本 trial 运动主平面基上，消传感器绑扎旋转偏差——水平轴（vx/vz）弱、z7 离群的深层原因很可能是"特征在传感器系、目标在实验室系"的残余错配（IMU 相对上电姿态、无重力对齐，实验室系航向原则上不可观测，只能从运动平面近似）。探针可验。
8. **跑齐 #5 基线**：LSTM/TCN + kinematic 特征（代码已接线，只剩跑）——0.9202 目前没有参照系。
9. **3 种子**：定显著性，论文口径。

### C. 需拍板的方向

10. **因果性**：三模型全因果（只用过去帧）。RK3588 实时部署已明确暂缓 → 当前任务本质是离线估计，**双向 LSTM/TCN 或 50–100ms lookahead 对峰值误差预计有立竿见影的改善**。可先做非因果版建立性能上界，真要做实时再退回因果版。目前未被利用的最大免费增益之一。
11. **步态周期相位重采样**：用压力鞋垫接触事件把 trial 归一化到 %gait cycle。步态文献中跨受试者泛化最有效的手段之一，但改变任务框架、对实时化不友好，属大改。
12. **CfC 替换 / ode_unfolds sweep**（§3.3）：本身不提精度，但每折 30min→几分钟意味着上面所有实验迭代速度翻数倍——作为基础设施投资划算。

### 建议执行顺序

**§4-A0 修 bug ✅ → A1 拼接实验 ✅（否定，见 §2.3）→ A2 动力学特征探针 ✅（正向 +0.046）→ EXP-003 ✅（LTC + kinematic_dyn：r 0.9300，7/8 折提升，+0.010）→ 下一步：B6 峰值加权损失 / A4 镜像增广 / A5 受试者自校准（z7） / B8 基线（沿用 kinematic_dyn 特征）；C12 提速基建可穿插 → C10 拍板因果性后决定是否上双向**。

---

## 5. 本轮结论

1. EXP-002 证实表征路线正确；下一块表征增益最可能在**动力学（差分）特征**（A-2），其次是 mounting 规整（B-7）。
2. right_vx 是全员系统性偏弱（8/8 受试者）的**数据质量问题**强信号，在模型侧继续投入前应先做数据层排查（§2.2 待办）。
3. "提速靠 AutoNCP"的判断已修正；提速正解是 ode_unfolds / CfC / compile（§3.3）。
4. 因果性（C-10）是当前架构里最大的未决免费增益，建议尽早拍板。
5. 拼接削峰假设已被免重训实验否定（2026-09-05，§2.3 / experiment-log）：跨窗预测一致（std 中位 ~0.01 N），峰值误差归因收敛到模型侧，B-6 上调；附带利好——推理对窗口起点不敏感，利于后续流式部署。
