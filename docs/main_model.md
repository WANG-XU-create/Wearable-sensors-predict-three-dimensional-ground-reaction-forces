# main.py 模型架构分析与改进建议

> 分析对象:`/root/autodl-tmp/main.py` — LTC(液态时间常数网络)+ 自注意力的 3D GRF 预测模型
> 任务:22 维关节运动学 + 38 维足底压力/COP → 左右脚各 3 分量(ML/V/AP)共 6 路 GRF,逐帧回归
> 整理日期:2026-09-06

---

## 一、整体架构

模型是"多尺度因果卷积输入投影 → 堆叠 LTC 循环层 → 自注意力上下文增强 → 逐帧 MLP 回归头"的混合架构。

```
(b, 200, 60) 原始特征
    │  22 关节运动学 + 16左足底压力 + 3左COP/合力 + 16右压力 + 3右COP
    ▼
CausalInputProj ──── (b, 200, 1024=4H)   ← 并行因果卷积,替代第0层的逐帧 weight_ix matmul
    │
    ▼  沿时间逐步递归 (Python for 循环, t=0..199)
LTC Layer 0 ──► LTC Layer 1 ──► ... (num_layers=2, 层间有残差 + LayerNorm)
    │
    ▼  堆叠每步顶层隐状态
ltc_seq (b, 200, 256)
    │
    ├──► SimpleSelfAttention (8头 + 可学习相对位置偏置) ──► 残差 + LayerNorm ──► enhanced
    │
    └──► 门控跳连: enhanced + sigmoid(skip_gate)·skip_proj(input_proj_all)
    ▼
FC 头 (256→256→128→6) ──► (b, 200, 6) 逐帧输出 6 路 GRF(左右脚各 ML/V/AP)
```

关键超参数:`hidden_size=256`,`num_layers=2`,`num_heads=8`,`dropout=0.5`,`seq_len=200`,滑动窗步长 50,batch 256。

数据侧:9 名受试者(S04–S12)× 9 个速度(步行 0.9–5.4 km/h、跑步 6.3–9.9 km/h)的 CSV 全部 `pd.concat` 拼接后统一滑窗;StandardScaler 只 fit 训练部分;按序列序号前 80% / 后 20% 划分。

---

## 二、模块细节

### 2.1 CausalInputProj —— 多尺度因果深度可分离卷积输入投影

- 先拼一阶差分 `delta = x[t] - x[t-1]`,通道翻倍 60→120,给模型显式的速度信息;
- 三个尺度的 depthwise 因果卷积:kernel/dilation = 3/1、7/2、15/3,感受野分别约 3、13、43 帧(因果填充全部在左侧,pad 数值 (2,0)/(12,0)/(42,0) 正确);
- 三尺度拼接(360 通道)后 pointwise 卷积投影到 4H=1024,**一次性算出整个序列的第 0 层输入投影**——这是对循环网络逐帧 matmul 的关键优化。注意 1024 恰好等于 LTCCell 4 个门拼接后的维度,直接作为 cell 的 `input_proj` 输入。

### 2.2 LTCCell —— LSTM 门控 + ODE 漏积分的混合体

```
门计算:  与 LSTM 完全一致 (i/f/g/o 四门,合并成 2 次 matmul)
细胞态:  new_cell = f·c + i·g          ← 标准 LSTM
隐状态:  dh = (o·tanh(new_cell) − h) / |τ|
         h_new = h + dh                ← 欧拉一步:dh/dt = (目标态 − h)/τ
```

- 隐状态不再是 LSTM 的直接输出 `o·tanh(c)`,而是以时间常数 τ 向该平衡态**指数趋近**;τ 逐单元可学习,取 abs 保证正性,初始化为 1;步长固定(隐含 Δt=1);
- 严格说这不是 Hasani et al. 的原生 LTC(原生 LTC 的时间常数是**输入依赖**的,公式形式也不同),更准确的描述是"leaky-integrator 化的 LSTM"。设计动机即"液态"特性:不同单元用不同 τ 捕捉不同时间尺度的动力学;
- 门偏置初始化沿用 LSTM 经典技巧:遗忘门偏置 = 1。

### 2.3 LTCGRFModel 主干

- 2 层堆叠,**第 0 层吃卷积输出(1024 维),深层吃上一层隐状态**(逐帧 `weight_ix` matmul);
- 每层输出过 LayerNorm;第 0 层是 `norm(h)`,深层是 `norm(h + prev_h)` 残差。注意归一化后的 h 会作为状态传给下一时间步,"状态"已不是纯 ODE 状态;
- **自注意力是双向的**(非因果),对整段 200 帧做全局上下文聚合,带可学习相对位置偏置(`rel_pos_bias`,形状 `(heads, 2L-1)`),用 SDPA 计算;残差 + LayerNorm;
- **门控跳连**:局部卷积特征 `input_proj_all` 经 1024→256 线性投影,乘 `sigmoid(skip_gate)` 加到注意力输出上——"快路径"(卷积局部特征)与"慢路径"(LTC+attention 长程动态)显式融合;
- 输出头为 3 层 MLP(ReLU + Dropout 0.5),逐帧输出 6 个 GRF 分量。

### 2.4 参数量估算(hidden=256 时约 2.0M)

| 模块 | 约参数量 |
|---|---|
| CausalInputProj(含 pointwise 360→1024) | ~373K |
| 2× LTCCell(weight_ix + weight_hx 各 1024×256) | ~1.05M |
| Self-Attention(qkv+out+位置偏置) | ~266K |
| skip_proj(1024→256) | ~262K |
| FC 头 | ~99K |

设计逻辑总结:循环部分负责时间动力学建模,注意力负责全局步态上下文,跳连保证高频局部信息不被循环平滑掉——结构有想法,主要问题集中在实现层面而非架构本身。

---

## 三、主要问题与解决方案

### 问题 1:第 0 层 LTCCell 的 weight_ix 是死参数

`forward` 中第 0 层直接吃已投影的 `input_proj_all`,其 `weight_ix`(1024×256 ≈ 26 万参数)从未被使用——被注册、收不到梯度,白占内存和 weight_decay。

**修复(一行级):** 给 `LTCCell` 加开关,第 0 层不创建 `weight_ix`:

```python
class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_input_weight=True):
        ...
        if use_input_weight:
            self.weight_ix = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))

# 模型里:
for i in range(num_layers):
    self.ltc_cells.append(LTCCell(hidden_size, hidden_size, use_input_weight=(i > 0)))
```

### 问题 2:skip gate 初始化自相矛盾

`skip_gate` 初始化为 0,但 `sigmoid(0)=0.5`——跳连初始就是半开,与"先学主干、再逐渐打开跳连"的意图不符。

**修复:**

```python
# 方案 A:负初始化,跳连从 ≈12% 开始,先学主干
self.skip_gate = nn.Parameter(torch.tensor(-2.0))

# 方案 B(更推荐):ReZero/LayerScale 风格,逐通道从 0 开始
self.skip_scale = nn.Parameter(torch.zeros(hidden_size))
...
return self.fc(enhanced + self.skip_scale * local_feat)
```

方案 B 去掉 sigmoid,梯度路径更干净,"主干先学、跳连逐渐打开"由 0 初始化保证,是残差分支里经过验证的做法。

### 问题 3:SDPA 落不到 Flash Attention + AMP 下 mask dtype 隐患

- 传入任意浮点 `attn_mask`(相对位置偏置)时,SDPA 最多落到 memory-efficient 后端,拿不到 FA2 kernel。seq=200 且瓶颈在时间循环,**速度问题不值得动**;
- **真正要修的隐患**:AMP 下 query 是 bf16 而 `rel_pos_bias` 是 fp32,mask dtype 与 query 不匹配会报错或强制回退到最慢的 math 后端。

**修复(一行级):**

```python
bias = self.rel_pos_bias[:, rel].to(dtype=q.dtype)
out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, ...)
```

如果以后确需提速:PyTorch ≥ 2.5 可换 `flex_attention` 用 `score_mod` 实现相对位置偏置,能编译出融合 kernel——建议 profile 后再动。

### 问题 4:Python 级时间循环是训练吞吐瓶颈

200 步 × 2 层的循环,每步十几 CUDA kernel launch,Python 开销占大头。按性价比排序:

```python
# 首选:compile 会把循环展开成一张大图,跨步做 kernel 融合,此类负载常见 2~5× 提升
model = torch.compile(model, dynamic=False)   # 放在 .to(device) 之后、训练前
```

- 编译时间较长(图很大),配合早停时值得;
- 退一档方案:`torch.jit.script` 单步函数;或把 batch_size 从 256 提到 512/1024,摊薄每次 kernel launch 的 Python 开销;
- **架构级选项**(速度仍不满意时):自注意力本来就是双向的、任务是离线分析——整个模型并不因果。可以把 2 层 LTC 砍到 1 层保住"动力学"语义,另加一层非因果的膨胀 TCN/Conformer 块做并行特征提取,循环深度减半、速度近似翻倍。

### 问题 5:滑窗跨试验边界 + 划分边界窗口泄漏

所有 CSV 先 `pd.concat` 再滑窗,窗口会**跨越受试者/速度试验的拼接边界**(如 S04_run_63 的末尾拼上 S04_walk_09 的开头);且按序列序号划分时 step=50 < seq=200,相邻窗口重叠 150 帧,边界处窗口同时出现在 train/val,指标虚高。

**修复:逐文件滑窗 + 按试验分组划分:**

```python
# (a) 逐文件滑窗,窗口绝不跨试验
for fp in file_paths:
    df_part = pd.read_csv(fp)          # 清洗逻辑照旧,但只在本文件内做
    feats = df_part[self.input_cols].values.astype(np.float32)
    tgts  = df_part[self.target_cols].values.astype(np.float32)
    for s in range(0, len(feats) - seq_len + 1, step):
        self.features.append(feats[s:s + seq_len])
        self.targets.append(tgts[s:s + seq_len])
        self.trial_ids.append(fp)      # 记录来源,划分时用
```

```python
# (b) 按试验分组划分,同组窗口不跨 train/val —— 边界泄漏自然消失
from sklearn.model_selection import GroupShuffleSplit
trial_ids = np.array(dataset.trial_ids)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(np.zeros(len(trial_ids)), groups=trial_ids))
```

### 问题 6:数据划分混淆受试者因素

文件按受试者顺序拼接、再按序划分,验证集基本是最后几个受试者(S11/S12 及其部分速度)——这实际是跨受试者泛化测试(更难、更诚实),但与"同分布随机划分"不可比,报告指标时必须说明。

**修复:明确评估协议,报两套指标:**

```python
# 协议 A:跨受试者泛化(LOSO,论文主指标的标准做法)
val_subjects = ["S12"]          # 每折留出 1 人,其余训练
train_trials = [t for t in trials if t.subject not in val_subjects]

# 协议 B:同受试者新速度泛化(按速度分组划分)
gss = GroupShuffleSplit(n_splits=1, test_size=2, groups=speed_of_each_trial)
```

- **LOSO(留一受试者)**回答"对新用户准不准"——可穿戴 GRF 预测的立身之本;
- **按速度划分**回答"训练速度能否插值/外推到新速度"——数据覆盖 0.9~9.9 km/h 走跑连续谱,该问题本身有价值;
- 两套协议下 scaler 都只 fit 训练折(现有代码模式已支持,改划分方式即可复用)。

---

## 四、优先级建议

| 修复 | 成本 | 收益 |
|---|---|---|
| #5+#6 数据划分(逐文件滑窗 + 分组/LOSO) | 中 | **最大**——当前 val 指标含窗口泄漏,不可信;改完的数字才敢写进论文 |
| #2 gate 初始化 | 一行 | 训练更稳,跳连语义正确 |
| #1 死参数 | 一行 | 代码卫生,省 26 万参数 |
| #3 mask dtype cast | 一行 | 防 AMP 下报错/回退 |
| #4 torch.compile | 一行 | 训练提速 2~5×,不改任何语义 |

实施顺序建议:先做 #5/#6(数据管线,直接影响结论可信度)和 #1/#2/#3(一行级修复),#4 的 `torch.compile` 留到最后单独验证数值一致性。
