"""序列模型：LTC 主模型 + LSTM/TCN 基线（ticket #5）+ CfC 提速变体（v2 §3.3 / C12）
+ LTCAttn 混合架构（借鉴 main.py 原型，issue #0011）
+ GitHub LNN 项目架构移植批次（issue #0014，2026-09-19 研究结论）：
  - MixedLTC 混合细胞（raminmh/CfC 官方旗舰模式 = drone_causality 默认细胞）：
    门控累加器与液态单元的双状态循环；
  - ltc_ncp（NCP motor 直读出，Nature MI 2020 驾驶 / drone_causality 一等候选）；
  - AdamW/weight-decay/init-gain 训练配方（官方两仓一致，train.py 侧）。

统一 seq2seq 回归接口：输入 (B, T, input_size) -> 输出 (B, T, output_size)，
共享训练/评估管线，仅 --model/--cell 切换。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC, CfC, LTCCell
from ncps.wirings import AutoNCP, FullyConnected


class MixedLTCCell(nn.Module):
    """官方 MixedCfcCell 的 torch + LTC 移植（raminmh/CfC train_* 的 use_mixed=True、
    drone_causality 默认细胞，issue #0014）。

    双状态循环（逐时间步）：
      1) 门控累加器：z = x·Win + h_ode·Wrec + b，new_cell = cell·σ(fg+forget_bias)
         + tanh(i)·σ(ig)。关键接缝：循环矩阵吃的是液态隐状态 h_ode（不是累加器
         自身隐状态），即液态动力学直接调制门控行为；
      2) ode_input = tanh(new_cell)·σ(og)：累加器的门控输出作为液态单元的输入；
      3) 液态单元：ode_out, h_ode' = LTCCell(ode_input, h_ode)（ncps 全连接 wiring，
         语义等价 int units 的稠密 LTC）。
    输出序列取每步 ode_out（FC wiring 下 = 液态全状态）。
    """

    def __init__(self, input_size, hidden, ode_unfolds=2, forget_bias=1.0):
        super().__init__()
        # 液态单元的输入是累加器的门控输出（hidden 维），原始输入经 input_kernel 进入
        # 门控累加器——与官方 MixedCfcCell 一致（cfc = CfcCell(units)）
        wiring = FullyConnected(hidden)
        wiring.build(hidden)
        self.ltc = LTCCell(wiring, ode_unfolds=ode_unfolds)
        self.input_kernel = nn.Linear(input_size, 4 * hidden)
        self.recurrent_kernel = nn.Linear(hidden, 4 * hidden, bias=False)
        self.hidden = hidden
        self.forget_bias = forget_bias

    def forward(self, x):
        # (B, T, F) -> (B, T, hidden)；官方语义 elapsed=1（均匀采样）
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden)
        c = x.new_zeros(B, self.hidden)
        outs = []
        for t in range(T):
            z = self.input_kernel(x[:, t]) + self.recurrent_kernel(h)
            i, ig, fg, og = z.chunk(4, dim=-1)
            c = c * torch.sigmoid(fg + self.forget_bias) + torch.tanh(i) * torch.sigmoid(ig)
            ode_in = torch.tanh(c) * torch.sigmoid(og)
            ode_out, h = self.ltc(ode_in, h, 1.0)
            outs.append(ode_out)
        return torch.stack(outs, dim=1)


class GaitLTC(nn.Module):
    """ncps.torch.LTC 堆叠 + dropout + 线性读出。

    每个 LTC 层用 int units（全隐态输出，可堆叠），层间与读出前加 dropout。
    tracer（#3）用最小配置（hidden 32、1 层）；#4 在此之上扩到 hidden 128、2 层。
    ode_unfolds 为半隐式 ODE 解算器的内层展开步数（库默认 6）；C12 提速
    （v2 §3.3）：降低该值减少每时间步的迭代开销，精度需 sweep 验证。
    该参数只影响前向计算图展开次数，不引入参数，checkpoint 兼容。
    cell="mix" 时各层换为 MixedLTCCell（官方混合细胞，issue #0014）。
    """

    def __init__(self, input_size, output_size=6, hidden=32, layers=1, dropout=0.0,
                 ode_unfolds=6, cell="ltc"):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers 必须 >= 1，得到 {layers}")
        if cell not in ("ltc", "mix"):
            raise ValueError(f"未知 cell {cell!r}，可选 ltc/mix")
        if cell == "mix":
            self.rnn = nn.ModuleList(
                [MixedLTCCell(input_size if i == 0 else hidden, hidden,
                              ode_unfolds=ode_unfolds)
                 for i in range(layers)]
            )
        else:
            self.rnn = nn.ModuleList(
                [LTC(input_size if i == 0 else hidden, units=hidden, batch_first=True,
                     ode_unfolds=ode_unfolds)
                 for i in range(layers)]
            )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, output_size)

    def forward(self, x):
        for rnn in self.rnn:
            out = rnn(x) if isinstance(rnn, MixedLTCCell) else rnn(x)[0]
            x = self.dropout(out)
        return self.readout(x)


class GaitCfC(nn.Module):
    """ncps.torch.CfC（闭式连续时间解）堆叠 + dropout + 线性读出。

    CfC 与 LTC 同为连续时间 RNN，但用闭式近似解替代 LTC 的半隐式 ODE
    数值解算，无内层展开循环、纯 Linear 运算——v2 §3.3 C12 的主要提速手段
    （预期 5–10×），接口与 LTC drop-in。default mode 即闭式解。
    """

    def __init__(self, input_size, output_size=6, hidden=32, layers=1, dropout=0.0):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers 必须 >= 1，得到 {layers}")
        self.rnn = nn.ModuleList(
            [CfC(input_size if i == 0 else hidden, units=hidden, batch_first=True)
             for i in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, output_size)

    def forward(self, x):
        for rnn in self.rnn:
            out, _ = rnn(x)
            x = self.dropout(out)
        return self.readout(x)


class GaitLSTM(nn.Module):
    """nn.LSTM 堆叠 + 层间 dropout + 线性读出。

    与 GaitLTC 同构（hidden/层数/dropout 参数一致），公平对比。
    nn.LSTM 自带层间 dropout（dropout 参数），读出前再补一次。
    """

    def __init__(self, input_size, output_size=6, hidden=32, layers=1, dropout=0.0):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers 必须 >= 1，得到 {layers}")
        self.rnn = nn.LSTM(
            input_size,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.readout(self.dropout(out))


class TemporalBlock(nn.Module):
    """TCN 基本块：两层因果空洞卷积 + 残差连接（输入输出通道不同时 1x1 投影）。"""

    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation  # 因果：只向左填充
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.kernel = kernel
        self.dilation = dilation

    def forward(self, x):
        # (B, C, T)；卷积右侧填充后裁掉，保证因果与等长；残差先投影再算主路
        T = x.shape[2]
        res = self.res(x)
        y = self.conv1(x)[:, :, :T]
        x = self.drop(torch.relu(y))
        y = self.conv2(x)[:, :, :T]
        x = self.drop(torch.relu(y))
        return torch.relu(x + res)


class GaitTCN(nn.Module):
    """因果空洞卷积残差网络 + 线性读出。

    hidden 为通道数，layers 为残差块数，空洞率按块指数增长 1,2,4,...
    （layers=6、kernel=5 时感受野覆盖整窗 100 帧）。
    卷积前把 (B, T, F) 转成 (B, F, T)，输出转回。
    """

    def __init__(self, input_size, output_size=6, hidden=32, layers=1, dropout=0.0, kernel=5):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers 必须 >= 1，得到 {layers}")
        blocks = []
        in_ch = input_size
        for i in range(layers):
            blocks.append(
                TemporalBlock(in_ch, hidden, kernel, dilation=2 ** i, dropout=dropout)
            )
            in_ch = hidden
        self.net = nn.Sequential(*blocks)
        self.readout = nn.Linear(hidden, output_size)

    def forward(self, x):
        y = self.net(x.transpose(1, 2))  # (B, T, F) -> (B, F, T)
        return self.readout(y.transpose(1, 2))  # (B, F, T) -> (B, T, F)


class CausalMultiScaleProj(nn.Module):
    """多尺度因果卷积输入投影（借鉴 main.py CausalInputProj，issue #0011）。

    三尺度 depthwise 因果卷积（kernel/dilation 3/1、7/2、15/3，感受野约 3/13/43
    帧）+ pointwise 跨通道混合。与 main.py 的两处差异（借用映射见 issue）：
    1) 不做 delta 拼接——kinematic_dyn 特征已显式含一/二阶差分，重复拼接冗余；
    2) pointwise 投影到 hidden 而非 4H——ncps LTC 第 0 层按 hidden 输入逐帧消费，
      main.py 的 4H 技巧只服务于其自制门控 cell 的合并 matmul。
    全部填充在左侧（(k-1)*dilation, 0），严格因果。
    """

    _SCALES = ((3, 1), (7, 2), (15, 3))

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.dw = nn.ModuleList()
        self.pads = []
        for k, d in self._SCALES:
            self.dw.append(nn.Conv1d(input_size, input_size, k, dilation=d,
                                     groups=input_size, bias=False))
            self.pads.append(((k - 1) * d, 0))
        self.pointwise = nn.Conv1d(input_size * len(self._SCALES), hidden_size, 1)

    def forward(self, x):
        # (B, T, F) -> (B, F, T)；LTC 按帧吃 hidden 维输入，卷积一次性并行算完整序列
        x = x.transpose(1, 2)
        feats = []
        for conv, pad in zip(self.dw, self.pads):
            feats.append(conv(F.pad(x, pad)))
        out = self.pointwise(torch.cat(feats, dim=1))
        return out.transpose(1, 2)  # (B, T, hidden)


class RelPosSelfAttention(nn.Module):
    """多头自注意力 + 可学习相对位置偏置（借鉴 main.py SimpleSelfAttention，双向）。

    双向（非因果）：离线场景的上下文聚合，兼作 v2 C10 非因果上界（用户拍板，
    issue #0011）。相对位置偏置在 SDPA 前 cast 到 query dtype（main.py 问题 3
    的 AMP 隐患修复）。max_len 为偏置表容量，L <= max_len 时按窗口切片。
    """

    def __init__(self, hidden_size, num_heads=8, dropout=0.0, max_len=100):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size {hidden_size} 不能被 num_heads {num_heads} 整除")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout
        self.max_len = max_len
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))

    def forward(self, x):
        B, L, D = x.shape
        if L > self.max_len:
            raise ValueError(f"序列长 {L} 超过 rel-pos 偏置容量 max_len={self.max_len}")
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        pos = torch.arange(L, device=x.device)
        rel = pos.unsqueeze(1) - pos.unsqueeze(0) + (self.max_len - 1)  # (L, L) ∈ [0, 2max-2]
        bias = self.rel_pos_bias[:, rel].to(dtype=q.dtype)  # (heads, L, L)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class GaitLTCNCP(nn.Module):
    """NCP motor 直读出变体（issue #0014，Nature MI 2020 驾驶 / drone_causality 构型）。

    稠密 LTC 层（int units）堆叠后，末层换成 NCP wiring 的 LTC：sensory -> inter
    -> command -> motor 的稀疏拓扑，motor 神经元数 = output_size，**motor 状态即
    输出**（无 MLP 头）——「可审计」的 NCP 构型。循环矩阵尺寸 inter+command =
    ncp_units - output_size ≈ hidden，计算成本与稠密方案中性。
    wiring 拓扑随 config（ncp_units/sparsity/seed）保存，checkpoint 重建一致。
    """

    def __init__(self, input_size, output_size=6, hidden=32, layers=1, dropout=0.0,
                 ode_unfolds=6, ncp_units=None, ncp_sparsity=0.5, ncp_seed=22222):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers 必须 >= 1，得到 {layers}")
        ncp_units = ncp_units if ncp_units is not None else hidden + output_size
        if ncp_units - output_size < 3:
            raise ValueError(
                f"ncp_units {ncp_units} - output_size {output_size} < 3："
                f"NCP 需要至少 1 inter + 2 command 神经元"
            )
        self.rnn = nn.ModuleList(
            [LTC(input_size if i == 0 else hidden, units=hidden, batch_first=True,
                 ode_unfolds=ode_unfolds)
             for i in range(layers - 1)]
        )
        wiring = AutoNCP(ncp_units, output_size, sparsity_level=ncp_sparsity,
                         seed=ncp_seed)
        self.ncp = LTC(hidden if layers > 1 else input_size, wiring,
                       batch_first=True, ode_unfolds=ode_unfolds)
        self.dropout = nn.Dropout(dropout)
        self.output_size = output_size

    def forward(self, x):
        for rnn in self.rnn:
            out, _ = rnn(x)
            x = self.dropout(out)
        out, _ = self.ncp(x)
        return out  # motor 状态即输出 (B, T, output_size)


class GaitLTCAttn(nn.Module):
    """LTC + 多尺度因果卷积前端 + 双向自注意力 + ReZero 门控（借鉴 main.py，issue #0011）。

    数据流：(B,T,F) -> 卷积投影(hidden) -> 堆叠 LTC（seq 级 LayerNorm/残差，
    不碰 LTC 内部状态）-> 双向注意力分支（ReZero 逐通道 zero-init 门控：模型
    自行决定接纳多少全局上下文——EXP-010 示注意力无条件混入会平滑峰值、放大
    z7 域偏移）-> ReZero 跳连（conv 快路径，zero-init）-> 两层 MLP 头。
    dropout 沿用 cfg（单因子纪律，不采纳 main.py 的 0.5）。
    """

    def __init__(self, input_size, output_size=6, hidden=32, layers=1, dropout=0.0,
                 ode_unfolds=6, attn_heads=8, max_len=100, cell="ltc"):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers 必须 >= 1，得到 {layers}")
        if cell not in ("ltc", "mix"):
            raise ValueError(f"未知 cell {cell!r}，可选 ltc/mix")
        self.input_proj = CausalMultiScaleProj(input_size, hidden)
        if cell == "mix":
            self.rnn = nn.ModuleList(
                [MixedLTCCell(hidden, hidden, ode_unfolds=ode_unfolds)
                 for _ in range(layers)]
            )
        else:
            self.rnn = nn.ModuleList(
                [LTC(hidden, units=hidden, batch_first=True, ode_unfolds=ode_unfolds)
                 for _ in range(layers)]
            )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.self_attn = RelPosSelfAttention(hidden, num_heads=attn_heads,
                                             dropout=dropout, max_len=max_len)
        self.attn_norm = nn.LayerNorm(hidden)
        # ReZero 门控（零初始化）：attn_scale 先拿到梯度、打开分支后注意力参数
        # 才开始学习——主干先学，全局上下文按需接入
        self.attn_scale = nn.Parameter(torch.zeros(hidden))
        self.skip_scale = nn.Parameter(torch.zeros(hidden))  # ReZero：从 0 开始
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_size),
        )

    def forward(self, x):
        feat = self.input_proj(x)  # 快路径（局部多尺度特征），跳连复用
        h = feat
        for i, rnn in enumerate(self.rnn):
            out = rnn(h) if isinstance(rnn, MixedLTCCell) else rnn(h)[0]
            # 第 0 层输入无同维残差（卷积投影非 hidden 语义），深层做序列级残差
            h = self.norms[i](out if i == 0 else out + h)
            h = self.dropout(h)
        h = h + self.attn_scale * self.attn_norm(h + self.self_attn(h))
        h = h + self.skip_scale * feat
        return self.head(h)


MODELS = {"ltc": GaitLTC, "ltc_attn": GaitLTCAttn, "ltc_ncp": GaitLTCNCP,
          "cfc": GaitCfC, "lstm": GaitLSTM, "tcn": GaitTCN}


def make_model(name, input_size, output_size=6, hidden=32, layers=1, dropout=0.0, kernel=5,
               ode_unfolds=6, attn_heads=8, cell="ltc", ncp_units=None,
               ncp_sparsity=0.5, ncp_seed=22222):
    """按名字构造模型；kernel 仅 TCN 使用，ode_unfolds 仅 LTC 系使用，attn_heads
    仅 ltc_attn，cell 仅 ltc/ltc_attn（ltc=稠密 / mix=官方混合细胞），ncp_* 仅 ltc_ncp。"""
    if name not in MODELS:
        raise ValueError(f"未知模型 {name!r}，可选：{sorted(MODELS)}")
    if name == "tcn":
        return GaitTCN(input_size, output_size, hidden, layers, dropout, kernel=kernel)
    if name == "ltc":
        return GaitLTC(input_size, output_size, hidden, layers, dropout,
                       ode_unfolds=ode_unfolds, cell=cell)
    if name == "ltc_ncp":
        return GaitLTCNCP(input_size, output_size, hidden, layers, dropout,
                          ode_unfolds=ode_unfolds, ncp_units=ncp_units,
                          ncp_sparsity=ncp_sparsity, ncp_seed=ncp_seed)
    if name == "ltc_attn":
        return GaitLTCAttn(input_size, output_size, hidden, layers, dropout,
                           ode_unfolds=ode_unfolds, attn_heads=attn_heads, cell=cell)
    return MODELS[name](input_size, output_size, hidden, layers, dropout)
