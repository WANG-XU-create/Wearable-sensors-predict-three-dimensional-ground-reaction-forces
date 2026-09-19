"""序列模型：LTC 主模型 + LSTM/TCN 基线（ticket #5）+ CfC 提速变体（v2 §3.3 / C12）
+ LTCAttn 混合架构（借鉴 main.py 原型，issue #0011）。

统一 seq2seq 回归接口：输入 (B, T, input_size) -> 输出 (B, T, output_size)，
共享训练/评估管线，仅 --model 切换。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC, CfC


class GaitLTC(nn.Module):
    """ncps.torch.LTC 堆叠 + dropout + 线性读出。

    每个 LTC 层用 int units（全隐态输出，可堆叠），层间与读出前加 dropout。
    tracer（#3）用最小配置（hidden 32、1 层）；#4 在此之上扩到 hidden 128、2 层。
    ode_unfolds 为半隐式 ODE 解算器的内层展开步数（库默认 6）；C12 提速
    （v2 §3.3）：降低该值减少每时间步的迭代开销，精度需 sweep 验证。
    该参数只影响前向计算图展开次数，不引入参数，checkpoint 兼容。
    """

    def __init__(self, input_size, output_size=6, hidden=32, layers=1, dropout=0.0,
                 ode_unfolds=6):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers 必须 >= 1，得到 {layers}")
        self.rnn = nn.ModuleList(
            [LTC(input_size if i == 0 else hidden, units=hidden, batch_first=True,
                 ode_unfolds=ode_unfolds)
             for i in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, output_size)

    def forward(self, x):
        for rnn in self.rnn:
            out, _ = rnn(x)
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


class GaitLTCAttn(nn.Module):
    """LTC + 多尺度因果卷积前端 + 双向自注意力 + ReZero 门控（借鉴 main.py，issue #0011）。

    数据流：(B,T,F) -> 卷积投影(hidden) -> 堆叠 LTC（seq 级 LayerNorm/残差，
    不碰 LTC 内部状态）-> 双向注意力分支（ReZero 逐通道 zero-init 门控：模型
    自行决定接纳多少全局上下文——EXP-010 示注意力无条件混入会平滑峰值、放大
    z7 域偏移）-> ReZero 跳连（conv 快路径，zero-init）-> 两层 MLP 头。
    dropout 沿用 cfg（单因子纪律，不采纳 main.py 的 0.5）。
    """

    def __init__(self, input_size, output_size=6, hidden=32, layers=1, dropout=0.0,
                 ode_unfolds=6, attn_heads=8, max_len=100):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers 必须 >= 1，得到 {layers}")
        self.input_proj = CausalMultiScaleProj(input_size, hidden)
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
            out, _ = rnn(h)
            # 第 0 层输入无同维残差（卷积投影非 hidden 语义），深层做序列级残差
            h = self.norms[i](out if i == 0 else out + h)
            h = self.dropout(h)
        h = h + self.attn_scale * self.attn_norm(h + self.self_attn(h))
        h = h + self.skip_scale * feat
        return self.head(h)


MODELS = {"ltc": GaitLTC, "ltc_attn": GaitLTCAttn, "cfc": GaitCfC, "lstm": GaitLSTM,
          "tcn": GaitTCN}


def make_model(name, input_size, output_size=6, hidden=32, layers=1, dropout=0.0, kernel=5,
               ode_unfolds=6, attn_heads=8):
    """按名字构造模型；kernel 仅 TCN 使用，ode_unfolds 仅 LTC 系使用，attn_heads 仅 ltc_attn。"""
    if name not in MODELS:
        raise ValueError(f"未知模型 {name!r}，可选：{sorted(MODELS)}")
    if name == "tcn":
        return GaitTCN(input_size, output_size, hidden, layers, dropout, kernel=kernel)
    if name == "ltc":
        return GaitLTC(input_size, output_size, hidden, layers, dropout,
                       ode_unfolds=ode_unfolds)
    if name == "ltc_attn":
        return GaitLTCAttn(input_size, output_size, hidden, layers, dropout,
                           ode_unfolds=ode_unfolds, attn_heads=attn_heads)
    return MODELS[name](input_size, output_size, hidden, layers, dropout)
