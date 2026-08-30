"""序列模型：LTC 主模型 + LSTM/TCN 基线（ticket #5）。

三者统一 seq2seq 回归接口：输入 (B, T, input_size) -> 输出 (B, T, output_size)，
共享训练/评估管线，仅 --model 切换。
"""

import torch
import torch.nn as nn
from ncps.torch import LTC


class GaitLTC(nn.Module):
    """ncps.torch.LTC 堆叠 + dropout + 线性读出。

    每个 LTC 层用 int units（全隐态输出，可堆叠），层间与读出前加 dropout。
    tracer（#3）用最小配置（hidden 32、1 层）；#4 在此之上扩到 hidden 128、2 层。
    """

    def __init__(self, input_size, output_size=6, hidden=32, layers=1, dropout=0.0):
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers 必须 >= 1，得到 {layers}")
        self.rnn = nn.ModuleList(
            [LTC(input_size if i == 0 else hidden, units=hidden, batch_first=True)
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


MODELS = {"ltc": GaitLTC, "lstm": GaitLSTM, "tcn": GaitTCN}


def make_model(name, input_size, output_size=6, hidden=32, layers=1, dropout=0.0, kernel=5):
    """按名字构造模型；kernel 仅 TCN 使用（卷积核宽，决定感受野）。"""
    if name not in MODELS:
        raise ValueError(f"未知模型 {name!r}，可选：{sorted(MODELS)}")
    if name == "tcn":
        return GaitTCN(input_size, output_size, hidden, layers, dropout, kernel=kernel)
    return MODELS[name](input_size, output_size, hidden, layers, dropout)
