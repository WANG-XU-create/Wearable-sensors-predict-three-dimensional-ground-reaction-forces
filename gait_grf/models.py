"""LTC 序列模型：ncps.torch.LTC 堆叠 + dropout + 线性读出。"""

import torch
import torch.nn as nn
from ncps.torch import LTC


class GaitLTC(nn.Module):
    """sequence-to-sequence 回归：输入 (B, T, input_size) -> 输出 (B, T, output_size)。

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
