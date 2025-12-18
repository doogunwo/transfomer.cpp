# Linear - ReLu -> Dropout - Linear

import torch # type: ignore
import torch.nn as nn # type: ignore

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):

        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.ReLU = nn.ReLU()

    def forward(self, x):

        x = self.w_1(x)
        x = self.ReLU(x)
        x = self.dropout(x)

        x = self.w_2(x)

        return x
