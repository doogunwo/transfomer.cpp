import torch # type: ignore
import torch.nn as nn # type: ignore
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        # 크기 0 행렬 생성

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 위치 인덱스

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 분모 계산, log 공간

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 배치 자원 추가
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x: Input Embedding을 통과한 데이터 (batch_size, seq_len, d_model)
        """
        # 입력된 문장 길이 seq_len 만큼 잘라서 더하기
        x = x+ self.pe[:, :x.size(1), :]

        return self.dropout(x)

