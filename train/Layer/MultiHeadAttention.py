# 쿼리 Q
# 키 K
# 밸류 V

# 입력 벡터 X에 서로 다른 3개 가중치 행렬 W(Q) W(K) W(V) 를 곱해서 3가지 버전만듬

# 어텐션 = (Q, K, V) = Softmax(QK(T)/ d(k)) V
# 쿼리와 키를 내적(행렬곱) -> 두 벡터가 비슷할수록 큰 값이 나옴
# / d(k) 스케일링 차원이 너무 커지면 기울기 사라짐, 그래서 나눔
# x V 확률만큼 밸류 정보를 가져와서 섞는다.

# 멀티헤드 -> 쪼개기
# 여러 개로 쪼개서 병렬로 수행함
# 쪼개진 헤드를 Conca, Wo 행렬으 통과시켜 정리함

import torch # type: ignore
import torch.nn as nn # type: ignore
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0, "d_model divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q,k,v, mask=None):
        batch_size = q.size(0)

        Q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # (3-2) 최종 선형 변환
        output = self.w_o(context)
        return output, attn_weights
    
def generate_square_subsequent_mask(sz):
        mask = torch.ones(sz, sz)
        mask = torch.tril(mask)
        return mask.unsqueeze(0).unsqueeze(0)
    