# 임베딩 행렬
# 입력: 단어 번호
# 출력: d(model) 차원 벡터 
# 거대한 행렬(Look-up Table) 참조 
# 모델은 V X d model 크기의 가중치 행렬 W(e)를 가짐
# V:: 단어 집합의 크기(보캡 사이즈)
# dmodel: 벡터 차원(논문기준 512)

# 1. 클래스 정의 nn.Module 상속
# 2. nn.Embedding(vocab_size, d_model) 사용
# 3. Forward 
#   입력 x 정수 인덱스 -> 임베딩 통과
#   결과값에 math.sqrt(d_model) 곱하기
import torch # type: ignore
import torch.nn as nn # type: ignore
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 2. 임베딩 변환 후 스케일링 적용
        # embbedding(x) * sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    
