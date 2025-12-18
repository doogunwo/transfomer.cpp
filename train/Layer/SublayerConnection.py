import torch # type: ignore
import torch.nn as nn # type: ignore

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__() # 1. super() 호출 방식 수정
        self.norm = nn.LayerNorm(size) # 2. dropout -> size로 변경 (중요!)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer): # 3. forwar -> forward 오타 수정
        # 논문 구조: LayerNorm(x + Dropout(Sublayer(x)))
        
        # (1) 서브레이어 통과
        sublayer_output = sublayer(x)
        
        # (2) 드롭아웃 적용
        sublayer_output = self.dropout(sublayer_output)
        
        # (3) 잔차 연결 (Add)
        add_output = x + sublayer_output
        
        # (4) 정규화 (Norm)
        return self.norm(add_output)