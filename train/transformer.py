import torch
import torch.nn as nn

# Layer 폴더의 blocks.py에서 모든 부품을 가져옵니다.
# (blocks.py에 클래스들이 잘 정의되어 있어야 합니다)
from Layer.blocks import (
    InputEmbedding,
    PositionalEncoding,
    MultiHeadAttention,
    PositionwiseFeedForward,
    SublayerConnection,
    generate_square_subsequent_mask # blocks.py에 이 함수가 없으면 아래 테스트 코드에 직접 넣어도 됨
)

class EncoderLayer(nn.Module):
    """
    인코더 레이어: [Self-Attention] -> [Feed Forward]
    """
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Add & Norm을 위한 서브레이어 연결 2개
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask):
        # 1. Self-Attention (Query=x, Key=x, Value=x)
        # lambda를 사용하여 sublayer 안에서 함수가 실행되도록 함
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask)[0])
        
        # 2. Feed Forward
        x = self.sublayer2(x, self.ffn)
        return x

class DecoderLayer(nn.Module):
    """
    디코더 레이어: [Masked Self-Attention] -> [Cross Attention] -> [Feed Forward]
    """
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.src_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask, trg_mask):
        # 1. Masked Self-Attention (Target끼리만 봄, 미래는 못 봄)
        m_attn = lambda x: self.self_attn(x, x, x, trg_mask)[0]
        x = self.sublayer1(x, m_attn)
        
        # 2. Cross Attention (Query=Decoder, Key=Encoder, Value=Encoder)
        # memory는 인코더의 최종 출력값
        c_attn = lambda x: self.src_attn(x, memory, memory, src_mask)[0]
        x = self.sublayer2(x, c_attn)
        
        # 3. Feed Forward
        x = self.sublayer3(x, self.ffn)
        return x

class Transformer(nn.Module):
    """
    트랜스포머 전체 모델 조립
    """
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=512, n_head=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        
        # 1. 임베딩 및 위치 인코딩 초기화
        self.src_embedding = InputEmbedding(d_model, src_vocab_size)
        self.src_pe = PositionalEncoding(d_model, max_len, dropout)
        
        self.trg_embedding = InputEmbedding(d_model, trg_vocab_size)
        self.trg_pe = PositionalEncoding(d_model, max_len, dropout)
        
        # 2. 인코더 레이어 쌓기 (ModuleList 사용)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_encoder_layers)
        ])
        
        # 3. 디코더 레이어 쌓기
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_decoder_layers)
        ])
        
        # 4. 최종 출력 선형층 (Output Probabilities)
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.d_model = d_model

    def encode(self, src, src_mask):
        # Source -> Embedding -> PE
        x = self.src_embedding(src)
        x = self.src_pe(x)
        
        # N개의 인코더 레이어 통과
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, trg, memory, src_mask, trg_mask):
        # Target -> Embedding -> PE
        x = self.trg_embedding(trg)
        x = self.trg_pe(x)
        
        # N개의 디코더 레이어 통과
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, trg_mask)
        return x

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        """
        전체 순전파 과정
        src: 원문 문장 (batch, seq_len)
        trg: 타겟 문장 (batch, seq_len)
        """
        # 1. 인코딩 (Memory 생성)
        memory = self.encode(src, src_mask)
        
        # 2. 디코딩 (Memory 활용)
        output = self.decode(trg, memory, src_mask, trg_mask)
        
        # 3. 최종 단어 예측
        return self.fc_out(output)

# --- 실행 테스트 (Run Test) ---
if __name__ == '__main__':
    # 1. 하이퍼파라미터 설정
    src_vocab = 5000
    trg_vocab = 5000
    d_model = 512
    n_head = 8
    
    # 2. 모델 생성
    print("Transformer 모델 생성 중...")
    model = Transformer(src_vocab, trg_vocab, d_model, n_head)
    
    # 3. 더미 데이터 생성
    src = torch.randint(0, src_vocab, (2, 10))  # 배치2, 길이10
    trg = torch.randint(0, trg_vocab, (2, 20))  # 배치2, 길이20
    
    # 4. 마스크 생성 (타겟 문장용 Look-ahead mask)
    # blocks.py에 generate_square_subsequent_mask가 없다면 여기서 에러가 날 수 있음
    # 그럴 경우, 이 파일 상단에 함수를 정의하거나 blocks.py에 추가해야 함
    trg_mask = generate_square_subsequent_mask(trg.size(1))
    
    # 5. 모델 실행
    output = model(src, trg, src_mask=None, trg_mask=trg_mask)
    
    print("-" * 30)
    print(f"입력 Source 형태: {src.shape}")
    print(f"입력 Target 형태: {trg.shape}")
    print(f"최종 Output 형태: {output.shape}") # 예상: [2, 20, 5000]
    print("-" * 30)
    print("✅ 트랜스포머 모델 조립 완료!")