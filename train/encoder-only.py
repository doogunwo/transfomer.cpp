import torch
import torch.nn as nn

from Layer.blocks import (
    InputEmbedding,
    PositionalEncoding,
    MultiHeadAttention,
    PositionwiseFeedForward,
    SublayerConnection,
)

class EncoderLayer(nn.Module):
    """
    Encoder layer: [Self-Attention] -> [Feed Forward]
    """
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask)[0])
        x = self.sublayer2(x, self.ffn)
        return x


class EncoderOnlyTransformer(nn.Module):
    """
    BERT-like Encoder-only backbone:
    input_ids -> Embedding+PE -> N layers -> (optional) head
    여기서는 예시로 token-level vocab logits(head) 붙여둠.
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_head=8,
        num_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
        with_lm_head=True,
    ):
        super().__init__()
        self.embedding = InputEmbedding(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])

        self.with_lm_head = with_lm_head
        self.lm_head = nn.Linear(d_model, vocab_size) if with_lm_head else None

    def forward(self, input_ids, attn_mask=None):
        """
        input_ids: (B, S)
        attn_mask: 패딩 마스크 등을 blocks.py의 MHA 스펙에 맞춰 전달
        """
        x = self.embedding(input_ids)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, attn_mask)

        if self.with_lm_head:
            return self.lm_head(x)   # (B, S, vocab)  (MLM류 실험용)
        return x                    # (B, S, d_model) (백본 출력)


if __name__ == "__main__":
    vocab = 5000
    model = EncoderOnlyTransformer(vocab_size=vocab, d_model=512, n_head=8)

    x = torch.randint(0, vocab, (2, 10))
    out = model(x, attn_mask=None)
    print(out.shape)  # 기대: (2, 10, 5000)
