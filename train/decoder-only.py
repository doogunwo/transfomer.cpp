import torch
import torch.nn as nn

from Layer.blocks import (
    InputEmbedding,
    PositionalEncoding,
    MultiHeadAttention,
    PositionwiseFeedForward,
    SublayerConnection,
    generate_square_subsequent_mask,
)

class DecoderOnlyLayer(nn.Module):
    """
    Decoder-only layer: [Causal Self-Attention] -> [Feed Forward]
    (Cross-attention 없음)
    """
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, causal_mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, causal_mask)[0])
        x = self.sublayer2(x, self.ffn)
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    GPT-like Decoder-only LM:
    input_ids -> Embedding+PE -> N layers -> LM head(vocab)
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
    ):
        super().__init__()
        self.embedding = InputEmbedding(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, causal_mask=None):
        """
        input_ids: (B, T)
        causal_mask: (T, T) 형태를 blocks.py의 MHA가 받는다고 가정 (네 테스트와 동일)
        """
        x = self.embedding(input_ids)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, causal_mask)

        return self.lm_head(x)  # (B, T, vocab)


if __name__ == "__main__":
    vocab = 5000
    model = DecoderOnlyTransformer(vocab_size=vocab, d_model=512, n_head=8)

    x = torch.randint(0, vocab, (2, 20))
    causal = generate_square_subsequent_mask(x.size(1))  # (T, T)

    out = model(x, causal_mask=causal)
    print(out.shape)  # 기대: (2, 20, 5000)
