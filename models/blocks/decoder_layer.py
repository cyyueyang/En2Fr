import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.norm import LayerNorm
from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_hidden, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, d_hidden, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, dec, enc, src_mask, tgt_mask):
        _x = dec
        _x = self.self_attention(_x, _x, _x, tgt_mask)
        _x = self.dropout1(_x)
        x = self.norm1(_x + dec)

        _x = x
        _x = self.cross_attention(x, enc, enc, src_mask)
        _x = self.dropout2(_x)
        x = self.norm2(_x + x)

        _x = x
        _x = self.ffn(_x)
        _x = self.dropout3(_x)
        x = self.norm3(_x + x)

        return x

if __name__ == "__main__":
    layer = DecoderLayer(512, n_head=8, d_hidden=128, dropout=0.1)
    enc = torch.randn(4, 128, 512)
    dec = torch.randn(4, 128, 512)

    y = layer(enc, dec, src_mask=None, tgt_mask=None)
    print(y.size())

