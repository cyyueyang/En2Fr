import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.norm import LayerNorm
from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.dropout = nn.Dropout(dropout)

        self.attention = MultiHeadAttention(self.d_model, self.n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(self.d_model, self.d_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        _x = x
        _x = self.attention(_x, _x, _x, mask=src_mask)
        _x = self.dropout1(_x)
        x = self.norm1(_x + x)

        _x = x
        _x = self.ffn(_x)
        _x = self.dropout2(_x)
        x = self.norm2(_x + x)

        return x

if __name__ == '__main__':
    layer = EncoderLayer(512, 64, 2048, 0.1)
    x = torch.randn(4, 128, 512)
    y = layer(x)
    print(y.size())

