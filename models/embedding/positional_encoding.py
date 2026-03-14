import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        ) # [d_model / 2]

        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model / 2]
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :]

if __name__ == '__main__':
    pos_encoding = PositionalEncoding(d_model=512)
    x = torch.randn(10, 128)
    print(pos_encoding(x).size())

