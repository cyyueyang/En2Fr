import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(self.d_model, self.num_heads * self.head_dim)
        self.w_k = nn.Linear(self.d_model, self.num_heads * self.head_dim)
        self.w_v = nn.Linear(self.d_model, self.num_heads * self.head_dim)

        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.d_model)

    def forward(self, q, k, v, mask=None):
        bs, seq_len, dim = q.size()
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        scores = F.softmax(attn, dim=-1)

        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        out = self.w_o(out)

        return out

if __name__ == '__main__':
    m = MultiHeadAttention(512, 8)
    x = torch.randn(4, 128, 512)
    y = m(x, x, x)
    print(y.size())