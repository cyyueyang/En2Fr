import torch.nn as nn
import torch
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        x_normed = (x - mean) / (torch.sqrt(var + self.eps))

        return self.gamma * x_normed + self.beta

if __name__ == "__main__":
    x = torch.randn(4, 128, 512)
    lm = LayerNorm(512)
    print(lm(x).size())