import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbeddings

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout):
        super(TransformerEmbeddings, self).__init__()
        self.token_embeddings = TokenEmbeddings(vocab_size, d_model)
        self.position_embeddings = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.token_embeddings(x) + self.position_embeddings(x)
        x = self.dropout(x)
        return x

if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)

    emb = TransformerEmbeddings(10, 128, 256, 0.1)
    print(emb(x).size())

