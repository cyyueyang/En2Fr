import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbeddings

class Encoder(nn.Module):
    def __init__(self, env_vocab_size, max_seq_len, d_model, d_ff, n_head, n_layer, dropout):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbeddings(
            env_vocab_size,
            d_model,
            max_seq_len,
            dropout
        )

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)]
        )

    def forward(self, x, src_mask=None):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)

        return x

if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.long)
    encoder = Encoder(256, 128, 512, 2048, 8, 6, 0.1)
    y = encoder(x)
    print(y.size())