import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbeddings

class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_seq_len, d_model, d_ff, n_head, n_layer, dropout):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbeddings(dec_vocab_size, d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)]
        )

        self.lm_head = nn.Linear(d_model, dec_vocab_size)

    def forward(self, tgt, enc_src, src_mask, tgt_mask):
        tgt = self.emb(tgt)

        for layer in self.layers:
            tgt = layer(tgt, enc_src, src_mask, tgt_mask)

        output = self.lm_head(tgt)

        return output
