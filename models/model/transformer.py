import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 tgt_pad_idx,
                 tgt_sos_idx,
                 enc_vocab_size,
                 dec_vocab_size,
                 d_model,
                 n_head,
                 max_len,
                 d_ff,
                 n_layer,
                 dropout):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.encoder = Encoder(enc_vocab_size, max_len, d_model, d_ff, n_head, n_layer, dropout)
        self.decoder = Decoder(dec_vocab_size, max_len, d_model, d_ff, n_head, n_layer, dropout)

    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        src_tgt_mask = self.make_pad_mask(tgt, src, self.tgt_pad_idx, self.src_pad_idx)
        tgt_mask = self.make_no_peak_mask(tgt, tgt) * self.make_pad_mask(tgt, tgt, self.tgt_pad_idx, self.tgt_pad_idx)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, src_tgt_mask, tgt_mask)
        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        q_len, k_len = q.size(1), k.size(1)

        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2) # [bs, 1, 1, k_len]
        k = k.repeat(1, 1, q_len, 1) # [bs, 1, q_len, k_len]

        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)  # [bs, 1, q_len, 1]
        q = q.repeat(1, 1, 1, k_len)  # [bs, 1, q_len, k_len]

        mask = q & k

        return mask

    def make_no_peak_mask(self, q, k):
        q_len, k_len = q.size(1), k.size(1)

        mask = torch.tril(torch.ones(q_len, k_len, dtype=torch.bool))

        return mask
