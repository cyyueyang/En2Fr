import config
from utils.data_loader import ModernDataloader

loader = ModernDataloader()

train_iter, valid_iter = loader.make_iter(batch_size=config.batch_size, device=config.device)

src_pad_idx = loader.src_vocab.stio["<pad>"]
tgt_pad_idx = loader.tgt_vocab.stio["<pad>"]
tgt_sos_idx = loader.tgt_vocab.stio["<sos>"]

enc_voc_size = len(loader.src_vocab)
dec_voc_size = len(loader.tgt_vocab)

