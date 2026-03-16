import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class ModernDataloader:
    def __init__(self, src_lang="de", tgt_lang="en"):
        self.src_tokenizer = get_tokenizer("spacy", language=f"{src_lang}_core_news_sm")
        self.tgt_tokenizer = get_tokenizer("spacy", language=f"{tgt_lang}_core_web_sm")

        self.UNK, self.PAD = 0, 1
        self.SOS, self.EOS = 2, 3

        self.specials = ["<unk>", "<pad>", "<sos>", "<eos>"]

    def yield_tokens(self, data_iter, tokenizer):
        for src, tgt in data_iter:
            yield tokenizer(src)

    def build_vocab(self, min_freq=2):
        train_iter = Multi30k(split="train", language_pair=("de", "en"))
        self.src_vocab = build_vocab_from_iterator(
            self.yield_tokens(train_iter, self.src_tokenizer),
            min_freq=min_freq,
            specials=self.specials,
            special_first=True
        )
        train_iter = Multi30k(split="train", language_pair=("en", "de"))
        self.tgt_vocab = build_vocab_from_iterator(
            self.yield_tokens(train_iter, self.tgt_tokenizer),
            min_freq=min_freq,
            specials=self.specials,
            special_first=True
        )
        self.src_vocab.set_default_index(self.UNK)
        self.tgt_vocab.set_default_index(self.UNK)

    def text_transform(self, text, tokenizer, vocab):
        tokens = tokenizer(text)
        return [self.SOS] + [vocab[token] for token in tokens] + [self.EOS]

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []

        for src_text, tgt_text in batch:
            src_encoded = self.text_transform(src_text, self.src_tokenizer, self.src_vocab)
            tgt_encoded = self.text_transform(tgt_text, self.tgt_tokenizer, self.tgt_vocab)

            src_batch.append(src_encoded)
            tgt_batch.append(tgt_encoded)

        src_batch = pad_sequence(src_batch, padding_value=self.PAD, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD, batch_first=True)

        return src_batch, tgt_batch

    def make_iter(self, batch_size=128, device="cuda"):
        train_iter = Multi30k(split="train", language_pair=("de", "en"))
        val_iter = Multi30k(split="val", language_pair=("de", "en"))
        test_iter = Multi30k(split="test", language_pair=("de", "en"))

        self.build_vocab(min_freq=2)

        train_loader = DataLoader(
            list(train_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        val_loader = DataLoader(
            list(val_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        test_loader = DataLoader(
            list(test_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    loader = ModernDataloader()
    train_loader, val_loader, test_loader = loader.make_iter(batch_size=128)
    for src, tgt in train_loader:
        print(src.shape, tgt.shape)  # [batch, seq_len]
        break
