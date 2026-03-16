import argparse
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

import config
import data
from data import *
from models.model.transformer import Transformer
from utils.bleu import idx_to_word, get_bleu
from utils.epoch_timer import epoch_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def init_weights(m):
    # 对norm 和 bias 不进行初始化
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_normal_(m.weight)

model = Transformer(
    src_pad_idx=data.src_pad_idx,
    tgt_pad_idx=data.tgt_pad_idx,
    tgt_sos_idx=data.tgt_sos_idx,
    enc_vocab_size=data.enc_voc_size,
    dec_vocab_size=data.dec_voc_size,
    d_model=config.d_model,
    n_head=config.n_heads,
    max_len=config.max_length,
    d_ff=config.d_ff,
    n_layer=config.n_layers,
    dropout=config.dropout,
)

model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, eps=config.adam_eps)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=config.factor, patience=config.patience)
criterion = nn.CrossEntropyLoss(ignore_index=data.src_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.to(config.device)
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.tgt

        optimizer.zero_grad()
        src = src.to(config.device)
        tgt = tgt.to(config.device)

        output = model(src, tgt[:, :-1])

        output_reshape = output.contiguous().view(-1, output.shape[-1])
        tgt_label_reshape = tgt[:, 1:].contiguous().view(-1)
        loss = criterion(output_reshape, tgt_label_reshape)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.tgt
            src = src.to(config.device)
            tgt = tgt.to(config.device)
            output = model(src, tgt[:, :-1])

            output_reshape = output.contiguous().view(-1, output.shape[-1])
            tgt_label_reshape = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, tgt_label_reshape)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(config.batch_size):
                try:
                    trg_words = idx_to_word(batch.tgt[j], loader.tgt_vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.tgt_vocab)
                    bleu = get_bleu(trg_words.split(), output_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)
    batch_bleu = sum(batch_bleu) / len(batch_bleu)

    return epoch_loss / len(iterator), batch_bleu

def run(total_epoch, best_loss):
    train_losses, val_losses, bleus = [], [], []

    for epoch in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, config.clip)
        val_loss, val_bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if epoch > config.warmup:
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        bleus.append(val_bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(val_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(val_losses))
        f.close()

        print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {val_loss:.3f} |  Val PPL: {math.exp(val_loss):7.3f}')
        print(f'\tBLEU Score: {val_bleu:.3f}')

if __name__ == '__main__':
    run(config.epoch, config.inf)





