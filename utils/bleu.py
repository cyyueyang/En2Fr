from collections import Counter
import math
import numpy as np

def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))

    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i: i + n]) for i in range(len(hypothesis) - n + 1)]
        )
        r_ngrams = Counter(
            [tuple(reference[i: i + n]) for i in range(len(reference) - n + 1)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))  # clipped count
        stats.append(max([len(hypothesis) - n + 1, 0]))

    return stats

def bleu(stats):
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0

    c, r = stats[: 2]

    log_bleu_prec = sum(
        [math.log(float(x) / y) for (x, y) in zip(stats[2::2], stats[3::2])]
    )

    bp = min([0, 1 - float(r) / c])

    return math.exp(bp + log_bleu_prec)

def idx_to_word(x, vocab):
    words = []

    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)

    words = " ".join(words)
    return words

