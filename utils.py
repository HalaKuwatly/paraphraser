import numpy as np
from config import EOS_TOKEN, SOS_TOKEN, UNK_TOKEN, max_length


def add_sentence_seperators(sent):
    return [SOS_TOKEN] + sent + [EOS_TOKEN]


def apply_vocab(words, wtoi):
    return [
        w if w in wtoi else UNK_TOKEN for w in add_sentence_seperators(words)
    ]


def encode_questions(wtoi, questions):
    n = len(questions)
    questions_arrays = np.zeros((n, max_length), dtype="uint32")
    questions_length = np.zeros(n, dtype="uint32")
    for i, q in enumerate(questions):
        questions_length[i] = min(max_length, len(q))
        for wi, w in enumerate(q):
            if wi < max_length:
                questions_arrays[i, wi] = wtoi[w]
    return questions_arrays, questions_length
