import pandas as pd
import numpy as np
import pickle
from os import path
import pathlib
import re
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize

from utils import apply_vocab, encode_questions
from config import (
    data_pickle_path,
    raw_data_file,
    tokenization_method,
    word_count_threshold,
    PAD_TOKEN,
    EOS_TOKEN,
    SOS_TOKEN,
    UNK_TOKEN,
    verbose,
    max_length,
)

tqdm.pandas()


class QuoraDatasetLoader:
    def __init__(self):
        if self._check_pickle_files():
            self._load_pickle_files()
        else:
            dataset = pd.read_csv(raw_data_file)
            dataset = dataset[dataset["is_duplicate"] == 1]

            train_set = dataset[:100002]
            test_set = dataset[100002:130002]
            val_set = dataset[130002:]

            for df in (test_set, val_set, train_set):
                df["processed_question1"] = df["question1"].progress_apply(
                    self._preprocess_question
                )
                df["processed_question2"] = df["question2"].progress_apply(
                    self._preprocess_question
                )

            self.vocab = self._build_vocab(
                train_set["processed_question1"]
                + train_set["processed_question2"]
            )
            self.itow = {
                i: w for i, w in enumerate(self.vocab)
            }  # a 1-indexed vocab translation table
            self.wtoi = {
                w: i for i, w in enumerate(self.vocab)
            }  # inverse table

            self._preprocess_tokens(test_set, val_set, train_set)
            self._save_to_pickle_files()

    def _preprocess_tokens(self, test_set, val_set, train_set):
        for df in (test_set, val_set, train_set):
            df["processed_tokens1"] = df["processed_question1"].progress_apply(
                apply_vocab, itow=self.itow
            )
            df["processed_tokens2"] = df["processed_question2"].progress_apply(
                apply_vocab, itow=self.itow
            )

        self.train_questions = [[], [], [], []]
        (self.train_questions[0], self.train_questions[1],) = encode_questions(
            self.wtoi, train_set["processed_tokens1"]
        )
        (self.train_questions[2], self.train_questions[3],) = encode_questions(
            self.wtoi, train_set["processed_tokens2"]
        )
        self.val_questions = [[], [], [], []]
        self.val_questions[0], self.val_questions[1] = encode_questions(
            self.wtoi, val_set["processed_tokens1"]
        )
        self.val_questions[2], self.val_questions[3] = encode_questions(
            self.wtoi, val_set["processed_tokens2"]
        )
        self.test_questions = [[], [], [], []]
        (self.test_questions[0], self.test_questions[1],) = encode_questions(
            self.wtoi, test_set["processed_tokens1"]
        )
        (self.test_questions[2], self.test_questions[3],) = encode_questions(
            self.wtoi, test_set["processed_tokens2"]
        )

    def _calc_word_counts(self, questions):
        counts = {}
        for q in questions:
            for w in q:
                counts[w] = counts.get(w, 0) + 1
        return (
            sorted(
                {(count, word) for word, count in counts.items()}, reverse=True
            ),
            counts,
        )

    def _build_vocab(self, questions):
        cw, counts = self._calc_word_counts(questions)
        # print some stats
        total_words = sum(counts.values())

        bad_words = [w for w, n in counts.items() if n <= word_count_threshold]
        vocab = []
        vocab.append(PAD_TOKEN)
        vocab.append(UNK_TOKEN)
        vocab.append(EOS_TOKEN)  # End of sentence
        vocab.append(SOS_TOKEN)  # Start of sentence
        vocab.extend(
            [w for w, n in counts.items() if n > word_count_threshold]
        )
        bad_count = sum(counts[w] for w in bad_words)
        if verbose:
            self._print_vocab_stats(
                bad_words, bad_count, counts, vocab, total_words, cw
            )

        # additional special UNK token we will use below to map infrequent words to
        return vocab

    def _print_vocab_stats(
        self, bad_words, bad_count, counts, vocab, total_words, cw
    ):
        print("top words and their counts:")
        print("\n".join(map(str, cw[:20])))
        print("total words:", total_words)
        print(
            "number of bad words (count less than threshold): %d/%d = %.2f%%"
            % (
                len(bad_words),
                len(counts),
                len(bad_words) * 100.0 / len(counts),
            )
        )
        print("number of words in vocab would be %d" % (len(vocab),))
        print(
            "number of UNKs: %d/%d = %.2f%%"
            % (bad_count, total_words, bad_count * 100.0 / total_words)
        )

    def _tokenize(self, sentence):
        return [
            i
            for i in re.split(
                r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence
            )
            if i not in {"", " ", "\n"}
        ]

    def _preprocess_question(self, text):
        if tokenization_method == "nltk":
            txt = word_tokenize(text)
        else:
            txt = self.tokenize(text)
        return txt

    def _check_pickle_files(self):
        return path.exists(data_pickle_path + "vocab")

    def _save_to_pickle_files(self):
        pathlib.Path(data_pickle_path).mkdir(parents=True, exist_ok=True)

        with open("data/vocab", "wb") as vocab_file:
            pickle.dump(self.vocab, vocab_file)
        with open("data/itow", "wb") as itow_file:
            pickle.dump(self.itow, itow_file)
        with open("data/wtoi", "wb") as wtoi_file:
            pickle.dump(self.wtoi, wtoi_file)
        with open("data/train_questions", "wb") as train_questions_file:
            pickle.dump(
                self.train_questions, train_questions_file,
            )
        with open("data/test_questions", "wb") as test_questions_file:
            pickle.dump(
                self.test_questions, test_questions_file,
            )
        with open("data/val_questions", "wb") as val_questions_file:
            pickle.dump(
                self.val_questions, val_questions_file,
            )

    def _load_pickle_files(self):
        with open(data_pickle_path + "vocab", "rb") as vocab_file:
            self.vocab = pickle.load(vocab_file)
        with open(data_pickle_path + "itow", "rb") as itow_file:
            self.itow = pickle.load(itow_file)
        with open(data_pickle_path + "wtoi", "rb") as wtoi_file:
            self.wtoi = pickle.load(wtoi_file)
        with open(
            data_pickle_path + "train_questions", "rb"
        ) as train_questions_file:
            self.train_questions = pickle.load(train_questions_file)
        with open(
            data_pickle_path + "val_questions", "rb"
        ) as val_questions_file:
            self.val_questions = pickle.load(val_questions_file)
        with open(
            data_pickle_path + "test_questions", "rb"
        ) as test_questions_file:
            self.test_questions = pickle.load(test_questions_file)
