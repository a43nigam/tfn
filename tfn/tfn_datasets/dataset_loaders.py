import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import TensorDataset

# -----------------------------------------------------------------------------
# Tokenisation utilities
# -----------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    import re
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(texts: List[str], vocab_size: int = 10000) -> Dict[str, int]:
    """Return word→idx mapping with <PAD>=0, <UNK>=1."""
    from collections import Counter

    counter = Counter()
    for t in texts:
        counter.update(_tokenise(t))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def texts_to_tensor(
    texts: List[str],
    word2idx: Dict[str, int],
    seq_len: int = 128,
    shuffle: bool = False,
) -> torch.Tensor:
    ids = []
    for t in texts:
        tokens = _tokenise(t)
        if shuffle:
            random.shuffle(tokens)
        seq = [word2idx.get(tok, 1) for tok in tokens][:seq_len]
        seq += [0] * (seq_len - len(seq))
        ids.append(seq)
    return torch.tensor(ids, dtype=torch.long)

# -----------------------------------------------------------------------------
# Dataset-specific loaders
# -----------------------------------------------------------------------------

from .text_classification import load_agnews as _modern_load_agnews


def load_agnews(*args, **kwargs):
    """Wrapper for AG News loader that uses the robust implementation in text_classification.py."""
    return _modern_load_agnews(*args, **kwargs)


def _load_hf(dataset_name: str, split: str):
    from datasets import load_dataset  # type: ignore
    return load_dataset(dataset_name, split=split)


def load_yelp_full(
    seq_len: int = 128,
    vocab_size: int = 10000,
    val_ratio: float = 0.1,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """Yelp Review Full (5-class) classification."""
    import pathlib, pandas as pd

    kaggle_path = Path(
        "/kaggle/input/yelp-dataset-yelp-review-full/yelp_review_full_csv/train.csv"
    )
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path, header=None)
        texts = df[1].tolist()
        labels = (df[0] - 1).tolist()
    else:
        data = _load_hf("yelp_review_full", "train")
        texts = [ex["text"] for ex in data]
        labels = [ex["label"] for ex in data]

    val_size = int(len(texts) * val_ratio)
    random.seed(42)
    idx = list(range(len(texts)))
    random.shuffle(idx)
    val_idx = set(idx[:val_size])

    train_texts, val_texts, train_labels, val_labels = [], [], [], []
    for i, (t, l) in enumerate(zip(texts, labels)):
        if i in val_idx:
            val_texts.append(t)
            val_labels.append(l)
        else:
            train_texts.append(t)
            train_labels.append(l)

    word2idx = build_vocab(train_texts, vocab_size)

    train_ids = texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train)
    val_ids = texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval)

    return (
        TensorDataset(train_ids, torch.tensor(train_labels)),
        TensorDataset(val_ids, torch.tensor(val_labels)),
        len(word2idx),
    )


def load_imdb(
    seq_len: int = 256,
    vocab_size: int = 20000,
    val_ratio: float = 0.1,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """IMDB movie review sentiment (binary)."""
    import pathlib, pandas as pd

    kaggle_path = Path(
        "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    )
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path)
        texts = df["review"].tolist()
        labels = [0 if s == "negative" else 1 for s in df["sentiment"]]
    else:
        train_data = _load_hf("imdb", "train")
        texts = [ex["text"] for ex in train_data]
        labels = [ex["label"] for ex in train_data]

    val_size = int(len(texts) * val_ratio)
    random.seed(42)
    idx = list(range(len(texts)))
    random.shuffle(idx)
    val_idx = set(idx[:val_size])

    train_texts, val_texts, train_labels, val_labels = [], [], [], []
    for i, (t, l) in enumerate(zip(texts, labels)):
        if i in val_idx:
            val_texts.append(t)
            val_labels.append(l)
        else:
            train_texts.append(t)
            train_labels.append(l)

    word2idx = build_vocab(train_texts, vocab_size)

    train_ids = texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train)
    val_ids = texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval)

    return (
        TensorDataset(train_ids, torch.tensor(train_labels)),
        TensorDataset(val_ids, torch.tensor(val_labels)),
        len(word2idx),
    ) 