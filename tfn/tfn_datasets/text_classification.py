from __future__ import annotations

"""tfn.datasets.text_classification
Light-weight text-classification dataset loaders that do **not** depend on any
project-specific utilities outside the `tfn` package. If HuggingFace `datasets`
can be imported they are used, otherwise each loader falls back to a tiny direct
CSV download (AG-News) or raises an informative error.
"""

from typing import List, Tuple, Dict
import random
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

# Optional dependency ---------------------------------------------------------
try:
    from datasets import load_dataset  # type: ignore
    _HAVE_HF = True
except ImportError:
    _HAVE_HF = False

# ---------------------------------------------------------------------------
# Tokenisation helpers (simple whitespace + punctuation split)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    import re
    return re.findall(r"\b\w+\b", text.lower())


def _build_vocab(texts: List[str], vocab_size: int = 10000) -> Dict[str, int]:
    from collections import Counter

    counter = Counter()
    for t in texts:
        counter.update(_tokenise(t))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def _texts_to_tensor(
    texts: List[str],
    word2idx: Dict[str, int],
    seq_len: int = 128,
    shuffle: bool = False,
) -> torch.Tensor:
    ids: List[List[int]] = []
    for t in texts:
        tokens = _tokenise(t)
        if shuffle:
            random.shuffle(tokens)
        seq = [word2idx.get(tok, 1) for tok in tokens][:seq_len]
        seq += [0] * (seq_len - len(seq))
        ids.append(seq)
    return torch.tensor(ids, dtype=torch.long)

# ---------------------------------------------------------------------------
# CSV fallback for AG-News (tiny helper)
# ---------------------------------------------------------------------------

def _download_agnews_csv() -> Tuple[List[str], List[int]]:
    import urllib.request, csv, io

    base = (
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/"
        "ag_news_csv/train.csv"
    )
    data = urllib.request.urlopen(base).read().decode("utf-8")
    reader = csv.reader(io.StringIO(data))
    texts, labels = [], []
    for row in reader:
        if len(row) < 3:
            continue
        labels.append(int(row[0]) - 1)
        texts.append(f"{row[1]} {row[2]}")
    return texts, labels

# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_agnews(
    seq_len: int = 128,
    vocab_size: int = 10000,
    val_split: int = 10000,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """AG-News 4-class classification (120 k train, 7.6 k test).

    Returns `(train_ds, val_ds, vocab_size)`.
    """
    if _HAVE_HF:
        train_hf = load_dataset("ag_news", split="train")
        texts = [ex["text"] for ex in train_hf]
        labels = [ex["label"] for ex in train_hf]
    else:
        texts, labels = _download_agnews_csv()

    # Deterministic shuffle + split
    rng = random.Random(42)
    idx = list(range(len(texts)))
    rng.shuffle(idx)

    val_idx = set(idx[:val_split])
    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    for i in idx:
        if i in val_idx:
            val_texts.append(texts[i])
            val_labels.append(labels[i])
        else:
            train_texts.append(texts[i])
            train_labels.append(labels[i])

    word2idx = _build_vocab(train_texts, vocab_size)
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval)

    return (
        TensorDataset(train_ids, torch.tensor(train_labels)),
        TensorDataset(val_ids, torch.tensor(val_labels)),
        len(word2idx),
    )


def _load_yelp_full_raw() -> Tuple[List[str], List[int]]:
    if not _HAVE_HF:
        raise RuntimeError("Yelp Review Full loader requires `datasets` library.")
    data = load_dataset("yelp_review_full", split="train")
    texts = [ex["text"] for ex in data]
    labels = [ex["label"] for ex in data]
    return texts, labels


def load_yelp_full(
    seq_len: int = 256,
    vocab_size: int = 20000,
    val_ratio: float = 0.1,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """Yelp Review Full 5-class classification."""
    texts, labels = _load_yelp_full_raw()

    rng = random.Random(42)
    idx = list(range(len(texts)))
    rng.shuffle(idx)

    val_size = int(len(texts) * val_ratio)
    val_idx = set(idx[:val_size])

    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    for i in idx:
        if i in val_idx:
            val_texts.append(texts[i])
            val_labels.append(labels[i])
        else:
            train_texts.append(texts[i])
            train_labels.append(labels[i])

    word2idx = _build_vocab(train_texts, vocab_size)
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval)

    return (
        TensorDataset(train_ids, torch.tensor(train_labels)),
        TensorDataset(val_ids, torch.tensor(val_labels)),
        len(word2idx),
    )


def _load_imdb_raw() -> Tuple[List[str], List[int]]:
    if not _HAVE_HF:
        raise RuntimeError("IMDB loader requires `datasets` library.")
    data = load_dataset("imdb", split="train")
    texts = [ex["text"] for ex in data]
    labels = [ex["label"] for ex in data]
    return texts, labels


def load_imdb(
    seq_len: int = 256,
    vocab_size: int = 20000,
    val_ratio: float = 0.1,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """IMDB movie-review sentiment (binary)."""
    texts, labels = _load_imdb_raw()

    rng = random.Random(42)
    idx = list(range(len(texts)))
    rng.shuffle(idx)

    val_size = int(len(texts) * val_ratio)
    val_idx = set(idx[:val_size])

    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    for i in idx:
        if i in val_idx:
            val_texts.append(texts[i])
            val_labels.append(labels[i])
        else:
            train_texts.append(texts[i])
            train_labels.append(labels[i])

    word2idx = _build_vocab(train_texts, vocab_size)
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval)

    return (
        TensorDataset(train_ids, torch.tensor(train_labels)),
        TensorDataset(val_ids, torch.tensor(val_labels)),
        len(word2idx),
    ) 