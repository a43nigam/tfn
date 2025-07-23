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

# ---------------------------------------------------------------------------
# Central tokenisation utils --------------------------------------------------
# ---------------------------------------------------------------------------

from tfn.data import tokenization as _tok
from tfn.utils.data_utils import split_indices

# Optional dependency ---------------------------------------------------------

_HAVE_HF = _tok.has_hf()


# ---------------------------------------------------------------------------
# Thin wrappers so rest of file stays unchanged --------------------------------
# ---------------------------------------------------------------------------


def _tokenise(text: str, tokenizer=None):
    return _tok.tokenize(text, tokenizer)


def _build_vocab(texts: List[str], vocab_size: int = 10000, tokenizer=None):
    return _tok.build_vocab(texts, vocab_size, tokenizer)


def _texts_to_tensor(
    texts: List[str],
    word2idx: Dict[str, int],
    seq_len: int = 128,
    shuffle: bool = False,
    tokenizer=None,
):
    return _tok.texts_to_tensor(texts, word2idx, seq_len=seq_len, shuffle_tokens=shuffle, tokenizer=tokenizer)

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
        train_hf = _tok.load_hf_dataset("ag_news", split="train")
        texts = [ex["text"] for ex in train_hf]
        labels = [ex["label"] for ex in train_hf]
    else:
        texts, labels = _download_agnews_csv()

    # Deterministic split via utils helper --------------------------------
    train_idx, val_idx = split_indices(len(texts), train_ratio=1 - val_split / len(texts), seed=42)
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    word2idx = _build_vocab(train_texts, vocab_size)
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval)

    meta = {"vocab_size": len(word2idx), "num_classes": int(max(labels)) + 1}
    return (
        TensorDataset(train_ids, torch.tensor(train_labels)),
        TensorDataset(val_ids, torch.tensor(val_labels)),
        meta,
    )


def _load_yelp_full_raw() -> Tuple[List[str], List[int]]:
    if not _HAVE_HF:
        raise RuntimeError("Yelp Review Full loader requires `datasets` library.")
    data = _tok.load_hf_dataset("yelp_review_full", split="train")
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

    train_idx, val_idx = split_indices(len(texts), train_ratio=1 - val_ratio, seed=42)
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    word2idx = _build_vocab(train_texts, vocab_size)
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval)

    meta = {"vocab_size": len(word2idx), "num_classes": 5}
    return (
        TensorDataset(train_ids, torch.tensor(train_labels)),
        TensorDataset(val_ids, torch.tensor(val_labels)),
        meta,
    )


def _load_imdb_raw() -> Tuple[List[str], List[int]]:
    if not _HAVE_HF:
        raise RuntimeError("IMDB loader requires `datasets` library.")
    data = _tok.load_hf_dataset("imdb", split="train")
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

    train_idx, val_idx = split_indices(len(texts), train_ratio=1 - val_ratio, seed=42)
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    word2idx = _build_vocab(train_texts, vocab_size)
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval)

    meta = {"vocab_size": len(word2idx), "num_classes": 2}
    return (
        TensorDataset(train_ids, torch.tensor(train_labels)),
        TensorDataset(val_ids, torch.tensor(val_labels)),
        meta,
    ) 