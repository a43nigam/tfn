from __future__ import annotations

"""tfn.datasets.glue_loader
GLUE benchmark dataset loaders with Kaggle dataset support.
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
# GLUE dataset loaders
# ---------------------------------------------------------------------------

def load_sst2(
    seq_len: int = 128,
    vocab_size: int = 10000,
    val_split: int = 1000,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """SST-2 sentiment analysis (binary)."""
    import pandas as pd
    
    # Try Kaggle path first
    kaggle_path = Path("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path, sep='\t')
        texts = df['Phrase'].tolist()
        labels = df['Sentiment'].tolist()
    elif _HAVE_HF:
        train_data = load_dataset("glue", "sst2", split="train")
        texts = [ex["sentence"] for ex in train_data]
        labels = [ex["label"] for ex in train_data]
    else:
        raise RuntimeError("SST-2 loader requires either Kaggle dataset or `datasets` library.")

    # Split train/val
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


def load_mrpc(
    seq_len: int = 128,
    vocab_size: int = 10000,
    val_split: int = 500,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """MRPC paraphrase detection (binary)."""
    import pandas as pd
    
    # Try Kaggle path first
    kaggle_path = Path("/kaggle/input/microsoft-research-paraphrase-corpus/msr_paraphrase_train.txt")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path, sep='\t', header=None)
        texts = []
        labels = []
        for _, row in df.iterrows():
            if len(row) >= 5:
                texts.append(f"{row[3]} {row[4]}")
                labels.append(row[0])
    elif _HAVE_HF:
        train_data = load_dataset("glue", "mrpc", split="train")
        texts = [f"{ex['sentence1']} {ex['sentence2']}" for ex in train_data]
        labels = [ex["label"] for ex in train_data]
    else:
        raise RuntimeError("MRPC loader requires either Kaggle dataset or `datasets` library.")

    # Split train/val
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


def load_qqp(
    seq_len: int = 128,
    vocab_size: int = 15000,
    val_split: int = 2000,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """QQP question similarity (binary)."""
    import pandas as pd
    
    # Try Kaggle path first
    kaggle_path = Path("/kaggle/input/quora-question-pairs/train.csv")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path)
        texts = []
        labels = []
        for _, row in df.iterrows():
            if pd.notna(row['question1']) and pd.notna(row['question2']):
                texts.append(f"{row['question1']} {row['question2']}")
                labels.append(row['is_duplicate'])
    elif _HAVE_HF:
        train_data = load_dataset("glue", "qqp", split="train")
        texts = [f"{ex['question1']} {ex['question2']}" for ex in train_data]
        labels = [ex["label"] for ex in train_data]
    else:
        raise RuntimeError("QQP loader requires either Kaggle dataset or `datasets` library.")

    # Split train/val
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


def load_qnli(
    seq_len: int = 256,
    vocab_size: int = 15000,
    val_split: int = 1000,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """QNLI question-answer entailment (binary)."""
    if not _HAVE_HF:
        raise RuntimeError("QNLI loader requires `datasets` library.")
    
    train_data = load_dataset("glue", "qnli", split="train")
    texts = [f"{ex['question']} {ex['sentence']}" for ex in train_data]
    labels = [ex["label"] for ex in train_data]

    # Split train/val
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


def load_rte(
    seq_len: int = 128,
    vocab_size: int = 10000,
    val_split: int = 500,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """RTE textual entailment (binary)."""
    if not _HAVE_HF:
        raise RuntimeError("RTE loader requires `datasets` library.")
    
    train_data = load_dataset("glue", "rte", split="train")
    texts = [f"{ex['sentence1']} {ex['sentence2']}" for ex in train_data]
    labels = [ex["label"] for ex in train_data]

    # Split train/val
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


def load_cola(
    seq_len: int = 128,
    vocab_size: int = 10000,
    val_split: int = 500,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """CoLA linguistic acceptability (binary)."""
    if not _HAVE_HF:
        raise RuntimeError("CoLA loader requires `datasets` library.")
    
    train_data = load_dataset("glue", "cola", split="train")
    texts = [ex["sentence"] for ex in train_data]
    labels = [ex["label"] for ex in train_data]

    # Split train/val
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


def load_stsb(
    seq_len: int = 128,
    vocab_size: int = 10000,
    val_split: int = 500,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """STS-B semantic similarity (regression)."""
    if not _HAVE_HF:
        raise RuntimeError("STS-B loader requires `datasets` library.")
    
    train_data = load_dataset("glue", "stsb", split="train")
    texts = [f"{ex['sentence1']} {ex['sentence2']}" for ex in train_data]
    labels = [ex["label"] for ex in train_data]

    # Split train/val
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
        TensorDataset(train_ids, torch.tensor(train_labels, dtype=torch.float)),
        TensorDataset(val_ids, torch.tensor(val_labels, dtype=torch.float)),
        len(word2idx),
    )


def load_wnli(
    seq_len: int = 128,
    vocab_size: int = 10000,
    val_split: int = 300,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """WNLI Winograd NLI (binary)."""
    if not _HAVE_HF:
        raise RuntimeError("WNLI loader requires `datasets` library.")
    
    train_data = load_dataset("glue", "wnli", split="train")
    texts = [f"{ex['sentence1']} {ex['sentence2']}" for ex in train_data]
    labels = [ex["label"] for ex in train_data]

    # Split train/val
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