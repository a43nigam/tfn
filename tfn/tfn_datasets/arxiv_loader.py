from __future__ import annotations

"""tfn.datasets.arxiv_loader
Arxiv papers dataset loader with Kaggle dataset support.
"""

from typing import List, Tuple, Dict
import random
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

# Optional dependency ---------------------------------------------------------
from tfn.data import tokenization as _tok

# ---------------------------------------------------------------------------
# Tokenisation helpers (delegate to central util) ----------------------------
# ---------------------------------------------------------------------------


def _tokenise(text: str, tokenizer=None):
    return _tok.tokenize(text, tokenizer)


def _build_vocab(texts: List[str], vocab_size: int = 20000, tokenizer=None):
    return _tok.build_vocab(texts, vocab_size, tokenizer)


def _texts_to_tensor(
    texts: List[str],
    word2idx: Dict[str, int],
    seq_len: int = 512,
    shuffle: bool = False,
    tokenizer=None,
):
    return _tok.texts_to_tensor(texts, word2idx, seq_len=seq_len, shuffle_tokens=shuffle, tokenizer=tokenizer)

# HF datasets optional
_HAVE_HF = _tok.has_hf()


def _build_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], List[int]]:
    """Build label mapping and convert labels to indices."""
    unique_labels = sorted(list(set(labels)))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_indices = [label2idx[label] for label in labels]
    return label2idx, label_indices


def load_arxiv(
    seq_len: int = 512,
    vocab_size: int = 20000,
    val_split: int = 2000,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """Arxiv papers classification by subject category."""
    import pandas as pd
    
    # Try Kaggle path first
    kaggle_path = Path("/kaggle/input/arxiv-papers-2021/arxiv_papers_2021.csv")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path)
        texts = []
        labels = []
        for _, row in df.iterrows():
            if pd.notna(row['title']) and pd.notna(row['abstract']) and pd.notna(row['categories']):
                # Combine title and abstract
                text = f"{row['title']} {row['abstract']}"
                # Take first category (primary subject)
                category = row['categories'].split()[0]
                texts.append(text)
                labels.append(category)
    elif _HAVE_HF:
        train_data = _tok.load_hf_dataset("arxiv_dataset", split="train")
        texts = []
        labels = []
        for ex in train_data:
            if ex['title'] and ex['abstract'] and ex['categories']:
                text = f"{ex['title']} {ex['abstract']}"
                category = ex['categories'].split()[0]
                texts.append(text)
                labels.append(category)
    else:
        raise RuntimeError("Arxiv loader requires either Kaggle dataset or `datasets` library.")

    # Build label mapping
    label2idx, label_indices = _build_label_mapping(labels)
    num_classes = len(label2idx)
    print(f"Found {num_classes} categories: {list(label2idx.keys())[:10]}...")

    # Split train/val
    rng = random.Random(42)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    
    val_idx = set(idx[:val_split])
    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    for i in idx:
        if i in val_idx:
            val_texts.append(texts[i])
            val_labels.append(label_indices[i])
        else:
            train_texts.append(texts[i])
            train_labels.append(label_indices[i])

    word2idx = _build_vocab(train_texts, vocab_size)
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval)

    return (
        TensorDataset(train_ids, torch.tensor(train_labels)),
        TensorDataset(val_ids, torch.tensor(val_labels)),
        len(word2idx),
    ) 