from __future__ import annotations

"""tfn.datasets.glue_loader
GLUE benchmark dataset loaders with Kaggle dataset support.
"""

from typing import List, Tuple, Dict
import random
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

# ---------------------------------------------------------------------------
# Central tokenisation utilities --------------------------------------------
# ---------------------------------------------------------------------------

from tfn.data import tokenization as _tok

# Alias helpers so the rest of this file stays unchanged --------------------

def _get_tokenizer():
    """Return the default tokenizer from `tfn.data.tokenization`."""
    return _tok.get_tokenizer()


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
# GLUE dataset loaders with improved tokenization
# ---------------------------------------------------------------------------

def load_sst2(
    seq_len: int = 128,
    vocab_size: int = 10000,
    val_split: int = 1000,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """SST-2 sentiment analysis (binary) with proper tokenization."""
    import pandas as pd
    
    # Initialize tokenizer
    print("🔧 Initializing tokenizer...")
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        print(f"✅ Using {tokenizer.__class__.__name__} tokenizer")
    elif _tok.has_nltk():
        print("✅ Using NLTK tokenizer")
    else:
        print("⚠️  Using basic regex tokenizer (fallback)")
    
    # Try Kaggle path first
    kaggle_path = Path("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path, sep='\t')
        texts = df['Phrase'].astype(str).tolist()
        labels = df['Sentiment'].tolist()
    elif _tok.has_hf():
        train_data = _tok.load_hf_dataset("glue", "sst2", split="train")
        texts = [ex["sentence"] for ex in train_data]
        labels = [ex["label"] for ex in train_data]
    else:
        raise RuntimeError("SST-2 loader requires either Kaggle dataset or `datasets` library.")

    # Split train/val with better validation split
    rng = random.Random(42)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    
    # Use larger validation split for more reliable metrics (10% instead of fixed 1000)
    val_split = max(val_split, int(0.1 * len(texts)))
    val_idx = set(idx[:val_split])
    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    for i in idx:
        if i in val_idx:
            val_texts.append(texts[i])
            val_labels.append(labels[i])
        else:
            train_texts.append(texts[i])
            train_labels.append(labels[i])

    print(f"📊 Dataset split: {len(train_texts)} train, {len(val_texts)} validation")
    
    # Build vocabulary with proper tokenization
    word2idx = _build_vocab(train_texts, vocab_size, tokenizer)
    print(f"📝 Built vocabulary: {len(word2idx)} tokens")
    
    # Convert to tensors with proper tokenization
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train, tokenizer)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval, tokenizer)

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
    """MRPC paraphrase detection (binary) with proper tokenization."""
    import pandas as pd
    
    # Initialize tokenizer
    tokenizer = _get_tokenizer()
    
    # Try Kaggle path first
    kaggle_path = Path("/kaggle/input/microsoft-research-paraphrase-corpus/msr_paraphrase_train.txt")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path, sep='\t', header=None)
        texts = []
        labels = []
        for _, row in df.iterrows():
            if len(row) >= 5:
                texts.append(f"{row[3]} [SEP] {row[4]}")  # Add [SEP] token between sentences
                labels.append(row[0])
    elif _tok.has_hf():
        train_data = _tok.load_hf_dataset("glue", "mrpc", split="train")
        texts = [f"{ex['sentence1']} [SEP] {ex['sentence2']}" for ex in train_data]
        labels = [ex["label"] for ex in train_data]
    else:
        raise RuntimeError("MRPC loader requires either Kaggle dataset or `datasets` library.")

    # Split train/val with better validation split
    rng = random.Random(42)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    
    # Use larger validation split for more reliable metrics
    val_split = max(val_split, int(0.1 * len(texts)))
    val_idx = set(idx[:val_split])
    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    for i in idx:
        if i in val_idx:
            val_texts.append(texts[i])
            val_labels.append(labels[i])
        else:
            train_texts.append(texts[i])
            train_labels.append(labels[i])

    print(f"📊 MRPC split: {len(train_texts)} train, {len(val_texts)} validation")
    
    # Build vocabulary with proper tokenization
    word2idx = _build_vocab(train_texts, vocab_size, tokenizer)
    
    # Convert to tensors with proper tokenization
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train, tokenizer)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval, tokenizer)

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
    elif _tok.has_hf():
        train_data = _tok.load_hf_dataset("glue", "qqp", split="train")
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
    if not _tok.has_hf():
        raise RuntimeError("QNLI loader requires `datasets` library.")
    
    train_data = _tok.load_hf_dataset("glue", "qnli", split="train")
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
    """RTE textual entailment (binary) with proper tokenization."""
    if not _tok.has_hf():
        raise RuntimeError("RTE loader requires `datasets` library.")
    
    # Initialize tokenizer
    tokenizer = _get_tokenizer()
    
    train_data = _tok.load_hf_dataset("glue", "rte", split="train")
    texts = [f"{ex['sentence1']} [SEP] {ex['sentence2']}" for ex in train_data]
    labels = [ex["label"] for ex in train_data]

    # Split train/val with better validation split
    rng = random.Random(42)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    
    # Use larger validation split for more reliable metrics
    val_split = max(val_split, int(0.1 * len(texts)))
    val_idx = set(idx[:val_split])
    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    for i in idx:
        if i in val_idx:
            val_texts.append(texts[i])
            val_labels.append(labels[i])
        else:
            train_texts.append(texts[i])
            train_labels.append(labels[i])

    print(f"📊 RTE split: {len(train_texts)} train, {len(val_texts)} validation")
    
    # Build vocabulary with proper tokenization
    word2idx = _build_vocab(train_texts, vocab_size, tokenizer)
    
    # Convert to tensors with proper tokenization
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train, tokenizer)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval, tokenizer)

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
    """CoLA linguistic acceptability (binary) with proper tokenization."""
    if not _tok.has_hf():
        raise RuntimeError("CoLA loader requires `datasets` library.")
    
    # Initialize tokenizer
    tokenizer = _get_tokenizer()
    
    train_data = _tok.load_hf_dataset("glue", "cola", split="train")
    texts = [ex["sentence"] for ex in train_data]
    labels = [ex["label"] for ex in train_data]

    # Split train/val with better validation split
    rng = random.Random(42)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    
    # Use larger validation split for more reliable metrics
    val_split = max(val_split, int(0.1 * len(texts)))
    val_idx = set(idx[:val_split])
    train_texts, train_labels, val_texts, val_labels = [], [], [], []
    for i in idx:
        if i in val_idx:
            val_texts.append(texts[i])
            val_labels.append(labels[i])
        else:
            train_texts.append(texts[i])
            train_labels.append(labels[i])

    print(f"📊 CoLA split: {len(train_texts)} train, {len(val_texts)} validation")
    
    # Build vocabulary with proper tokenization
    word2idx = _build_vocab(train_texts, vocab_size, tokenizer)
    
    # Convert to tensors with proper tokenization
    train_ids = _texts_to_tensor(train_texts, word2idx, seq_len, shuffle_train, tokenizer)
    val_ids = _texts_to_tensor(val_texts, word2idx, seq_len, shuffle_eval, tokenizer)

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
    if not _tok.has_hf():
        raise RuntimeError("STS-B loader requires `datasets` library.")
    
    train_data = _tok.load_hf_dataset("glue", "stsb", split="train")
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
    if not _tok.has_hf():
        raise RuntimeError("WNLI loader requires `datasets` library.")
    
    train_data = _tok.load_hf_dataset("glue", "wnli", split="train")
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