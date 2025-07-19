import random
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import TensorDataset

from .dataset_loaders import _tokenise, build_vocab, texts_to_tensor  # updated relative import

CONLL_TAGS = [
    'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'
]

TAG2IDX: Dict[str, int] = {t: i for i, t in enumerate(CONLL_TAGS)}
IDX2TAG: Dict[int, str] = {i: t for t, i in TAG2IDX.items()}


def load_conll2003(
    seq_len: int = 128,
    vocab_size: int = 10000,
    shuffle_train: bool = False,
    data_root: Optional[str] = None,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, int, int]:
    """Return (train_ds, val_ds, test_ds, vocab_size, num_tags)."""
    import json, pathlib

    root_path = pathlib.Path(data_root) if data_root else pathlib.Path("/kaggle/input/conll2003-dataset")
    if root_path.exists():
        if (root_path / "train.json").exists():
            with open(root_path / "train.json") as f:
                train_split = json.load(f)
            with open(root_path / "validation.json") as f:
                val_split = json.load(f)
            with open(root_path / "test.json") as f:
                test_split = json.load(f)
        elif (root_path / "train.txt").exists():
            def _read_iob(path):
                texts, tags = [], []
                tokens, ner_tags = [], []
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            if tokens:
                                texts.append(" ".join(tokens))
                                tags.append(ner_tags)
                                tokens, ner_tags = [], []
                            continue
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        tok, tag = parts[0], parts[-1]
                        tokens.append(tok)
                        ner_tags.append(tag)
                return texts, tags

            train_texts, train_tags = _read_iob(root_path / "train.txt")
            val_texts, val_tags = _read_iob(root_path / "valid.txt")
            test_texts, test_tags = _read_iob(root_path / "test.txt")
        else:
            raise FileNotFoundError(f"No JSON or TXT files found in {root_path}")
    else:
        from datasets import load_dataset  # type: ignore

        dataset = load_dataset("conll2003")

        def _flatten(split):
            texts, tags = [], []
            for tokens, ner_tags in zip(split["tokens"], split["ner_tags"]):
                texts.append(" ".join(tokens))
                tags.append([IDX2TAG.get(t, 'O') for t in ner_tags])
            return texts, tags

        train_texts, train_tags = _flatten(dataset["train"])
        val_texts, val_tags = _flatten(dataset["validation"])
        test_texts, test_tags = _flatten(dataset["test"])

    word2idx = build_vocab(train_texts, vocab_size)

    def _encode(texts: List[str], tags: List[List[str]], shuffle: bool):
        seq_ids = texts_to_tensor(texts, word2idx, seq_len, shuffle)
        tag_ids = []
        for tag_seq in tags:
            tag_idx = [TAG2IDX.get(t, 0) for t in tag_seq][:seq_len]
            tag_idx += [TAG2IDX['O']] * (seq_len - len(tag_idx))
            tag_ids.append(tag_idx)
        return seq_ids, torch.tensor(tag_ids, dtype=torch.long)

    train_ids, train_tag_ids = _encode(train_texts, train_tags, shuffle_train)
    val_ids, val_tag_ids = _encode(val_texts, val_tags, False)
    test_ids, test_tag_ids = _encode(test_texts, test_tags, False)

    train_ds = TensorDataset(train_ids, train_tag_ids)
    val_ds = TensorDataset(val_ids, val_tag_ids)
    test_ds = TensorDataset(test_ids, test_tag_ids)

    return train_ds, val_ds, test_ds, len(word2idx), len(CONLL_TAGS) 