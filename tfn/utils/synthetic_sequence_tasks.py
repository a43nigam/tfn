from __future__ import annotations

"""synthetic_sequence_tasks.py
Utilities to generate simple algorithmic sequence-to-sequence datasets (copy, reverse).

These tasks are commonly used to benchmark sequence models for long-context retention
and serve as the first stress-test for our 1-D Token Field Network.

Each dataset sample consists of an integer input sequence and a target sequence of the
same length (token-level supervision). The module exposes:

• SyntheticSequenceDataset – torch.utils.data.Dataset subclass.
• get_synthetic_sequence_dataloaders – convenience function returning train / val
  DataLoader pairs with identical vocabulary.

All generated tensors are on the CPU; callers are expected to move them to the desired
device in their training loop.

NOTE: We deliberately avoid any torchtext / HF datasets dependencies to keep the
module lightweight and fully deterministic
(important for unit tests and reproducibility).
"""

from typing import Tuple, List
import random
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = [
    "SyntheticSequenceDataset",
    "get_synthetic_sequence_dataloaders",
]


class SyntheticSequenceDataset(Dataset):
    """Dataset for synthetic copy / reverse tasks.

    Parameters
    ----------
    task : str
        One of {"copy", "reverse"}.
    seq_len : int
        Length of each input sequence.
    vocab_size : int
        Size of the discrete vocabulary *including* PAD (0).
        Random tokens are drawn uniformly from ``[1, vocab_size - 1]``.
    num_samples : int
        Number of (input, target) pairs to generate.
    seed : int
        RNG seed for reproducibility. Each index will deterministically map to
        a unique (input, target) pair when the same seed is given.
    """

    def __init__(
        self,
        task: str = "copy",
        seq_len: int = 128,
        vocab_size: int = 100,
        num_samples: int = 10_000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if task not in {"copy", "reverse"}:
            raise ValueError(f"Unsupported task '{task}'. Choose from 'copy', 'reverse'.")
        if vocab_size < 3:
            raise ValueError("vocab_size must be at least 3 (includes <PAD> and <UNK>).")
        self.task = task
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seed = seed

        # Pre-compute RNG states for deterministic access without storing all samples
        self._rng = random.Random(seed)
        self._base_states: List[int] = [self._rng.randrange(2 ** 32) for _ in range(num_samples)]

    # ----------------------------- PyTorch API ----------------------------- #

    def __len__(self) -> int:  # noqa: D401 – imperative style OK here
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Restore the *per-sample* RNG state to generate deterministically
        rng_state = self._base_states[idx]
        rng = random.Random(rng_state)
        # Tokens drawn from [1, vocab_size - 1]; 0 reserved for PAD
        seq = [rng.randint(1, self.vocab_size - 1) for _ in range(self.seq_len)]
        x = torch.tensor(seq, dtype=torch.long)

        if self.task == "copy":
            y = x.clone()
        elif self.task == "reverse":
            y = torch.flip(x, dims=[0])
        else:  # unreachable due to validation in __init__; keeps mypy happy
            raise RuntimeError("Invalid task type")

        return x, y


# ---------------------------------------------------------------------------
# Convenience helper (mainly for CLI training script)
# ---------------------------------------------------------------------------

def get_synthetic_sequence_dataloaders(
    task: str = "copy",
    seq_len: int = 128,
    vocab_size: int = 100,
    train_samples: int = 10_000,
    val_samples: int = 2_000,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Return train & validation DataLoader pair and effective vocab size.

    This helper is primarily aimed at unit tests / quick benchmarks – it keeps
    everything in memory and uses deterministic synthetic data, so multiple
    instantiations with the same parameters yield identical batches.
    """

    train_ds = SyntheticSequenceDataset(
        task=task,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_samples=train_samples,
        seed=seed,
    )
    val_ds = SyntheticSequenceDataset(
        task=task,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_samples=val_samples,
        seed=seed + 1,  # ensure different samples
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, vocab_size 