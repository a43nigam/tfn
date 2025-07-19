from __future__ import annotations

import torch
from tfn.utils.synthetic_sequence_tasks import SyntheticSequenceDataset


def test_copy_task_shapes():
    ds = SyntheticSequenceDataset(task="copy", seq_len=32, vocab_size=50, num_samples=10, seed=123)
    x, y = ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)
    # Copy means identical tensors
    assert torch.equal(x, y)


def test_reverse_task_targets():
    ds = SyntheticSequenceDataset(task="reverse", seq_len=16, vocab_size=30, num_samples=5, seed=99)
    x, y = ds[2]
    assert torch.equal(y, torch.flip(x, dims=[0])) 