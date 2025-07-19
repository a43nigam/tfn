from __future__ import annotations

import torch
from tfn.utils.synthetic_sequence_tasks import SyntheticSequenceDataset
from tfn.model.seq_baselines import TFNSeqModel, SimpleTransformerSeqModel, SimplePerformerSeqModel


@torch.no_grad()
def _get_batch(task: str = "copy"):
    ds = SyntheticSequenceDataset(task=task, seq_len=8, vocab_size=20, num_samples=4, seed=0)
    x, y = zip(*(ds[i] for i in range(len(ds))))
    return torch.stack(x), torch.stack(y)


def test_tfn_forward_backward():
    x, y = _get_batch()
    model = TFNSeqModel(vocab_size=20, seq_len=8, embed_dim=16)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 20), y.view(-1))
    loss.backward()


def test_transformer_forward_backward():
    x, y = _get_batch()
    model = SimpleTransformerSeqModel(vocab_size=20, seq_len=8, embed_dim=16, num_layers=1, num_heads=2)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 20), y.view(-1))
    loss.backward()


def test_performer_forward_backward():
    x, y = _get_batch()
    model = SimplePerformerSeqModel(vocab_size=20, seq_len=8, embed_dim=16, num_layers=1, proj_dim=8)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 20), y.view(-1))
    loss.backward() 