"""Lightweight sanity checks for the Field-Interference subsystem.

These tests cover the absolutely essential behaviours needed for daily
regression testing:

1. `TokenFieldInterference` – forward shape preservation & differentiability.
2. `DynamicFieldPropagator` – forward shape preservation on a small grid.

Everything runs on toy tensors (≤ 2 kB) so the full pytest suite remains under
20 s and <200 MB RAM.
"""

from __future__ import annotations

import torch

import pytest

from tfn.core.field_interference import TokenFieldInterference
from tfn.core.field_evolution import DynamicFieldPropagator


@pytest.mark.parametrize("batch, tokens, embed_dim", [(1, 4, 8), (2, 6, 16)])
def test_token_field_interference_forward_and_grad(batch: int, tokens: int, embed_dim: int) -> None:
    """Forward pass keeps shape and gradients propagate back to input."""
    torch.manual_seed(0)
    x = torch.randn(batch, tokens, embed_dim, requires_grad=True)

    module = TokenFieldInterference(embed_dim=embed_dim)
    out = module(x)
    assert out.shape == x.shape

    out.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dynamic_field_propagator_small_grid() -> None:
    """DynamicFieldPropagator should preserve shape on a tiny example."""
    torch.manual_seed(0)
    batch, tokens, embed_dim = 1, 5, 8
    x = torch.randn(batch, tokens, embed_dim, requires_grad=True)

    positions = torch.linspace(0, 1, tokens).reshape(1, tokens, 1).expand(batch, tokens, 1)

    propagator = DynamicFieldPropagator(
        embed_dim=embed_dim,
        pos_dim=1,
        evolution_type="diffusion",
        num_steps=2,
        dt=0.05,
    )

    out = propagator(x, positions)
    assert out.shape == x.shape

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() 