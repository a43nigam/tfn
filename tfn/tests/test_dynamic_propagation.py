import pytest
import torch

from tfn.core.dynamic_propagation import DynamicFieldPropagator


@pytest.mark.parametrize(
    "evolution_type",
    ["diffusion", "wave", "cnn"],
)
def test_dynamic_propagator_forward_shapes(evolution_type: str) -> None:
    """DynamicFieldPropagator preserves [B, N, D] shape and gradients for various evolution types."""
    torch.manual_seed(0)
    batch, tokens, embed_dim = 2, 8, 16
    token_fields = torch.randn(batch, tokens, embed_dim, requires_grad=True)

    propagator = DynamicFieldPropagator(
        embed_dim=embed_dim,
        pos_dim=1,
        evolution_type=evolution_type,
        num_steps=3,
        dt=0.01,
    )

    out = propagator(token_fields)  # forward
    assert out.shape == (batch, tokens, embed_dim)

    # Ensure differentiability
    out.mean().backward()
    assert token_fields.grad is not None  # gradient flowed


def test_dynamic_propagator_unknown_evolution_type() -> None:
    """Unknown evolution type should raise ValueError during forward pass."""
    batch, tokens, embed_dim = 1, 4, 8
    x = torch.randn(batch, tokens, embed_dim)
    propagator = DynamicFieldPropagator(embed_dim=embed_dim, pos_dim=1, evolution_type="unknown")
    with pytest.raises(ValueError):
        _ = propagator(x) 