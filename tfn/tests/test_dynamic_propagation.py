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

    # Create dummy grid_points [B, N, P] (P=1 for 1D)
    grid_points = torch.linspace(0, 1, tokens).reshape(1, tokens, 1).expand(batch, tokens, 1)
    out = propagator(token_fields, grid_points)  # forward
    assert out.shape == (batch, tokens, embed_dim)

    # Ensure differentiability
    out.mean().backward()
    assert token_fields.grad is not None  # gradient flowed


def test_dynamic_propagator_unknown_evolution_type() -> None:
    """Unknown evolution type should raise ValueError during forward pass."""
    batch, tokens, embed_dim = 1, 4, 8
    x = torch.randn(batch, tokens, embed_dim)
    propagator = DynamicFieldPropagator(embed_dim=embed_dim, pos_dim=1, evolution_type="unknown")
    grid_points = torch.linspace(0, 1, tokens).reshape(1, tokens, 1)
    with pytest.raises(ValueError):
        _ = propagator(x, grid_points) 