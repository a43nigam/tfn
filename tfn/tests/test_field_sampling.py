import torch
import pytest
from tfn.core.field_sampling import FieldSampler

def make_test_field(batch_size=2, grid_size=10, embed_dim=4):
    # Uniform grid in [0, 1]
    grid_points = torch.linspace(0, 1, grid_size).view(1, grid_size, 1).repeat(batch_size, 1, 1)
    # Field: f(x) = x (for each dim), shape [B, G, D]
    field = grid_points.expand(-1, -1, embed_dim).clone()
    return field, grid_points

def test_shape_and_basic_sampling():
    """Test output shape and basic sampling for nearest and linear modes."""
    B, G, D, N = 2, 10, 4, 5
    field, grid_points = make_test_field(B, G, D)
    sample_positions = torch.rand(B, N, 1)
    for mode in ['nearest', 'linear']:
        sampler = FieldSampler(mode=mode)
        out = sampler(field, grid_points, sample_positions)
        assert out.shape == (B, N, D)

def test_exact_grid_points():
    """Sampling exactly at grid points should return the field value at that grid point (for both modes)."""
    B, G, D = 1, 8, 3
    field, grid_points = make_test_field(B, G, D)
    sampler = FieldSampler(mode='linear')
    # Sample at all grid points
    out = sampler(field, grid_points, grid_points)
    assert torch.allclose(out, field, atol=1e-6)
    sampler = FieldSampler(mode='nearest')
    out = sampler(field, grid_points, grid_points)
    assert torch.allclose(out, field, atol=1e-6)

def test_linear_interpolation_accuracy():
    """Test that linear interpolation is accurate for a linear field."""
    B, G, D = 1, 10, 2
    field, grid_points = make_test_field(B, G, D)
    sampler = FieldSampler(mode='linear')
    # Sample at midpoints between grid points
    midpoints = (grid_points[:, :-1, :] + grid_points[:, 1:, :]) / 2
    out = sampler(field, grid_points, midpoints)
    # For a linear field, value at midpoint should be mean of neighbors
    expected = (field[:, :-1, :] + field[:, 1:, :]) / 2
    assert torch.allclose(out, expected, atol=1e-6)

def test_nearest_interpolation():
    """Test that nearest interpolation picks the closest grid value."""
    B, G, D = 1, 6, 1
    field, grid_points = make_test_field(B, G, D)
    sampler = FieldSampler(mode='nearest')
    # Sample at points just left/right of grid points
    offsets = torch.tensor([[-0.01], [0.01]])
    # Create samples for both batches properly
    samples = grid_points.repeat(2,1,1)[:, 2:4, :] + offsets.unsqueeze(1)  # shape [2, 2, 1]
    out = sampler(field.repeat(2,1,1), grid_points.repeat(2,1,1), samples)
    # Should match grid_points[:,2,:] for -0.01, grid_points[:,3,:] for +0.01
    assert torch.allclose(out[0,0], field[0,2], atol=1e-6)
    assert torch.allclose(out[0,1], field[0,3], atol=1e-6)
    assert torch.allclose(out[1,0], field[0,2], atol=1e-6)  # 0.41 → 0.4
    assert torch.allclose(out[1,1], field[0,3], atol=1e-6)  # 0.61 → 0.6

def test_gradient_flow():
    """Test that gradients flow through the sampler."""
    B, G, D, N = 1, 8, 2, 4
    field, grid_points = make_test_field(B, G, D)
    field.requires_grad_()
    sample_positions = torch.rand(B, N, 1, requires_grad=True)
    sampler = FieldSampler(mode='linear')
    out = sampler(field, grid_points, sample_positions)
    loss = out.sum()
    loss.backward()
    assert field.grad is not None
    assert sample_positions.grad is not None
    assert field.grad.abs().sum() > 0
    assert sample_positions.grad.abs().sum() > 0

def test_out_of_bounds():
    """Sampling out of grid bounds should clamp to edge values (no crash)."""
    B, G, D, N = 1, 6, 2, 3
    field, grid_points = make_test_field(B, G, D)
    sampler = FieldSampler(mode='linear')
    # Sample positions outside [0,1]
    sample_positions = torch.tensor([[[-0.5],[1.5],[2.0]]], dtype=grid_points.dtype)
    out = sampler(field, grid_points, sample_positions)
    # Should match edge values
    assert torch.allclose(out[:,0], field[:,0], atol=1e-6)
    assert torch.allclose(out[:,1], field[:,-1], atol=1e-6)
    assert torch.allclose(out[:,2], field[:,-1], atol=1e-6) 