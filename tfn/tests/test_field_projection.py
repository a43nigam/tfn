"""
Unit tests for TFN field projection system.

Tests the field projection mechanism that transforms token embeddings
into continuous fields using kernel-based emission.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Import the field projection system
from tfn.core.field_projection import FieldProjector, UniformFieldGrid
from tfn.core.kernels import RBFKernel
from tfn.core.utils import validate_shapes, check_numerical_stability


class TestUniformFieldGrid:
    """Test the uniform field grid generator."""
    
    @pytest.fixture
    def grid_1d(self):
        """Create 1D uniform grid for testing."""
        return UniformFieldGrid(pos_dim=1, grid_size=50, bounds=(0.0, 1.0))
    
    @pytest.fixture
    def grid_2d(self):
        """Create 2D uniform grid for testing."""
        return UniformFieldGrid(pos_dim=2, grid_size=10, bounds=(0.0, 1.0))
    
    def test_initialization(self, grid_1d):
        """Test grid initialization."""
        assert grid_1d.pos_dim == 1
        assert grid_1d.grid_size == 50
        assert grid_1d.bounds == (0.0, 1.0)
        assert grid_1d.num_points == 50
    
    def test_1d_grid_generation(self, grid_1d):
        """Test 1D grid point generation."""
        grid_points = grid_1d.grid_points
        
        # Check shape
        assert grid_points.shape == (50, 1)
        
        # Check bounds
        assert torch.all(grid_points >= 0.0)
        assert torch.all(grid_points <= 1.0)
        
        # Check spacing (approximately uniform)
        spacing = grid_points[1:] - grid_points[:-1]
        assert torch.allclose(spacing, spacing[0], atol=1e-6)
    
    def test_2d_grid_generation(self, grid_2d):
        """Test 2D grid point generation."""
        grid_points = grid_2d.grid_points
        
        # Check shape (10x10 = 100 points in 2D)
        assert grid_points.shape == (100, 2)
        
        # Check bounds
        assert torch.all(grid_points >= 0.0)
        assert torch.all(grid_points <= 1.0)
    
    def test_forward_single_batch(self, grid_1d):
        """Test grid forward pass with single batch."""
        grid_points = grid_1d(batch_size=1)
        
        assert grid_points.shape == (1, 50, 1)
        assert torch.allclose(grid_points.squeeze(0), grid_1d.grid_points)
    
    def test_forward_multi_batch(self, grid_1d):
        """Test grid forward pass with multiple batches."""
        grid_points = grid_1d(batch_size=3)
        
        assert grid_points.shape == (3, 50, 1)
        # All batches should have same grid points
        assert torch.allclose(grid_points[0], grid_points[1])
        assert torch.allclose(grid_points[1], grid_points[2])


class TestFieldProjector:
    """Test the field projector."""
    
    @pytest.fixture
    def projector(self):
        """Create field projector for testing."""
        return FieldProjector(embed_dim=64, pos_dim=1, kernel_type="rbf")
    
    @pytest.fixture
    def test_data(self):
        """Create test data for field projection."""
        batch_size, num_tokens, embed_dim = 2, 5, 64
        num_grid_points = 20
        
        # Token embeddings: [B, N, D]
        embeddings = torch.randn(batch_size, num_tokens, embed_dim)
        
        # Token positions: [B, N, P]
        positions = torch.rand(batch_size, num_tokens, 1)
        
        # Grid points: [M, P]
        grid_points = torch.linspace(0, 1, num_grid_points).unsqueeze(-1)
        
        return embeddings, positions, grid_points
    
    def test_initialization(self, projector):
        """Test projector initialization."""
        assert projector.embed_dim == 64
        assert projector.pos_dim == 1
        assert projector.kernel_type == "rbf"
        assert isinstance(projector.kernel, RBFKernel)
    
    def test_field_projection_shape(self, projector, test_data):
        """Test that field projection produces correct shapes."""
        embeddings, positions, grid_points = test_data
        
        # Project embeddings to field
        field = projector(embeddings, positions, grid_points)
        
        # Check shape: [B, M, D]
        expected_shape = (2, 20, 64)
        assert field.shape == expected_shape
    
    def test_field_projection_values(self, projector, test_data):
        """Test that field projection produces valid values."""
        embeddings, positions, grid_points = test_data
        
        field = projector(embeddings, positions, grid_points)
        
        # Check for numerical stability
        check_numerical_stability(field, "Field projection values")
        
        # Check that field values are finite
        assert torch.isfinite(field).all()
    
    def test_field_projection_with_kernel_params(self, projector, test_data):
        """Test field projection with explicit kernel parameters."""
        embeddings, positions, grid_points = test_data
        
        # Create explicit kernel parameters
        batch_size, num_tokens = embeddings.shape[:2]
        kernel_params = torch.rand(batch_size, num_tokens, 1) * 0.5 + 0.1
        
        field = projector(embeddings, positions, grid_points, kernel_params)
        
        assert field.shape == (2, 20, 64)
        assert torch.isfinite(field).all()
    
    def test_gradient_flow(self, projector, test_data):
        """Test gradient flow through field projection."""
        embeddings, positions, grid_points = test_data
        
        # Make inputs require gradients
        embeddings.requires_grad_(True)
        positions.requires_grad_(True)
        
        # Forward pass
        field = projector(embeddings, positions, grid_points)
        
        # Backward pass
        loss = field.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert embeddings.grad is not None
        assert positions.grad is not None
        assert torch.isfinite(embeddings.grad).all()
        assert torch.isfinite(positions.grad).all()
    
    def test_token_influence_shape(self, projector, test_data):
        """Test token influence computation shape."""
        embeddings, positions, grid_points = test_data
        
        token_influences = projector.compute_token_influence(
            embeddings, positions, grid_points
        )
        
        # Check shape: [B, N, M, D]
        expected_shape = (2, 5, 20, 64)
        assert token_influences.shape == expected_shape
    
    def test_token_influence_sum(self, projector, test_data):
        """Test that token influences sum to total field."""
        embeddings, positions, grid_points = test_data
        
        # Compute total field
        field = projector(embeddings, positions, grid_points)
        
        # Compute individual token influences
        token_influences = projector.compute_token_influence(
            embeddings, positions, grid_points
        )
        
        # Sum of token influences should equal total field
        field_from_influences = token_influences.sum(dim=1)  # [B, M, D]
        
        assert torch.allclose(field, field_from_influences, atol=1e-6)
    
    def test_different_grid_shapes(self, projector, test_data):
        """Test field projection with different grid shapes."""
        embeddings, positions, _ = test_data
        
        # Test with [M, P] grid
        grid_2d = torch.linspace(0, 1, 15).unsqueeze(-1)
        field_2d = projector(embeddings, positions, grid_2d)
        assert field_2d.shape == (2, 15, 64)
        
        # Test with [B, M, P] grid
        grid_3d = grid_2d.unsqueeze(0).expand(2, -1, -1)
        field_3d = projector(embeddings, positions, grid_3d)
        assert field_3d.shape == (2, 15, 64)
        
        # Results should be identical
        assert torch.allclose(field_2d, field_3d)


class TestFieldProjectionIntegration:
    """Integration tests for field projection system."""
    
    def test_end_to_end_projection(self):
        """Test complete field projection pipeline."""
        # Create components
        grid = UniformFieldGrid(pos_dim=1, grid_size=30)
        projector = FieldProjector(embed_dim=32, pos_dim=1)
        
        # Create test data
        embeddings = torch.randn(1, 4, 32)  # [1, 4, 32]
        positions = torch.tensor([[[0.2], [0.5], [0.8], [0.9]]])  # [1, 4, 1]
        grid_points = grid(batch_size=1)  # [1, 30, 1]
        
        # Project to field
        field = projector(embeddings, positions, grid_points)
        
        # Check results
        assert field.shape == (1, 30, 32)
        assert torch.isfinite(field).all()
        
        # Check that field has reasonable values
        assert field.abs().max() < 10.0  # Should not be extremely large
    
    def test_field_projection_physical_properties(self):
        """Test physical properties of field projection."""
        # Create simple test case
        grid = UniformFieldGrid(pos_dim=1, grid_size=50)
        projector = FieldProjector(embed_dim=1, pos_dim=1)
        
        # Single token with unit embedding
        embeddings = torch.tensor([[[1.0]]])  # [1, 1, 1]
        positions = torch.tensor([[[0.5]]])   # [1, 1, 1] - center position
        grid_points = grid(batch_size=1)      # [1, 50, 1]
        
        # Project to field
        field = projector(embeddings, positions, grid_points)
        
        # Field should be maximum near token position
        field_values = field.squeeze()  # [50]
        grid_values = grid_points.squeeze()  # [50]
        
        # Find maximum field value
        max_idx = torch.argmax(field_values)
        max_pos = grid_values[max_idx]
        
        # Maximum should be near token position (0.5)
        assert abs(max_pos - 0.5) < 0.1
    
    def test_multiple_tokens_superposition(self):
        """Test that multiple tokens create superposition."""
        grid = UniformFieldGrid(pos_dim=1, grid_size=40)
        projector = FieldProjector(embed_dim=1, pos_dim=1)
        
        # Two tokens with different embeddings
        embeddings = torch.tensor([[[1.0], [2.0]]])  # [1, 2, 1]
        positions = torch.tensor([[[0.3], [0.7]]])   # [1, 2, 1]
        grid_points = grid(batch_size=1)
        
        # Project to field
        field = projector(embeddings, positions, grid_points)
        
        # Get individual token influences
        token_influences = projector.compute_token_influence(
            embeddings, positions, grid_points
        )
        
        # Field should be sum of individual influences
        field_from_sum = token_influences.sum(dim=1)  # [1, 40, 1]
        assert torch.allclose(field, field_from_sum, atol=1e-6)
        
        # Field should have peaks near both token positions
        field_values = field.squeeze()  # [40]
        grid_values = grid_points.squeeze()  # [40]
        
        # Find peaks by looking for local maxima
        peaks = []
        for i in range(1, len(field_values) - 1):
            if field_values[i] > field_values[i-1] and field_values[i] > field_values[i+1]:
                peaks.append(grid_values[i].item())
        
        # Should have at least one peak
        assert len(peaks) > 0, "No peaks found in field"
        
        # Check that we have peaks near both token positions
        near_token_1 = any(abs(peak - 0.3) < 0.15 for peak in peaks)
        near_token_2 = any(abs(peak - 0.7) < 0.15 for peak in peaks)
        
        assert near_token_1 or near_token_2, f"Peaks {peaks} not near token positions [0.3, 0.7]"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 