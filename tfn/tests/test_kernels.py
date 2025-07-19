"""
Unit tests for TFN kernel system.

Tests all kernel types (RBF, Compact, Fourier, Learnable) for:
- Shape validation
- Gradient flow
- Numerical stability
- Physical properties
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Import the kernel system
from ..core.kernels import (
    KernelBasis, RBFKernel, CompactKernel, FourierKernel, 
    LearnableKernel, KernelFactory
)
from ..core.utils import validate_shapes, check_gradient_flow, check_numerical_stability


class TestKernelBasis:
    """Test the abstract kernel base class."""
    
    def test_abstract_base_class(self):
        """Test that KernelBasis cannot be instantiated directly."""
        with pytest.raises(TypeError):
            KernelBasis(pos_dim=1)
    
    def test_pos_dim_attribute(self):
        """Test that pos_dim is properly set."""
        kernel = RBFKernel(pos_dim=3)
        assert kernel.pos_dim == 3


class TestRBFKernel:
    """Test the RBF (Gaussian) kernel."""
    
    @pytest.fixture
    def kernel(self):
        """Create RBF kernel for testing."""
        return RBFKernel(pos_dim=1)
    
    @pytest.fixture
    def test_data(self):
        """Create test data for kernel computation."""
        batch_size, num_tokens, num_grid_points = 2, 5, 10
        
        # Grid points: [M, P]
        z = torch.linspace(0, 1, num_grid_points).unsqueeze(-1)  # [10, 1]
        
        # Token positions: [B, N, P]
        mu = torch.rand(batch_size, num_tokens, 1)  # [2, 5, 1]
        
        # Spread parameters: [B, N, 1]
        sigma = torch.rand(batch_size, num_tokens, 1) * 0.5 + 0.1  # [2, 5, 1]
        
        return z, mu, sigma
    
    def test_initialization(self, kernel):
        """Test kernel initialization."""
        assert kernel.pos_dim == 1
        assert kernel.min_sigma == 0.1
        assert kernel.max_sigma == 10.0
    
    def test_shape_validation(self, kernel, test_data):
        """Test that kernel handles different input shapes correctly."""
        z, mu, sigma = test_data
        
        # Test with [M, P] grid points
        kernel_values = kernel(z, mu, sigma)
        assert kernel_values.shape == (2, 5, 10)  # [B, N, M]
        
        # Test with [B, M, P] grid points
        z_batched = z.unsqueeze(0).expand(2, -1, -1)  # [2, 10, 1]
        kernel_values_batched = kernel(z_batched, mu, sigma)
        assert kernel_values_batched.shape == (2, 5, 10)
        
        # Verify results are identical
        assert torch.allclose(kernel_values, kernel_values_batched)
    
    def test_sigma_clamping(self, kernel, test_data):
        """Test that sigma values are properly clamped."""
        z, mu, sigma = test_data
        
        # Test with sigma values outside valid range
        sigma_too_small = torch.full_like(sigma, 0.05)  # Below min_sigma
        sigma_too_large = torch.full_like(sigma, 15.0)   # Above max_sigma
        
        kernel_values_small = kernel(z, mu, sigma_too_small)
        kernel_values_large = kernel(z, mu, sigma_too_large)
        
        # Should not raise errors and should be finite
        assert torch.isfinite(kernel_values_small).all()
        assert torch.isfinite(kernel_values_large).all()
    
    def test_gradient_flow(self, kernel, test_data):
        """Test gradient flow through RBF kernel."""
        z, mu, sigma = test_data
        
        # Make inputs require gradients
        mu.requires_grad_(True)
        sigma.requires_grad_(True)
        
        # Forward pass
        kernel_values = kernel(z, mu, sigma)
        
        # Backward pass
        loss = kernel_values.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert mu.grad is not None
        assert sigma.grad is not None
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(sigma.grad).all()
    
    def test_numerical_stability(self, kernel, test_data):
        """Test numerical stability of RBF kernel."""
        z, mu, sigma = test_data
        
        kernel_values = kernel(z, mu, sigma)
        
        # Check for NaN and inf values
        check_numerical_stability(kernel_values, "RBF kernel values")
        
        # Check that values are in reasonable range [0, 1]
        assert (kernel_values >= 0).all()
        assert (kernel_values <= 1).all()
    
    def test_physical_properties(self, kernel):
        """Test physical properties of RBF kernel."""
        # Test that kernel is symmetric (K(z, μ) = K(μ, z))
        z = torch.tensor([[0.0], [0.5], [1.0]])  # [3, 1]
        mu = torch.tensor([[0.5]])  # [1, 1]
        sigma = torch.tensor([[0.2]])  # [1, 1]
        
        kernel_values = kernel(z, mu, sigma)  # [1, 1, 3]
        
        # Should be maximum at mu (position of token)
        max_idx = torch.argmax(kernel_values.squeeze())
        assert max_idx == 1  # Should be maximum at position 0.5
        
        # Should decay with distance
        distances = torch.abs(z.squeeze() - mu.squeeze())
        kernel_values_flat = kernel_values.squeeze()
        
        # Check that kernel values decrease with distance
        for i in range(len(distances) - 1):
            if distances[i] < distances[i + 1]:
                assert kernel_values_flat[i] >= kernel_values_flat[i + 1]


class TestCompactKernel:
    """Test the compact support kernel."""
    
    @pytest.fixture
    def kernel(self):
        """Create compact kernel for testing."""
        return CompactKernel(pos_dim=1)
    
    @pytest.fixture
    def test_data(self):
        """Create test data for kernel computation."""
        batch_size, num_tokens, num_grid_points = 2, 5, 10
        
        # Grid points: [M, P]
        z = torch.linspace(0, 1, num_grid_points).unsqueeze(-1)  # [10, 1]
        
        # Token positions: [B, N, P]
        mu = torch.rand(batch_size, num_tokens, 1)  # [2, 5, 1]
        
        # Radius parameters: [B, N, 1]
        radius = torch.rand(batch_size, num_tokens, 1) * 0.3 + 0.1  # [2, 5, 1]
        
        return z, mu, radius
    
    def test_initialization(self, kernel):
        """Test kernel initialization."""
        assert kernel.pos_dim == 1
        assert kernel.min_radius == 0.1
        assert kernel.max_radius == 5.0
    
    def test_compact_support(self, kernel, test_data):
        """Test that kernel has compact support (zero outside radius)."""
        z, mu, radius = test_data
        
        kernel_values = kernel(z, mu, radius)
        
        # Check that values are non-negative
        assert (kernel_values >= 0).all()
        
        # Check that values are at most 1
        assert (kernel_values <= 1).all()
    
    def test_gradient_flow(self, kernel, test_data):
        """Test gradient flow through compact kernel."""
        z, mu, radius = test_data
        
        # Make inputs require gradients
        mu.requires_grad_(True)
        radius.requires_grad_(True)
        
        # Forward pass
        kernel_values = kernel(z, mu, radius)
        
        # Backward pass
        loss = kernel_values.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert mu.grad is not None
        assert radius.grad is not None
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(radius.grad).all()
    
    def test_numerical_stability(self, kernel, test_data):
        """Test numerical stability of compact kernel."""
        z, mu, radius = test_data
        
        kernel_values = kernel(z, mu, radius)
        
        # Check for NaN and inf values
        check_numerical_stability(kernel_values, "Compact kernel values")
    
    def test_radius_clamping(self, kernel, test_data):
        """Test that radius values are properly clamped."""
        z, mu, radius = test_data
        
        # Test with radius values outside valid range
        radius_too_small = torch.full_like(radius, 0.05)  # Below min_radius
        radius_too_large = torch.full_like(radius, 10.0)  # Above max_radius
        
        kernel_values_small = kernel(z, mu, radius_too_small)
        kernel_values_large = kernel(z, mu, radius_too_large)
        
        # Should not raise errors and should be finite
        assert torch.isfinite(kernel_values_small).all()
        assert torch.isfinite(kernel_values_large).all()


class TestFourierKernel:
    """Test the Fourier (cosine) kernel."""
    
    @pytest.fixture
    def kernel(self):
        """Create Fourier kernel for testing."""
        return FourierKernel(pos_dim=1)
    
    @pytest.fixture
    def test_data(self):
        """Create test data for kernel computation."""
        batch_size, num_tokens, num_grid_points = 2, 5, 10
        
        # Grid points: [M, P]
        z = torch.linspace(0, 1, num_grid_points).unsqueeze(-1)  # [10, 1]
        
        # Token positions: [B, N, P]
        mu = torch.rand(batch_size, num_tokens, 1)  # [2, 5, 1]
        
        # Frequency parameters: [B, N, 1]
        freq = torch.rand(batch_size, num_tokens, 1) * 5.0 + 0.5  # [2, 5, 1]
        
        return z, mu, freq
    
    def test_initialization(self, kernel):
        """Test kernel initialization."""
        assert kernel.pos_dim == 1
        assert kernel.min_freq == 0.1
        assert kernel.max_freq == 10.0
    
    def test_oscillatory_behavior(self, kernel):
        """Test that Fourier kernel shows oscillatory behavior."""
        z = torch.linspace(0, 2*np.pi, 100).unsqueeze(-1)  # [100, 1]
        mu = torch.tensor([[0.0]])  # [1, 1]
        freq = torch.tensor([[1.0]])  # [1, 1]
        
        kernel_values = kernel(z, mu, freq).squeeze()  # [100]
        
        # Should oscillate between -1 and 1
        assert kernel_values.min() >= -1.0
        assert kernel_values.max() <= 1.0
        
        # Should have oscillatory pattern
        # Check that there are multiple sign changes
        sign_changes = torch.sum(torch.sign(kernel_values[1:]) != torch.sign(kernel_values[:-1]))
        assert sign_changes > 0
    
    def test_gradient_flow(self, kernel, test_data):
        """Test gradient flow through Fourier kernel."""
        z, mu, freq = test_data
        
        # Make inputs require gradients
        mu.requires_grad_(True)
        freq.requires_grad_(True)
        
        # Forward pass
        kernel_values = kernel(z, mu, freq)
        
        # Backward pass
        loss = kernel_values.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert mu.grad is not None
        assert freq.grad is not None
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(freq.grad).all()
    
    def test_numerical_stability(self, kernel, test_data):
        """Test numerical stability of Fourier kernel."""
        z, mu, freq = test_data
        
        kernel_values = kernel(z, mu, freq)
        
        # Check for NaN and inf values
        check_numerical_stability(kernel_values, "Fourier kernel values")
        
        # Check that values are in reasonable range [-1, 1]
        assert (kernel_values >= -1).all()
        assert (kernel_values <= 1).all()


class TestLearnableKernel:
    """Test the learnable kernel."""
    
    @pytest.fixture
    def kernel(self):
        """Create learnable kernel for testing."""
        return LearnableKernel(pos_dim=1, hidden_dim=32)
    
    @pytest.fixture
    def test_data(self):
        """Create test data for kernel computation."""
        batch_size, num_tokens, num_grid_points = 2, 5, 10
        
        # Grid points: [M, P]
        z = torch.linspace(0, 1, num_grid_points).unsqueeze(-1)  # [10, 1]
        
        # Token positions: [B, N, P]
        mu = torch.rand(batch_size, num_tokens, 1)  # [2, 5, 1]
        
        # Learnable parameters: [B, N, hidden_dim]
        params = torch.rand(batch_size, num_tokens, 32)  # [2, 5, 32]
        
        return z, mu, params
    
    def test_initialization(self, kernel):
        """Test kernel initialization."""
        assert kernel.pos_dim == 1
        assert kernel.hidden_dim == 32
        assert isinstance(kernel.distance_net, nn.Sequential)
    
    def test_learnable_parameters(self, kernel, test_data):
        """Test that learnable kernel has trainable parameters."""
        z, mu, params = test_data
        
        # Count parameters
        num_params = sum(p.numel() for p in kernel.parameters())
        assert num_params > 0
        
        # Test forward pass
        kernel_values = kernel(z, mu, params)
        assert kernel_values.shape == (2, 5, 10)
    
    def test_gradient_flow(self, kernel, test_data):
        """Test gradient flow through learnable kernel."""
        z, mu, params = test_data
        
        # Make inputs require gradients
        mu.requires_grad_(True)
        params.requires_grad_(True)
        
        # Forward pass
        kernel_values = kernel(z, mu, params)
        
        # Backward pass
        loss = kernel_values.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert mu.grad is not None
        assert params.grad is not None
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(params.grad).all()
    
    def test_numerical_stability(self, kernel, test_data):
        """Test numerical stability of learnable kernel."""
        z, mu, params = test_data
        
        kernel_values = kernel(z, mu, params)
        
        # Check for NaN and inf values
        check_numerical_stability(kernel_values, "Learnable kernel values")
        
        # Check that values are in reasonable range [0, 1] (due to sigmoid)
        assert (kernel_values >= 0).all()
        assert (kernel_values <= 1).all()


class TestKernelFactory:
    """Test the kernel factory."""
    
    def test_available_kernels(self):
        """Test that factory returns correct available kernels."""
        available = KernelFactory.get_available_kernels()
        expected = ["rbf", "compact", "fourier", "learnable"]
        assert set(available) == set(expected)
    
    def test_create_rbf_kernel(self):
        """Test creating RBF kernel through factory."""
        kernel = KernelFactory.create("rbf", pos_dim=2)
        assert isinstance(kernel, RBFKernel)
        assert kernel.pos_dim == 2
    
    def test_create_compact_kernel(self):
        """Test creating compact kernel through factory."""
        kernel = KernelFactory.create("compact", pos_dim=3)
        assert isinstance(kernel, CompactKernel)
        assert kernel.pos_dim == 3
    
    def test_create_fourier_kernel(self):
        """Test creating Fourier kernel through factory."""
        kernel = KernelFactory.create("fourier", pos_dim=1)
        assert isinstance(kernel, FourierKernel)
        assert kernel.pos_dim == 1
    
    def test_create_learnable_kernel(self):
        """Test creating learnable kernel through factory."""
        kernel = KernelFactory.create("learnable", pos_dim=2, hidden_dim=64)
        assert isinstance(kernel, LearnableKernel)
        assert kernel.pos_dim == 2
        assert kernel.hidden_dim == 64
    
    def test_invalid_kernel_type(self):
        """Test that factory raises error for invalid kernel type."""
        with pytest.raises(ValueError, match="Unknown kernel type"):
            KernelFactory.create("invalid", pos_dim=1)


class TestKernelIntegration:
    """Integration tests for kernel system."""
    
    def test_kernel_compute_influence_matrix(self):
        """Test the compute_influence_matrix method."""
        kernel = RBFKernel(pos_dim=1)
        
        # Test data
        z = torch.linspace(0, 1, 5).unsqueeze(-1)  # [5, 1]
        mu = torch.rand(2, 3, 1)  # [2, 3, 1]
        sigma = torch.rand(2, 3, 1) * 0.5 + 0.1  # [2, 3, 1]
        
        # Compute influence matrix
        influence_matrix = kernel.compute_influence_matrix(z, mu, sigma)
        
        # Check shape
        assert influence_matrix.shape == (2, 3, 5)  # [B, N, M]
        
        # Check that values are finite
        assert torch.isfinite(influence_matrix).all()
    
    def test_kernel_consistency(self):
        """Test that different kernels produce consistent shapes."""
        kernels = [
            RBFKernel(pos_dim=1),
            CompactKernel(pos_dim=1),
            FourierKernel(pos_dim=1),
            LearnableKernel(pos_dim=1, hidden_dim=16)
        ]
        
        # Test data
        z = torch.linspace(0, 1, 10).unsqueeze(-1)  # [10, 1]
        mu = torch.rand(2, 4, 1)  # [2, 4, 1]
        
        for kernel in kernels:
            if isinstance(kernel, LearnableKernel):
                params = torch.rand(2, 4, 16)  # [2, 4, 16]
                kernel_values = kernel(z, mu, params)
            else:
                sigma = torch.rand(2, 4, 1) * 0.5 + 0.1  # [2, 4, 1]
                kernel_values = kernel(z, mu, sigma)
            
            # All kernels should produce same shape
            assert kernel_values.shape == (2, 4, 10)  # [B, N, M]
            assert torch.isfinite(kernel_values).all()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 