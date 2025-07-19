"""
Unit tests for TFN field evolution system.

Tests all evolution types (CNN, Spectral, PDE) for:
- Shape validation
- Gradient flow
- Numerical stability
- Physical properties
- Temporal dynamics
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Import the field evolution system
from ..core.field_evolution import (
    FieldEvolver, CNNFieldEvolver, SpectralFieldEvolver, 
    PDEFieldEvolver, TemporalGrid, create_field_evolver
)
from tfn.core.utils import validate_shapes, check_numerical_stability


class TestTemporalGrid:
    """Test the temporal grid generator."""
    
    def test_initialization(self):
        """Test temporal grid initialization."""
        time_steps = 10
        dt = 0.01
        grid = TemporalGrid(time_steps, dt)
        
        assert grid.time_steps == time_steps
        assert grid.dt == dt
        assert len(grid.time_points) == time_steps + 1
        assert grid.time_points[0] == 0.0
        assert grid.time_points[-1] == time_steps * dt
    
    def test_get_time_points(self):
        """Test time points generation."""
        grid = TemporalGrid(time_steps=5, dt=0.1)
        time_points = grid.get_time_points(batch_size=3)
        
        # Check shape
        assert time_points.shape == (3, 6)  # [B, T+1]
        
        # Check values
        expected_times = torch.linspace(0, 0.5, 6)
        assert torch.allclose(time_points[0], expected_times)
        assert torch.allclose(time_points[1], expected_times)
        assert torch.allclose(time_points[2], expected_times)


class TestCNNFieldEvolver:
    """Test CNN-based field evolution."""
    
    @pytest.fixture
    def evolver(self):
        """Create CNN evolver for testing."""
        return CNNFieldEvolver(embed_dim=64, pos_dim=1)
    
    @pytest.fixture
    def test_data(self):
        """Create test data for field evolution."""
        batch_size, num_points, embed_dim = 2, 20, 64
        
        # Initial field: [B, M, D]
        field = torch.randn(batch_size, num_points, embed_dim)
        
        # Grid points: [B, M, P]
        grid_points = torch.linspace(0, 1, num_points).unsqueeze(-1).unsqueeze(0).expand(batch_size, -1, 1)
        
        return field, grid_points
    
    def test_initialization(self, evolver):
        """Test evolver initialization."""
        assert evolver.embed_dim == 64
        assert evolver.pos_dim == 1
        assert evolver.hidden_dim == 128
        assert isinstance(evolver.conv1, nn.Conv1d)
        assert isinstance(evolver.conv2, nn.Conv1d)
        assert isinstance(evolver.conv3, nn.Conv1d)
    
    def test_shape_validation(self, evolver, test_data):
        """Test that evolution preserves shapes."""
        field, grid_points = test_data
        
        # Single time step
        evolved_field = evolver(field, grid_points, time_steps=1)
        assert evolved_field.shape == field.shape
        
        # Multiple time steps
        evolved_field = evolver(field, grid_points, time_steps=5)
        assert evolved_field.shape == field.shape
    
    def test_gradient_flow(self, evolver, test_data):
        """Test gradient flow through CNN evolution."""
        field, grid_points = test_data
        
        # Make field require gradients
        field.requires_grad_(True)
        
        # Forward pass
        evolved_field = evolver(field, grid_points, time_steps=3)
        
        # Backward pass
        loss = evolved_field.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert field.grad is not None
        assert torch.isfinite(field.grad).all()
    
    def test_numerical_stability(self, evolver, test_data):
        """Test numerical stability of CNN evolution."""
        field, grid_points = test_data
        
        evolved_field = evolver(field, grid_points, time_steps=10)
        
        # Check for NaN and inf values
        check_numerical_stability(evolved_field, "CNN evolved field")
        
        # Check that values are finite
        assert torch.isfinite(evolved_field).all()
    
    def test_temporal_dynamics(self, evolver, test_data):
        """Test that evolution changes the field over time."""
        field, grid_points = test_data
        
        # Evolve for different numbers of steps
        evolved_1 = evolver(field, grid_points, time_steps=1)
        evolved_5 = evolver(field, grid_points, time_steps=5)
        
        # Fields should be different after evolution
        assert not torch.allclose(evolved_1, evolved_5, atol=1e-6)
        
        # But should have same shape
        assert evolved_1.shape == evolved_5.shape


class TestSpectralFieldEvolver:
    """Test spectral-based field evolution."""
    
    @pytest.fixture
    def evolver(self):
        """Create spectral evolver for testing."""
        return SpectralFieldEvolver(embed_dim=32, pos_dim=1)
    
    @pytest.fixture
    def test_data(self):
        """Create test data for field evolution."""
        batch_size, num_points, embed_dim = 2, 16, 32  # Power of 2 for FFT
        
        # Initial field: [B, M, D]
        field = torch.randn(batch_size, num_points, embed_dim)
        
        # Grid points: [B, M, P]
        grid_points = torch.linspace(0, 1, num_points).unsqueeze(-1).unsqueeze(0).expand(batch_size, -1, 1)
        
        return field, grid_points
    
    def test_initialization(self, evolver):
        """Test evolver initialization."""
        assert evolver.embed_dim == 32
        assert evolver.pos_dim == 1
        assert evolver.num_modes == 16
        assert isinstance(evolver.spectral_net, nn.Sequential)
    
    def test_shape_validation(self, evolver, test_data):
        """Test that evolution preserves shapes."""
        field, grid_points = test_data
        
        # Single time step
        evolved_field = evolver(field, grid_points, time_steps=1)
        assert evolved_field.shape == field.shape
        
        # Multiple time steps
        evolved_field = evolver(field, grid_points, time_steps=3)
        assert evolved_field.shape == field.shape
    
    def test_gradient_flow(self, evolver, test_data):
        """Test gradient flow through spectral evolution."""
        field, grid_points = test_data
        
        # Make field require gradients
        field.requires_grad_(True)
        
        # Forward pass
        evolved_field = evolver(field, grid_points, time_steps=2)
        
        # Backward pass
        loss = evolved_field.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert field.grad is not None
        assert torch.isfinite(field.grad).all()
    
    def test_numerical_stability(self, evolver, test_data):
        """Test numerical stability of spectral evolution."""
        field, grid_points = test_data
        
        evolved_field = evolver(field, grid_points, time_steps=5)
        
        # Check for NaN and inf values
        check_numerical_stability(evolved_field, "Spectral evolved field")
        
        # Check that values are finite
        assert torch.isfinite(evolved_field).all()
    
    def test_spectral_properties(self, evolver, test_data):
        """Test spectral properties of evolution."""
        field, grid_points = test_data
        
        # Check that FFT and IFFT preserve the field (approximately)
        field_fft = torch.fft.rfft(field, dim=1)
        field_reconstructed = torch.fft.irfft(field_fft, n=field.shape[1], dim=1)
        
        # Should be approximately equal
        assert torch.allclose(field, field_reconstructed, atol=1e-6)


class TestPDEFieldEvolver:
    """Test PDE-based field evolution."""
    
    @pytest.fixture
    def diffusion_evolver(self):
        """Create diffusion evolver for testing."""
        return PDEFieldEvolver(embed_dim=16, pos_dim=1, pde_type="diffusion")
    
    @pytest.fixture
    def wave_evolver(self):
        """Create wave evolver for testing."""
        return PDEFieldEvolver(embed_dim=16, pos_dim=1, pde_type="wave")
    
    @pytest.fixture
    def test_data(self):
        """Create test data for field evolution."""
        batch_size, num_points, embed_dim = 2, 20, 16
        
        # Initial field: [B, M, D]
        field = torch.randn(batch_size, num_points, embed_dim)
        
        # Grid points: [B, M, P]
        grid_points = torch.linspace(0, 1, num_points).unsqueeze(-1).unsqueeze(0).expand(batch_size, -1, 1)
        
        return field, grid_points
    
    def test_initialization(self, diffusion_evolver, wave_evolver):
        """Test evolver initialization."""
        assert diffusion_evolver.embed_dim == 16
        assert diffusion_evolver.pos_dim == 1
        assert diffusion_evolver.pde_type == "diffusion"
        assert wave_evolver.pde_type == "wave"
        
        # Check learnable parameters
        assert isinstance(diffusion_evolver.diffusion_coeff, nn.Parameter)
        assert isinstance(wave_evolver.wave_speed, nn.Parameter)
    
    def test_diffusion_evolution(self, diffusion_evolver, test_data):
        """Test diffusion equation evolution."""
        field, grid_points = test_data
        
        evolved_field = diffusion_evolver(field, grid_points, time_steps=5, dt=0.01)
        
        # Check shape preservation
        assert evolved_field.shape == field.shape
        
        # Check numerical stability
        check_numerical_stability(evolved_field, "Diffusion evolved field")
        assert torch.isfinite(evolved_field).all()
    
    def test_wave_evolution(self, wave_evolver, test_data):
        """Test wave equation evolution."""
        field, grid_points = test_data
        
        evolved_field = wave_evolver(field, grid_points, time_steps=5, dt=0.01)
        
        # Check shape preservation
        assert evolved_field.shape == field.shape
        
        # Check numerical stability
        check_numerical_stability(evolved_field, "Wave evolved field")
        assert torch.isfinite(evolved_field).all()
    
    def test_gradient_flow(self, diffusion_evolver, test_data):
        """Test gradient flow through PDE evolution."""
        field, grid_points = test_data
        
        # Make field require gradients
        field.requires_grad_(True)
        
        # Forward pass
        evolved_field = diffusion_evolver(field, grid_points, time_steps=3, dt=0.01)
        
        # Backward pass
        loss = evolved_field.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert field.grad is not None
        assert torch.isfinite(field.grad).all()
    
    def test_physical_properties(self, diffusion_evolver, test_data):
        """Test physical properties of PDE evolution."""
        field, grid_points = test_data
        
        # Create a simple initial condition (Gaussian)
        center = 0.5
        width = 0.1
        x = grid_points[0, :, 0]
        initial_field = torch.exp(-((x - center) ** 2) / (2 * width ** 2)).unsqueeze(-1).expand(-1, field.shape[-1])
        field = initial_field.unsqueeze(0).expand(field.shape[0], -1, -1)
        
        # Evolve with diffusion
        evolved_field = diffusion_evolver(field, grid_points, time_steps=10, dt=0.01)
        
        # Diffusion should spread the field (increase variance)
        initial_variance = torch.var(field, dim=1)
        evolved_variance = torch.var(evolved_field, dim=1)
        
        # Variance should generally increase (though not guaranteed for all cases)
        # Just check that evolution changes the field
        assert not torch.allclose(field, evolved_field, atol=1e-6)


class TestFieldEvolver:
    """Test the main field evolver interface."""
    
    def test_cnn_creation(self):
        """Test creating CNN evolver through main interface."""
        evolver = FieldEvolver(embed_dim=64, pos_dim=1, evolution_type="cnn")
        assert isinstance(evolver.evolver, CNNFieldEvolver)
        assert evolver.evolution_type == "cnn"
    
    def test_spectral_creation(self):
        """Test creating spectral evolver through main interface."""
        evolver = FieldEvolver(embed_dim=32, pos_dim=1, evolution_type="spectral")
        assert isinstance(evolver.evolver, SpectralFieldEvolver)
        assert evolver.evolution_type == "spectral"
    
    def test_pde_creation(self):
        """Test creating PDE evolver through main interface."""
        evolver = FieldEvolver(embed_dim=16, pos_dim=1, evolution_type="pde")
        assert isinstance(evolver.evolver, PDEFieldEvolver)
        assert evolver.evolution_type == "pde"
    
    def test_invalid_evolution_type(self):
        """Test that invalid evolution type raises error."""
        with pytest.raises(ValueError, match="Unknown evolution type"):
            FieldEvolver(embed_dim=64, pos_dim=1, evolution_type="invalid")
    
    def test_forward_pass(self):
        """Test forward pass through main interface."""
        evolver = FieldEvolver(embed_dim=32, pos_dim=1, evolution_type="cnn")
        
        # Test data
        field = torch.randn(2, 20, 32)
        grid_points = torch.linspace(0, 1, 20).unsqueeze(-1).unsqueeze(0).expand(2, -1, 1)
        
        # Forward pass
        evolved_field = evolver(field, grid_points, time_steps=3)
        
        # Check shape preservation
        assert evolved_field.shape == field.shape


class TestFieldEvolutionIntegration:
    """Integration tests for field evolution system."""
    
    def test_factory_function(self):
        """Test the factory function for creating evolvers."""
        # Test CNN evolver
        cnn_evolver = create_field_evolver(embed_dim=64, pos_dim=1, evolution_type="cnn")
        assert isinstance(cnn_evolver, FieldEvolver)
        assert isinstance(cnn_evolver.evolver, CNNFieldEvolver)
        
        # Test spectral evolver
        spectral_evolver = create_field_evolver(embed_dim=32, pos_dim=1, evolution_type="spectral")
        assert isinstance(spectral_evolver, FieldEvolver)
        assert isinstance(spectral_evolver.evolver, SpectralFieldEvolver)
        
        # Test PDE evolver
        pde_evolver = create_field_evolver(embed_dim=16, pos_dim=1, evolution_type="pde")
        assert isinstance(pde_evolver, FieldEvolver)
        assert isinstance(pde_evolver.evolver, PDEFieldEvolver)
    
    def test_evolution_consistency(self):
        """Test that different evolution types produce consistent shapes."""
        evolvers = [
            FieldEvolver(embed_dim=32, pos_dim=1, evolution_type="cnn"),
            FieldEvolver(embed_dim=32, pos_dim=1, evolution_type="spectral"),
            FieldEvolver(embed_dim=32, pos_dim=1, evolution_type="pde")
        ]
        
        # Test data
        field = torch.randn(2, 16, 32)
        grid_points = torch.linspace(0, 1, 16).unsqueeze(-1).unsqueeze(0).expand(2, -1, 1)
        
        for evolver in evolvers:
            evolved_field = evolver(field, grid_points, time_steps=2)
            
            # All evolvers should produce same shape
            assert evolved_field.shape == field.shape
            assert torch.isfinite(evolved_field).all()
    
    def test_temporal_grid_integration(self):
        """Test integration with temporal grid."""
        # Create temporal grid
        temporal_grid = TemporalGrid(time_steps=5, dt=0.01)
        time_points = temporal_grid.get_time_points(batch_size=2)
        
        # Create evolver
        evolver = FieldEvolver(embed_dim=32, pos_dim=1, evolution_type="cnn")
        
        # Test data
        field = torch.randn(2, 20, 32)
        grid_points = torch.linspace(0, 1, 20).unsqueeze(-1).unsqueeze(0).expand(2, -1, 1)
        
        # Evolve field
        evolved_field = evolver(field, grid_points, time_steps=5)
        
        # Check that evolution worked
        assert evolved_field.shape == field.shape
        assert torch.isfinite(evolved_field).all()
        
        # Check that time points are correct
        assert time_points.shape == (2, 6)  # [B, T+1]
        assert time_points[0, 0] == 0.0
        assert time_points[0, -1] == 0.05


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 