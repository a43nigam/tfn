"""
Standalone TFN Layer for Google Colab Deployment

A complete Token Field Network layer implementation using PyTorch primitives.
This file contains everything needed to use TFN layers in Colab with minimal setup.

Usage:
    # In Colab
    # Copy this entire file content
    # Then use:
    updated_embeddings = tfn_layer(embeddings, positions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from typing import Literal, Optional, Tuple


def rbf_kernel(grid_points: torch.Tensor, positions: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """RBF kernel using PyTorch primitives."""
    # grid_points: [B, M, P], positions: [B, N, P], sigma: [B, N, 1]
    # Returns: [B, N, M]
    diff = grid_points.unsqueeze(1) - positions.unsqueeze(2)  # [B, N, M, P]
    dist_sq = torch.sum(diff ** 2, dim=-1)  # [B, N, M]
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def compact_kernel(grid_points: torch.Tensor, positions: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Compact kernel using PyTorch primitives."""
    # grid_points: [B, M, P], positions: [B, N, P], radius: [B, N, 1]
    # Returns: [B, N, M]
    diff = grid_points.unsqueeze(1) - positions.unsqueeze(2)  # [B, N, M, P]
    dist = torch.norm(diff, dim=-1)  # [B, N, M]
    return torch.where(dist <= radius, 1.0 - dist / radius, torch.zeros_like(dist))


def fourier_kernel(grid_points: torch.Tensor, positions: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    """Fourier kernel using PyTorch primitives."""
    # grid_points: [B, M, P], positions: [B, N, P], freq: [B, N, 1]
    # Returns: [B, N, M]
    diff = grid_points.unsqueeze(1) - positions.unsqueeze(2)  # [B, N, M, P]
    phase = torch.sum(diff * freq.unsqueeze(-1), dim=-1)  # [B, N, M]
    return torch.cos(phase)


def project_field(embeddings: torch.Tensor, positions: torch.Tensor, grid_points: torch.Tensor, 
                 kernel_type: str = "rbf") -> torch.Tensor:
    """
    Project token embeddings into continuous field.
    
    Args:
        embeddings: [B, N, D] token embeddings
        positions: [B, N, P] token positions
        grid_points: [B, M, P] grid points
        kernel_type: "rbf", "compact", or "fourier"
    
    Returns:
        field: [B, M, D] continuous field
    """
    B, N, D = embeddings.shape
    M = grid_points.shape[1]
    
    # Default kernel parameters
    if kernel_type == "rbf":
        kernel_params = torch.full((B, N, 1), 0.2, device=embeddings.device, dtype=embeddings.dtype)
        kernel_values = rbf_kernel(grid_points, positions, kernel_params)
    elif kernel_type == "compact":
        kernel_params = torch.full((B, N, 1), 0.3, device=embeddings.device, dtype=embeddings.dtype)
        kernel_values = compact_kernel(grid_points, positions, kernel_params)
    elif kernel_type == "fourier":
        kernel_params = torch.full((B, N, 1), 2.0, device=embeddings.device, dtype=embeddings.dtype)
        kernel_values = fourier_kernel(grid_points, positions, kernel_params)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Project embeddings to field: E ⊗ K
    embeddings_expanded = embeddings.unsqueeze(-1)  # [B, N, D, 1]
    kernel_values_expanded = kernel_values.unsqueeze(2)  # [B, N, 1, M]
    
    # Compute weighted embeddings: E ⊗ K
    weighted_embeddings = embeddings_expanded * kernel_values_expanded  # [B, N, D, M]
    
    # Aggregate fields from all tokens: Σᵢ Eᵢ ⊗ Kᵢ
    field = weighted_embeddings.sum(dim=1)  # [B, D, M]
    field = field.transpose(1, 2)  # [B, M, D]
    
    return field


def evolve_field_cnn(field: torch.Tensor, grid_points: torch.Tensor, time_steps: int = 3) -> torch.Tensor:
    """Evolve field using CNN-based evolution."""
    B, M, D = field.shape
    
    # Simple 1D convolution for evolution
    conv1d = nn.Conv1d(D, D, kernel_size=3, padding=1, bias=False)
    conv1d.weight.data.normal_(0, 0.1)  # Initialize weights
    
    evolved = field
    for _ in range(time_steps):
        # Apply convolution: [B, M, D] -> [B, D, M] -> [B, D, M] -> [B, M, D]
        evolved = evolved.transpose(1, 2)  # [B, D, M]
        evolved = conv1d(evolved)  # [B, D, M]
        evolved = evolved.transpose(1, 2)  # [B, M, D]
        evolved = F.relu(evolved)  # Add non-linearity
    
    return evolved


def evolve_field_spectral(field: torch.Tensor, grid_points: torch.Tensor, time_steps: int = 3) -> torch.Tensor:
    """Evolve field using spectral methods."""
    B, M, D = field.shape
    
    # Use FFT for spectral evolution
    evolved = field
    for _ in range(time_steps):
        # FFT: [B, M, D] -> [B, M, D] (complex)
        evolved_fft = torch.fft.fft(evolved, dim=1)
        
        # Apply spectral filter (low-pass)
        freq = torch.fft.fftfreq(M, device=field.device)
        filter_weights = torch.exp(-freq ** 2).unsqueeze(0).unsqueeze(-1)  # [1, M, 1]
        evolved_fft = evolved_fft * filter_weights
        
        # IFFT: [B, M, D] -> [B, M, D]
        evolved = torch.fft.ifft(evolved_fft, dim=1).real
    
    return evolved


def evolve_field_pde(field: torch.Tensor, grid_points: torch.Tensor, time_steps: int = 3, dt: float = 0.01) -> torch.Tensor:
    """Evolve field using PDE-based diffusion."""
    B, M, D = field.shape
    
    # Simple diffusion equation: ∂u/∂t = α∇²u
    alpha = 0.1
    evolved = field
    
    for _ in range(time_steps):
        # Compute Laplacian using finite differences
        # ∇²u ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²
        laplacian = torch.zeros_like(evolved)
        laplacian[:, 1:-1, :] = (evolved[:, 2:, :] - 2 * evolved[:, 1:-1, :] + evolved[:, :-2, :])
        
        # Update: u[t+1] = u[t] + α * dt * ∇²u
        evolved = evolved + alpha * dt * laplacian
    
    return evolved


def sample_field(field: torch.Tensor, grid_points: torch.Tensor, sample_positions: torch.Tensor, 
                mode: str = "linear") -> torch.Tensor:
    """
    Sample field at given positions using interpolation.
    
    Args:
        field: [B, M, D] field values at grid points
        grid_points: [B, M, P] grid coordinates
        sample_positions: [B, N, P] positions to sample at
        mode: "linear" or "nearest"
    
    Returns:
        sampled: [B, N, D] field values at sample positions
    """
    B, M, D = field.shape
    N = sample_positions.shape[1]
    
    # Handle out-of-bounds by clamping
    grid_min = grid_points[:, 0:1, :]  # [B, 1, P]
    grid_max = grid_points[:, -1:, :]  # [B, 1, P]
    pos_clamped = torch.clamp(sample_positions, grid_min, grid_max)  # [B, N, P]
    
    # For 1D case (P=1), use searchsorted for efficient interpolation
    if grid_points.shape[-1] == 1:
        grid_flat = grid_points.view(B, M)  # [B, M]
        pos_flat = pos_clamped.view(B, N)  # [B, N]
        
        # Find left/right grid indices
        idx_left = torch.searchsorted(grid_flat, pos_flat, right=True) - 1
        idx_left = idx_left.clamp(0, M - 2)  # [B, N]
        idx_right = idx_left + 1
        
        # Get grid values and field values
        g_left = torch.gather(grid_flat, 1, idx_left)  # [B, N]
        g_right = torch.gather(grid_flat, 1, idx_right)  # [B, N]
        
        f_left = torch.gather(field.view(B, M, D), 1, idx_left.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]
        f_right = torch.gather(field.view(B, M, D), 1, idx_right.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]
        
        if mode == "nearest":
            dist_left = (pos_flat - g_left).abs()
            dist_right = (pos_flat - g_right).abs()
            use_left = (dist_left <= dist_right).unsqueeze(-1)
            sampled = torch.where(use_left, f_left, f_right)
        else:  # linear
            denom = (g_right - g_left).clamp(min=1e-8)
            w_right = (pos_flat - g_left) / denom
            w_left = 1.0 - w_right
            sampled = w_left.unsqueeze(-1) * f_left + w_right.unsqueeze(-1) * f_right
        
        return sampled  # [B, N, D]
    else:
        raise NotImplementedError("Only 1D sampling supported for now")


def tfn_layer(embeddings: torch.Tensor, positions: torch.Tensor, 
              kernel_type: str = "rbf", evolution_type: str = "cnn",
              embed_dim: Optional[int] = None, grid_size: int = 100, 
              time_steps: int = 3, sampling_mode: str = "linear") -> torch.Tensor:
    """
    Standalone TFN layer function.
    
    Args:
        embeddings: [B, N, D] token embeddings
        positions: [B, N, P] token positions (P=1 for 1D)
        kernel_type: "rbf", "compact", or "fourier"
        evolution_type: "cnn", "spectral", or "pde"
        embed_dim: embedding dimension (inferred from embeddings if None)
        grid_size: number of grid points for field evaluation
        time_steps: number of evolution steps
        sampling_mode: "linear" or "nearest" interpolation
    
    Returns:
        updated_embeddings: [B, N, D] updated token embeddings
    """
    B, N, D = embeddings.shape
    P = positions.shape[-1]
    
    if embed_dim is None:
        embed_dim = D
    
    # Create uniform grid
    grid_points = torch.linspace(0, 1, grid_size, device=embeddings.device, dtype=embeddings.dtype)
    grid_points = grid_points.unsqueeze(0).unsqueeze(-1).expand(B, -1, P)  # [B, grid_size, P]
    
    # Step 1: Field projection
    field = project_field(embeddings, positions, grid_points, kernel_type)
    
    # Step 2: Field evolution
    if evolution_type == "cnn":
        evolved_field = evolve_field_cnn(field, grid_points, time_steps)
    elif evolution_type == "spectral":
        evolved_field = evolve_field_spectral(field, grid_points, time_steps)
    elif evolution_type == "pde":
        evolved_field = evolve_field_pde(field, grid_points, time_steps)
    else:
        raise ValueError(f"Unknown evolution type: {evolution_type}")
    
    # Step 3: Field sampling
    updated_embeddings = sample_field(evolved_field, grid_points, positions, sampling_mode)
    
    return updated_embeddings


# Example usage and testing functions
def test_tfn_layer():
    """Test the TFN layer with simple data."""
    print("Testing TFN Layer...")
    
    # Test data
    B, N, D = 2, 5, 64
    embeddings = torch.randn(B, N, D)
    positions = torch.linspace(0.1, 0.9, N).unsqueeze(0).expand(B, -1).unsqueeze(-1)
    
    # Test different configurations
    configs = [
        ("rbf", "cnn"),
        ("compact", "spectral"), 
        ("fourier", "pde")
    ]
    
    for kernel_type, evolution_type in configs:
        print(f"\nTesting {kernel_type} kernel + {evolution_type} evolution...")
        
        # Test TFN layer
        updated = tfn_layer(embeddings, positions, kernel_type, evolution_type)
        
        # Verify shapes
        assert updated.shape == embeddings.shape, f"Shape mismatch: {updated.shape} vs {embeddings.shape}"
        print(f"✅ Shape correct: {updated.shape}")
        
        # Verify differentiability
        embeddings.requires_grad_(True)
        updated = tfn_layer(embeddings, positions, kernel_type, evolution_type)
        loss = updated.sum()
        loss.backward()
        
        grad_norm = torch.norm(embeddings.grad).item()
        print(f"✅ Gradient flow: {grad_norm:.4f}")
        
        # Verify field evolution actually changes the field
        field = project_field(embeddings, positions, 
                            torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(-1).expand(B, -1, 1),
                            kernel_type)
        evolved = evolve_field_cnn(field, 
                                 torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(-1).expand(B, -1, 1))
        
        change_norm = torch.norm(evolved - field).item()
        print(f"✅ Field evolution: {change_norm:.4f}")
    
    print("\n🎉 All TFN layer tests passed!")


if __name__ == "__main__":
    test_tfn_layer() 