from __future__ import annotations

"""tfn.datasets.physics_loader
Physics dataset loaders for PDE-Bench datasets.
Tests TFN's natural fit for spatial domains with grid evolution.
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# Optional dependency ---------------------------------------------------------
try:
    from datasets import load_dataset  # type: ignore
    _HAVE_HF = True
except ImportError:
    _HAVE_HF = False


class BurgersDataset(Dataset):
    """Burgers equation dataset for 1D PDE evolution."""
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 100,
        input_steps: int = 10,
        output_steps: int = 40,
        normalize: bool = True
    ):
        """
        Args:
            data: Array of shape [num_samples, time_steps, grid_points]
            seq_len: Length of input sequence
            input_steps: Number of input time steps
            output_steps: Number of output time steps to predict
            normalize: Whether to normalize the data
        """
        self.data = data
        self.seq_len = seq_len
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize
        
        # Normalize data
        if normalize:
            self.data_mean = np.mean(data)
            self.data_std = np.std(data)
            self.data = (data - self.data_mean) / self.data_std
        
        # Create samples
        self.samples = []
        for i in range(len(data)):
            for t in range(input_steps, len(data[i]) - output_steps + 1):
                self.samples.append((i, t))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input and target sequences."""
        sample_idx, time_idx = self.samples[idx]
        
        # Input sequence: [input_steps, grid_points]
        input_seq = self.data[sample_idx, time_idx - self.input_steps:time_idx]
        
        # Target sequence: [output_steps, grid_points]
        target_seq = self.data[sample_idx, time_idx:time_idx + self.output_steps]
        
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


class WaveDataset(Dataset):
    """Wave equation dataset for 1D PDE evolution."""
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 100,
        input_steps: int = 10,
        output_steps: int = 40,
        normalize: bool = True
    ):
        """
        Args:
            data: Array of shape [num_samples, time_steps, grid_points]
            seq_len: Length of input sequence
            input_steps: Number of input time steps
            output_steps: Number of output time steps to predict
            normalize: Whether to normalize the data
        """
        self.data = data
        self.seq_len = seq_len
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize
        
        # Normalize data
        if normalize:
            self.data_mean = np.mean(data)
            self.data_std = np.std(data)
            self.data = (data - self.data_mean) / self.data_std
        
        # Create samples
        self.samples = []
        for i in range(len(data)):
            for t in range(input_steps, len(data[i]) - output_steps + 1):
                self.samples.append((i, t))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input and target sequences."""
        sample_idx, time_idx = self.samples[idx]
        
        # Input sequence: [input_steps, grid_points]
        input_seq = self.data[sample_idx, time_idx - self.input_steps:time_idx]
        
        # Target sequence: [output_steps, grid_points]
        target_seq = self.data[sample_idx, time_idx:time_idx + self.output_steps]
        
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


class HeatDataset(Dataset):
    """Heat equation dataset for 1D PDE evolution."""
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 100,
        input_steps: int = 10,
        output_steps: int = 40,
        normalize: bool = True
    ):
        """
        Args:
            data: Array of shape [num_samples, time_steps, grid_points]
            seq_len: Length of input sequence
            input_steps: Number of input time steps
            output_steps: Number of output time steps to predict
            normalize: Whether to normalize the data
        """
        self.data = data
        self.seq_len = seq_len
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize
        
        # Normalize data
        if normalize:
            self.data_mean = np.mean(data)
            self.data_std = np.std(data)
            self.data = (data - self.data_mean) / self.data_std
        
        # Create samples
        self.samples = []
        for i in range(len(data)):
            for t in range(input_steps, len(data[i]) - output_steps + 1):
                self.samples.append((i, t))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input and target sequences."""
        sample_idx, time_idx = self.samples[idx]
        
        # Input sequence: [input_steps, grid_points]
        input_seq = self.data[sample_idx, time_idx - self.input_steps:time_idx]
        
        # Target sequence: [output_steps, grid_points]
        target_seq = self.data[sample_idx, time_idx:time_idx + self.output_steps]
        
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


def generate_burgers_data(
    num_samples: int = 1000,
    grid_points: int = 128,
    time_steps: int = 100,
    dt: float = 0.001,  # Reduced time step for stability
    dx: float = 1.0 / 128,
    nu: float = 0.1  # Increased viscosity for stability
) -> np.ndarray:
    """Generate synthetic Burgers equation data.
    
    Args:
        num_samples: Number of samples to generate
        grid_points: Number of spatial grid points
        time_steps: Number of time steps
        dt: Time step size
        dx: Spatial step size
        nu: Viscosity parameter
    
    Returns:
        Array of shape [num_samples, time_steps, grid_points]
    """
    data = np.zeros((num_samples, time_steps, grid_points))
    
    for i in range(num_samples):
        # Initial condition: random wave
        u = np.sin(2 * np.pi * np.arange(grid_points) / grid_points)
        u += 0.1 * np.random.randn(grid_points)
        
        # Time evolution using finite difference
        for t in range(time_steps):
            data[i, t] = u
            
            # Burgers equation: du/dt + u * du/dx = nu * d²u/dx²
            u_new = u.copy()
            
            # Convection term: u * du/dx (upwind scheme with clipping)
            du_dx = np.zeros_like(u)
            du_dx[1:] = (u[1:] - u[:-1]) / dx
            du_dx[0] = (u[1] - u[-1]) / dx  # Periodic boundary
            # Clip gradients to prevent overflow
            du_dx = np.clip(du_dx, -10.0, 10.0)
            
            # Diffusion term: nu * d²u/dx²
            d2u_dx2 = np.zeros_like(u)
            d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2)
            d2u_dx2[0] = (u[1] - 2 * u[0] + u[-1]) / (dx ** 2)
            d2u_dx2[-1] = (u[0] - 2 * u[-1] + u[-2]) / (dx ** 2)
            # Clip second derivatives to prevent overflow
            d2u_dx2 = np.clip(d2u_dx2, -100.0, 100.0)
            
            # Update with clipping
            u_new = u - dt * (u * du_dx - nu * d2u_dx2)
            # Clip solution to prevent explosion
            u_new = np.clip(u_new, -5.0, 5.0)
            
            # Periodic boundary conditions
            u_new[0] = u_new[-1]
            u_new[-1] = u_new[0]
            
            u = u_new
    
    return data


def generate_wave_data(
    num_samples: int = 1000,
    grid_points: int = 128,
    time_steps: int = 100,
    dt: float = 0.01,
    dx: float = 1.0 / 128,
    c: float = 1.0
) -> np.ndarray:
    """Generate synthetic wave equation data.
    
    Args:
        num_samples: Number of samples to generate
        grid_points: Number of spatial grid points
        time_steps: Number of time steps
        dt: Time step size
        dx: Spatial step size
        c: Wave speed
    
    Returns:
        Array of shape [num_samples, time_steps, grid_points]
    """
    data = np.zeros((num_samples, time_steps, grid_points))
    
    for i in range(num_samples):
        # Initial condition: Gaussian pulse
        x = np.linspace(0, 1, grid_points)
        u = np.exp(-((x - 0.5) ** 2) / 0.01)
        v = np.zeros_like(u)  # Initial velocity
        
        # Time evolution using finite difference
        for t in range(time_steps):
            data[i, t] = u
            
            # Wave equation: d²u/dt² = c² * d²u/dx²
            u_new = u.copy()
            v_new = v.copy()
            
            # Spatial derivatives
            d2u_dx2 = np.zeros_like(u)
            d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2)
            d2u_dx2[0] = (u[1] - 2 * u[0] + u[-1]) / (dx ** 2)
            d2u_dx2[-1] = (u[0] - 2 * u[-1] + u[-2]) / (dx ** 2)
            
            # Update using velocity formulation
            v_new = v + dt * (c ** 2) * d2u_dx2
            u_new = u + dt * v_new
            
            # Periodic boundary conditions
            u_new[0] = u_new[-1]
            u_new[-1] = u_new[0]
            v_new[0] = v_new[-1]
            v_new[-1] = v_new[0]
            
            u = u_new
            v = v_new
    
    return data


def generate_heat_data(
    num_samples: int = 1000,
    grid_points: int = 128,
    time_steps: int = 100,
    dt: float = 0.001,
    dx: float = 1.0 / 128,
    alpha: float = 0.1
) -> np.ndarray:
    """Generate synthetic heat equation data.
    
    Args:
        num_samples: Number of samples to generate
        grid_points: Number of spatial grid points
        time_steps: Number of time steps
        dt: Time step size
        dx: Spatial step size
        alpha: Thermal diffusivity
    
    Returns:
        Array of shape [num_samples, time_steps, grid_points]
    """
    data = np.zeros((num_samples, time_steps, grid_points))
    
    for i in range(num_samples):
        # Initial condition: random temperature distribution
        u = np.random.randn(grid_points)
        u = np.sin(2 * np.pi * np.arange(grid_points) / grid_points) + 0.1 * u
        
        # Time evolution using finite difference
        for t in range(time_steps):
            data[i, t] = u
            
            # Heat equation: du/dt = alpha * d²u/dx²
            u_new = u.copy()
            
            # Spatial derivatives
            d2u_dx2 = np.zeros_like(u)
            d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2)
            d2u_dx2[0] = (u[1] - 2 * u[0] + u[-1]) / (dx ** 2)
            d2u_dx2[-1] = (u[0] - 2 * u[-1] + u[-2]) / (dx ** 2)
            
            # Update
            u_new = u + dt * alpha * d2u_dx2
            
            # Periodic boundary conditions
            u_new[0] = u_new[-1]
            u_new[-1] = u_new[0]
            
            u = u_new
    
    return data


def load_physics_dataset(
    dataset_type: str = "burgers",
    num_samples: int = 1000,
    grid_points: int = 128,
    time_steps: int = 100,
    input_steps: int = 10,
    output_steps: int = 40,
    train_split: float = 0.8,
    normalize: bool = True,
    **kwargs
) -> Tuple[Dataset, Dataset]:
    """Load physics dataset for PDE evolution.
    
    Args:
        dataset_type: Type of PDE ("burgers", "wave", "heat")
        num_samples: Number of samples to generate
        grid_points: Number of spatial grid points
        time_steps: Number of time steps
        input_steps: Number of input time steps
        output_steps: Number of output time steps to predict
        train_split: Fraction of data for training
        normalize: Whether to normalize the data
        **kwargs: Additional parameters for data generation
    
    Returns:
        (train_dataset, val_dataset)
    """
    
    # Generate data
    if dataset_type == "burgers":
        data = generate_burgers_data(num_samples, grid_points, time_steps, **kwargs)
        dataset_class = BurgersDataset
    elif dataset_type == "wave":
        data = generate_wave_data(num_samples, grid_points, time_steps, **kwargs)
        dataset_class = WaveDataset
    elif dataset_type == "heat":
        data = generate_heat_data(num_samples, grid_points, time_steps, **kwargs)
        dataset_class = HeatDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Split data
    train_size = int(num_samples * train_split)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Create datasets
    train_dataset = dataset_class(
        train_data, seq_len=grid_points, input_steps=input_steps, 
        output_steps=output_steps, normalize=normalize
    )
    
    val_dataset = dataset_class(
        val_data, seq_len=grid_points, input_steps=input_steps, 
        output_steps=output_steps, normalize=normalize
    )
    
    return train_dataset, val_dataset


def create_physics_dataloader(
    dataset_type: str = "burgers",
    batch_size: int = 8,
    num_samples: int = 1000,
    grid_points: int = 128,
    time_steps: int = 100,
    input_steps: int = 10,
    output_steps: int = 40,
    train_split: float = 0.8,
    normalize: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create physics dataloaders for training.
    
    Returns:
        (train_loader, val_loader)
    """
    
    train_dataset, val_dataset = load_physics_dataset(
        dataset_type=dataset_type,
        num_samples=num_samples,
        grid_points=grid_points,
        time_steps=time_steps,
        input_steps=input_steps,
        output_steps=output_steps,
        train_split=train_split,
        normalize=normalize,
        **kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    return train_loader, val_loader


def compute_pde_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> Dict[str, float]:
    """Compute metrics for PDE prediction.
    
    Args:
        predictions: Predicted values [batch, time, space]
        targets: Target values [batch, time, space]
        reduction: Reduction method ("mean", "sum")
    
    Returns:
        Dictionary of metrics
    """
    
    # MSE
    mse = torch.mean((predictions - targets) ** 2)
    
    # MAE
    mae = torch.mean(torch.abs(predictions - targets))
    
    # Relative L2 error
    relative_l2 = torch.sqrt(torch.mean((predictions - targets) ** 2)) / torch.sqrt(torch.mean(targets ** 2))
    
    # Maximum error
    max_error = torch.max(torch.abs(predictions - targets))
    
    return {
        "mse": mse.item(),
        "mae": mae.item(),
        "relative_l2": relative_l2.item(),
        "max_error": max_error.item()
    }


def visualize_pde_prediction(
    input_seq: torch.Tensor,
    target_seq: torch.Tensor,
    pred_seq: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """Visualize PDE prediction results.
    
    Args:
        input_seq: Input sequence [time, space]
        target_seq: Target sequence [time, space]
        pred_seq: Predicted sequence [time, space]
        save_path: Path to save plot (optional)
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input sequence
    im1 = axes[0].imshow(input_seq.T, aspect='auto', cmap='viridis')
    axes[0].set_title('Input Sequence')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Space')
    plt.colorbar(im1, ax=axes[0])
    
    # Target sequence
    im2 = axes[1].imshow(target_seq.T, aspect='auto', cmap='viridis')
    axes[1].set_title('Target Sequence')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Space')
    plt.colorbar(im2, ax=axes[1])
    
    # Predicted sequence
    im3 = axes[2].imshow(pred_seq.T, aspect='auto', cmap='viridis')
    axes[2].set_title('Predicted Sequence')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Space')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test physics dataset generation
    print("Testing physics dataset generation...")
    
    # Generate Burgers data
    burgers_data = generate_burgers_data(num_samples=10, grid_points=64, time_steps=50)
    print(f"Burgers data shape: {burgers_data.shape}")
    
    # Create dataset
    train_dataset, val_dataset = load_physics_dataset(
        dataset_type="burgers",
        num_samples=100,
        grid_points=64,
        time_steps=50,
        input_steps=5,
        output_steps=10
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Test a sample
    input_seq, target_seq = train_dataset[0]
    print(f"Input shape: {input_seq.shape}")
    print(f"Target shape: {target_seq.shape}")
    
    # Test dataloader
    train_loader, val_loader = create_physics_dataloader(
        dataset_type="burgers",
        batch_size=4,
        num_samples=50,
        grid_points=64,
        time_steps=50
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        input_batch, target_batch = batch
        print(f"Input batch shape: {input_batch.shape}")
        print(f"Target batch shape: {target_batch.shape}")
        break 