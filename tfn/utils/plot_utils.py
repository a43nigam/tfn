"""
Plotting utilities for TFN models.

Visualization tools for Token Field Networks, including training curves,
field visualizations, and model comparisons.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import seaborn as sns


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                        train_metrics: Optional[List[float]] = None,
                        val_metrics: Optional[List[float]] = None,
                        metric_name: str = "Accuracy",
                        save_path: Optional[str] = None) -> None:
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2 if train_metrics else 1, figsize=(12, 4))
    
    if train_metrics:
        # Loss plot
        axes[0].plot(train_losses, label='Train Loss', color='blue')
        axes[0].plot(val_losses, label='Val Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metric plot
        axes[1].plot(train_metrics, label=f'Train {metric_name}', color='blue')
        axes[1].plot(val_metrics, label=f'Val {metric_name}', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'Training and Validation {metric_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Only loss plot
        axes.plot(train_losses, label='Train Loss', color='blue')
        axes.plot(val_losses, label='Val Loss', color='red')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.set_title('Training and Validation Loss')
        axes.legend()
        axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_field_evolution(field: torch.Tensor, grid_points: torch.Tensor,
                        time_steps: List[int] = None, 
                        save_path: Optional[str] = None) -> None:
    """Plot field evolution over time."""
    if time_steps is None:
        time_steps = list(range(field.shape[0]))
    
    fig, axes = plt.subplots(1, len(time_steps), figsize=(4*len(time_steps), 4))
    
    if len(time_steps) == 1:
        axes = [axes]
    
    for i, t in enumerate(time_steps):
        field_t = field[t].detach().cpu().numpy()
        grid_t = grid_points[t].detach().cpu().numpy()
        
        # Plot field magnitude
        field_magnitude = np.linalg.norm(field_t, axis=1)
        axes[i].plot(grid_t.flatten(), field_magnitude, 'b-', linewidth=2)
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Field Magnitude')
        axes[i].set_title(f'Field at t={t}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_kernel_comparison(kernels: Dict[str, torch.Tensor], positions: torch.Tensor,
                          grid_points: torch.Tensor, save_path: Optional[str] = None) -> None:
    """Plot different kernel types for comparison."""
    fig, axes = plt.subplots(1, len(kernels), figsize=(4*len(kernels), 4))
    
    if len(kernels) == 1:
        axes = [axes]
    
    for i, (kernel_name, kernel_values) in enumerate(kernels.items()):
        # Plot kernel values for a single token position
        token_idx = 0
        kernel_slice = kernel_values[0, token_idx].detach().cpu().numpy()
        grid_slice = grid_points[0, :, 0].detach().cpu().numpy()
        
        axes[i].plot(grid_slice, kernel_slice, 'r-', linewidth=2)
        axes[i].set_xlabel('Grid Position')
        axes[i].set_ylabel('Kernel Value')
        axes[i].set_title(f'{kernel_name} Kernel')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(model_results: Dict[str, Dict[str, float]],
                         metric: str = "accuracy", save_path: Optional[str] = None) -> None:
    """Plot model comparison bar chart."""
    model_names = list(model_results.keys())
    metric_values = [model_results[name].get(metric, 0) for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel(metric.title())
    plt.title(f'Model Comparison - {metric.title()}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attention_heatmap(attention_weights: torch.Tensor, 
                          save_path: Optional[str] = None) -> None:
    """Plot attention heatmap for visualization."""
    attention_np = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_np, cmap='viridis', annot=False, cbar=True)
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_parameter_distributions(model: torch.nn.Module, 
                               save_path: Optional[str] = None) -> None:
    """Plot parameter distributions for model analysis."""
    param_names = []
    param_values = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_names.append(name)
            param_values.append(param.data.flatten().detach().cpu().numpy())
    
    n_params = len(param_names)
    cols = 3
    rows = (n_params + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (name, values) in enumerate(zip(param_names, param_values)):
        row = i // cols
        col = i % cols
        
        axes[row, col].hist(values, bins=50, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[row, col].set_title(f'{name}\n(μ={values.mean():.4f}, σ={values.std():.4f})')
        axes[row, col].set_xlabel('Parameter Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_params, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_gradient_flow(model: torch.nn.Module, loss: torch.Tensor,
                      save_path: Optional[str] = None) -> None:
    """Plot gradient flow through the model."""
    # Compute gradients
    loss.backward()
    
    layer_names = []
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            grad_norms.append(param.grad.norm().item())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(layer_names)), grad_norms, color='lightcoral', edgecolor='darkred')
    
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow Through Model')
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, grad_norms):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_training_dashboard(train_losses: List[float], val_losses: List[float],
                            train_metrics: List[float], val_metrics: List[float],
                            save_path: Optional[str] = None) -> None:
    """Create a comprehensive training dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Metric curves
    axes[0, 1].plot(train_metrics, label='Train Metric', color='blue')
    axes[0, 1].plot(val_metrics, label='Val Metric', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Metric')
    axes[0, 1].set_title('Training and Validation Metric')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss difference
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    axes[1, 0].plot(loss_diff, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('|Train Loss - Val Loss|')
    axes[1, 0].set_title('Loss Gap')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metric difference
    metric_diff = [abs(t - v) for t, v in zip(train_metrics, val_metrics)]
    axes[1, 1].plot(metric_diff, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('|Train Metric - Val Metric|')
    axes[1, 1].set_title('Metric Gap')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
