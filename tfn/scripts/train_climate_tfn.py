#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Climate Time Series Training CLI for Token Field Networks (TFN)

This standalone script makes it easy to train and evaluate a TFN regressor
on climate and time series datasets directly from the command line.

Example usages
--------------
# Train on Electricity Transformer Temperature
python -m tfn.scripts.train_climate_tfn --dataset electricity --epochs 10

# Train on Jena Climate with custom parameters
python -m tfn.scripts.train_climate_tfn --dataset jena --embed_dim 128 --num_layers 3 \
       --batch_size 64 --epochs 20 --lr 3e-4

The script will automatically detect GPU availability, save checkpoints, and
print final MSE/RMSE metrics.
"""

# Standard library imports
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Local imports
from tfn.model.tfn_classifiers import TFNClassifier
from tfn.model.baseline_regressors import (
    TransformerRegressor, PerformerRegressor, LSTMRegressor, CNNRegressor
)
import tfn.tfn_datasets.climate_loader as cl

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _prepare_device(device_arg: str | None = None) -> torch.device:
    """Return appropriate torch.device (CPU / CUDA)."""
    if device_arg is None or device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"Using device: {device}")
    return device


def _get_dataloaders(dataset_name: str, seq_len: int, batch_size: int,
                     num_workers: int = 2, shuffle_train: bool = True,
                     **kwargs) -> Tuple[DataLoader, DataLoader, int, int]:
    """Return train/val loaders, num_features, output_dim.

    Parameters
    ----------
    dataset_name : str
        electricity | jena | jena_multi
    seq_len : int
        Sequence length to use.
    batch_size : int
        Loader batch size.
    num_workers : int
        DataLoader workers (set 0 on Windows/CPU-only Colab).
    shuffle_train : bool
        Shuffle training examples.
    **kwargs : Additional arguments for dataset loaders
    """

    dataset_name = dataset_name.lower()
    loader_map = {
        "electricity": cl.load_electricity_transformer,
        "jena": cl.load_jena_climate,
        "jena_multi": cl.load_jena_climate_multi,
    }
    if dataset_name not in loader_map:
        raise ValueError(f"Unsupported climate dataset: {dataset_name}. "
                         f"Supported: {list(loader_map)}")

    print(f"Loading {dataset_name} …")
    train_ds, val_ds, num_features = loader_map[dataset_name](seq_len=seq_len, **kwargs)

    # For time series regression, output dimension is 1 (single target)
    # For multi-variable, output dimension equals number of features
    output_dim = 1 if dataset_name != "jena_multi" else num_features

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(f"✓ Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"✓ Num features: {num_features}, Output dim: {output_dim}")
    return train_loader, val_loader, num_features, output_dim

# -----------------------------------------------------------------------------
# Training / Evaluation helpers
# -----------------------------------------------------------------------------

def _train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                 optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total = 0
    
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            
            # Calculate MSE
            mse = ((logits - labels) ** 2).sum().item()
            total_mse += mse
            total += labels.size(0)
    
    return {
        "loss": total_loss / len(loader.dataset),
        "mse": total_mse / total,
        "rmse": (total_mse / total) ** 0.5,
    }

# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train regressor on climate datasets")

    # Dataset / task
    p.add_argument("--dataset", type=str, required=True,
                   choices=["electricity", "jena", "jena_multi"],
                   help="Climate dataset name")
    p.add_argument("--seq_len", type=int, default=128,
                   help="Sequence length (time steps)")
    p.add_argument("--step", type=int, default=1,
                   help="Step size for sliding window")

    # Model selection
    p.add_argument("--model", type=str, default="tfn",
                   choices=["tfn", "transformer", "performer", "lstm", "cnn"],
                   help="Model architecture to use")

    # Model hyper-parameters
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--kernel_type", type=str, default="rbf",
                   choices=["rbf", "compact", "fourier"])
    p.add_argument("--evolution_type", type=str, default="cnn",
                   choices=["cnn", "spectral", "pde"])
    p.add_argument("--grid_size", type=int, default=64)
    p.add_argument("--time_steps", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training hyper-parameters
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="auto",
                   help="cuda | cpu | auto")
    p.add_argument("--num_workers", type=int, default=2)

    # I/O
    p.add_argument("--save_dir", type=str, default="outputs",
                   help="Directory to save checkpoints & logs")
    p.add_argument("--tag", type=str, default="",
                   help="Optional tag for run naming")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = _prepare_device(args.device)

    # Dataloaders
    train_loader, val_loader, num_features, output_dim = _get_dataloaders(
        args.dataset, args.seq_len, args.batch_size, args.num_workers,
        step=args.step)

    # Model selection
    if args.model == "tfn":
        model = TFNClassifier(
            vocab_size=num_features,  # Use num_features as vocab_size for time series
            embed_dim=args.embed_dim,
            num_classes=output_dim,   # Use output_dim as num_classes
            num_layers=args.num_layers,
            kernel_type=args.kernel_type,
            evolution_type=args.evolution_type,
            grid_size=args.grid_size,
            time_steps=args.time_steps,
            dropout=args.dropout,
        )
    elif args.model == "transformer":
        model = TransformerRegressor(
            input_dim=num_features,
            embed_dim=args.embed_dim,
            output_dim=output_dim,
            seq_len=args.seq_len,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "performer":
        model = PerformerRegressor(
            input_dim=num_features,
            embed_dim=args.embed_dim,
            output_dim=output_dim,
            seq_len=args.seq_len,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "lstm":
        model = LSTMRegressor(
            input_dim=num_features,
            embed_dim=args.embed_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "cnn":
        model = CNNRegressor(
            input_dim=num_features,
            embed_dim=args.embed_dim,
            output_dim=output_dim,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # Output dirs
    run_name = f"{args.dataset}_{args.model}_ed{args.embed_dim}_L{args.num_layers}{('-'+args.tag) if args.tag else ''}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_rmse": []}

    best_val_mse = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _evaluate(model, val_loader, criterion, device)
        val_loss, val_mse, val_rmse = metrics["loss"], metrics["mse"], metrics["rmse"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_rmse"].append(val_rmse)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
              f"Val MSE {val_mse:.4f} | Val RMSE {val_rmse:.4f}")

        # Save best checkpoint
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            ckpt_path = save_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ New best MSE — model saved to {ckpt_path}")

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete! Best Val MSE: {best_val_mse:.4f}")


if __name__ == "__main__":
    main() 