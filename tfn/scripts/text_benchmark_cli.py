#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Text Benchmark CLI for Token Field Networks (TFN)

This standalone script makes it easy to train and evaluate a TFN classifier
on common text classification datasets directly from the command line, making
it Colab- and Kaggle-friendly.

Example usages
--------------
# Train on AG News with default hyper-parameters
python -m tfn.scripts.text_benchmark_cli --dataset agnews --epochs 10

# Train on IMDB with a larger model
python -m tfn.scripts.text_benchmark_cli --dataset imdb --embed_dim 128 --num_layers 3 \
       --batch_size 64 --epochs 20 --lr 3e-4

The script will automatically detect GPU availability, save checkpoints, and
print final accuracy metrics.
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
from tfn.tfn_datasets import dataset_loaders as dl

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
                     num_workers: int = 2, shuffle_train: bool = True
                     ) -> Tuple[DataLoader, DataLoader, int, int]:
    """Return train/val loaders, vocab_size, num_classes.

    Parameters
    ----------
    dataset_name : str
        agnews | yelp_full | imdb
    seq_len : int
        Sequence length to truncate/pad.
    batch_size : int
        Loader batch size.
    num_workers : int
        DataLoader workers (set 0 on Windows/CPU-only Colab).
    shuffle_train : bool
        Shuffle training examples.
    """

    dataset_name = dataset_name.lower()
    loader_map = {
        "agnews": dl.load_agnews,
        "yelp_full": dl.load_yelp_full,
        "imdb": dl.load_imdb,
    }
    if dataset_name not in loader_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         f"Supported: {list(loader_map)}")

    print(f"Loading {dataset_name} …")
    train_ds, val_ds, vocab_size = loader_map[dataset_name](seq_len=seq_len)

    # Infer number of classes from labels (labels are 0-indexed ints)
    def _num_classes(ds: TensorDataset) -> int:
        labels = ds.tensors[1]
        return int(labels.max().item() + 1)

    num_classes = _num_classes(train_ds)

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
    print(f"✓ Vocab size: {vocab_size}, Num classes: {num_classes}")
    return train_loader, val_loader, vocab_size, num_classes

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
    correct = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": correct / len(loader.dataset),
    }

# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TFN classifier on text datasets")

    # Dataset / task
    p.add_argument("--dataset", type=str, required=True,
                   choices=["agnews", "yelp_full", "imdb"],
                   help="Dataset name")
    p.add_argument("--seq_len", type=int, default=128,
                   help="Sequence length (tokens)")

    # Model hyper-parameters
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--kernel_type", type=str, default="rbf",
                   choices=["rbf", "compact", "fourier"])
    p.add_argument("--evolution_type", type=str, default="cnn",
                   choices=["cnn", "pde"])
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
    train_loader, val_loader, vocab_size, num_classes = _get_dataloaders(
        args.dataset, args.seq_len, args.batch_size, args.num_workers)

    # Model
    model = TFNClassifier(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        vocab_size=vocab_size,
        # seq_len=args.seq_len,  # Removed, not accepted by UnifiedTFN
        kernel_type=args.kernel_type,
        evolution_type=args.evolution_type,
        grid_size=args.grid_size,
        time_steps=args.time_steps,
        dropout=args.dropout,
        task="classification",  # Added to satisfy UnifiedTFN
        device=device,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Output dirs
    run_name = f"{args.dataset}_ed{args.embed_dim}_L{args.num_layers}{('-'+args.tag) if args.tag else ''}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _evaluate(model, val_loader, criterion, device)
        val_loss, val_acc = metrics["loss"], metrics["accuracy"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc*100:.2f}%")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = save_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ New best accuracy — model saved to {ckpt_path}")

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Training complete! Best Val Acc: " f"{best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main() 