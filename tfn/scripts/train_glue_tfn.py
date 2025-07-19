#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
GLUE Benchmark Training CLI for Token Field Networks (TFN)

This standalone script makes it easy to train and evaluate a TFN classifier
on GLUE benchmark tasks directly from the command line.

Example usages
--------------
# Train on SST-2 with default hyper-parameters
python -m tfn.scripts.train_glue_tfn --task sst2 --epochs 10

# Train on MRPC with a larger model
python -m tfn.scripts.train_glue_tfn --task mrpc --embed_dim 128 --num_layers 3 \
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
from tfn.model.baseline_classifiers import (
    TransformerClassifier, PerformerClassifier, LSTMClassifier, CNNClassifier
)
import tfn.tfn_datasets.glue_loader as gl

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


def _get_dataloaders(task_name: str, seq_len: int, batch_size: int,
                     num_workers: int = 2, shuffle_train: bool = True
                     ) -> Tuple[DataLoader, DataLoader, int, int]:
    """Return train/val loaders, vocab_size, num_classes.

    Parameters
    ----------
    task_name : str
        sst2 | mrpc | qqp | qnli | rte | cola | stsb | wnli
    seq_len : int
        Sequence length to truncate/pad.
    batch_size : int
        Loader batch size.
    num_workers : int
        DataLoader workers (set 0 on Windows/CPU-only Colab).
    shuffle_train : bool
        Shuffle training examples.
    """

    task_name = task_name.lower()
    loader_map = {
        "sst2": gl.load_sst2,
        "mrpc": gl.load_mrpc,
        "qqp": gl.load_qqp,
        "qnli": gl.load_qnli,
        "rte": gl.load_rte,
        "cola": gl.load_cola,
        "stsb": gl.load_stsb,
        "wnli": gl.load_wnli,
    }
    if task_name not in loader_map:
        raise ValueError(f"Unsupported GLUE task: {task_name}. "
                         f"Supported: {list(loader_map)}")

    print(f"Loading {task_name} …")
    train_ds, val_ds, vocab_size = loader_map[task_name](seq_len=seq_len)

    # Infer number of classes from labels
    def _num_classes(ds: TensorDataset) -> int:
        labels = ds.tensors[1]
        if labels.dtype == torch.float:
            # STS-B is regression, return 1 for regression head
            return 1
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
              device: torch.device, task: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            
            if task == "stsb":
                # STS-B is regression, use MSE
                preds = logits.squeeze()
                correct += ((preds - labels) ** 2).sum().item()
                total += labels.size(0)
            else:
                # Classification tasks
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    
    metrics = {"loss": total_loss / len(loader.dataset)}
    
    if task == "stsb":
        metrics["mse"] = correct / total
        metrics["rmse"] = (correct / total) ** 0.5
    else:
        metrics["accuracy"] = correct / total
    
    return metrics

# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train classifier on GLUE tasks")

    # Dataset / task
    p.add_argument("--task", type=str, required=True,
                   choices=["sst2", "mrpc", "qqp", "qnli", "rte", "cola", "stsb", "wnli"],
                   help="GLUE task name")
    p.add_argument("--seq_len", type=int, default=128,
                   help="Sequence length (tokens)")

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
    train_loader, val_loader, vocab_size, num_classes = _get_dataloaders(
        args.task, args.seq_len, args.batch_size, args.num_workers)

    # Model selection
    if args.model == "tfn":
        model = TFNClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            kernel_type=args.kernel_type,
            evolution_type=args.evolution_type,
            grid_size=args.grid_size,
            time_steps=args.time_steps,
            dropout=args.dropout,
        )
    elif args.model == "transformer":
        model = TransformerClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            seq_len=args.seq_len,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "performer":
        model = PerformerClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            seq_len=args.seq_len,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "lstm":
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif args.model == "cnn":
        model = CNNClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.task == "stsb":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Output dirs
    run_name = f"{args.task}_{args.model}_ed{args.embed_dim}_L{args.num_layers}{('-'+args.tag) if args.tag else ''}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    if args.task == "stsb":
        history["val_mse"] = []
        history["val_rmse"] = []
    else:
        history["val_acc"] = []

    best_val_metric = float('inf') if args.task == "stsb" else 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _evaluate(model, val_loader, criterion, device, args.task)
        val_loss = metrics["loss"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if args.task == "stsb":
            val_mse, val_rmse = metrics["mse"], metrics["rmse"]
            history["val_mse"].append(val_mse)
            history["val_rmse"].append(val_rmse)
            print(f"Epoch {epoch:02d}/{args.epochs} | "
                  f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                  f"Val MSE {val_mse:.4f} | Val RMSE {val_rmse:.4f}")
            current_metric = val_mse
        else:
            val_acc = metrics["accuracy"]
            history["val_acc"].append(val_acc)
            print(f"Epoch {epoch:02d}/{args.epochs} | "
                  f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                  f"Val Acc {val_acc*100:.2f}%")
            current_metric = val_acc

        # Save best checkpoint
        if (args.task == "stsb" and current_metric < best_val_metric) or \
           (args.task != "stsb" and current_metric > best_val_metric):
            best_val_metric = current_metric
            ckpt_path = save_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ New best {'MSE' if args.task == 'stsb' else 'accuracy'} — model saved to {ckpt_path}")

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if args.task == "stsb":
        print(f"Training complete! Best Val MSE: {best_val_metric:.4f}")
    else:
        print(f"Training complete! Best Val Acc: {best_val_metric*100:.2f}%")


if __name__ == "__main__":
    main() 