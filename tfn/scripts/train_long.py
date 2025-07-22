#!/usr/bin/env python3
"""train_long.py
Full-dataset training loop for Token Field Networks (TFN).

Key points
==========
1. Lives entirely under the ``tfn`` package – upload the single *tfn/* directory
   (plus requirements) to Kaggle/Colab and this script will run.
2. Supports both the 1-D ``TFNClassifier`` and the 2-D ``TFNClassifier2D``
   via the *--model* CLI argument.
3. Uses the standard dataset loaders in ``tfn.datasets.dataset_loaders`` so no
   external helper files are required.
4. Implements a cosine LR schedule with linear warm-up and gradient clipping.

Example
-------
Train a 1-D TFN on AG-News for 5 epochs:

    python -m tfn.scripts.train_long --dataset agnews --epochs 5 --model tfn1d

Train a 2-D TFN on CIFAR-10 text tokens (just as demonstration):

    python -m tfn.scripts.train_long --dataset agnews --model tfn2d \
           --embed_dim 256 --grid_height 32 --grid_width 32
"""
from __future__ import annotations

# pyright: reportMissingImports=false, reportGeneralTypeIssues=false

import argparse
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tfn.tfn_datasets import dataset_loaders as dl
from tfn.model import TFNClassifier
from tfn.model.registry import validate_kernel_evolution
from tfn.model.tfn_pytorch import ImageTFN as TFNClassifier2D

# Simple replacement for create_tfn2d_variants using ImageTFN
def create_tfn2d_variants(num_classes: int = 10) -> Dict[str, nn.Module]:
    return {
        "image_basic": TFNClassifier2D(num_classes=num_classes),
    }


# -----------------------------------------------------------------------------
# Scheduler utilities
# -----------------------------------------------------------------------------


class WarmCosineLR(optim.lr_scheduler._LRScheduler):
    """Linear warm-up followed by cosine decay to zero."""

    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch + 1  # because scheduler.step() is called after update
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# -----------------------------------------------------------------------------
# Helper – create model dict similar to the old ``create_tfn_variants``
# -----------------------------------------------------------------------------

def create_tfn_variants(vocab_size: int, num_classes: int, embed_dim: int = 128, 
                       kernel_type: str = "rbf", evolution_type: str = "cnn",
                       grid_size: int = 100, time_steps: int = 3) -> Dict[str, nn.Module]:
    """Return a minimal set of 1-D TFN variants for convenience."""
    return {
        "tfn_basic": TFNClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_layers=2,
            kernel_type=kernel_type,
            evolution_type=evolution_type,
            grid_size=grid_size,
            time_steps=time_steps,
            task="classification"
        ),
        "tfn_deep": TFNClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_layers=4,
            kernel_type=kernel_type,
            evolution_type=evolution_type,
            grid_size=grid_size,
            time_steps=time_steps,
            task="classification"
        ),
    }


# -----------------------------------------------------------------------------
# Training / evaluation loops
# -----------------------------------------------------------------------------

def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.argmax(dim=-1) == target).float().mean().item()


def _train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: WarmCosineLR, device: torch.device):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)[0] if isinstance(model, TFNClassifier2D) else model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        bsz = inputs.size(0)
        total_loss += loss.item() * bsz
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_seen += bsz
    return total_loss / total_seen, total_correct / total_seen


def _eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)[0] if isinstance(model, TFNClassifier2D) else model(inputs)
            loss = criterion(logits, labels)
            bsz = inputs.size(0)
            total_loss += loss.item() * bsz
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_seen += bsz
    return total_loss / total_seen, total_correct / total_seen


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Long-run TFN trainer (all dependencies inside tfn package)")
    # Dataset
    p.add_argument("--dataset", choices=["agnews", "yelp", "imdb"], default="agnews")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=10000)

    # Model
    p.add_argument("--model", choices=["tfn1d", "tfn2d"], default="tfn1d")
    p.add_argument("--variant", type=str, default="tfn_basic", help="Variant key for 1-D or 2-D model dict")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    
    # Add missing TFN parameters for full configurability
    p.add_argument("--kernel_type", type=str, default="rbf",
                   choices=["rbf", "compact", "fourier"],
                   help="Kernel type for field projection")
    p.add_argument("--evolution_type", type=str, default="cnn",
                   choices=["cnn", "pde"],
                   help="Evolution type for field dynamics")
    p.add_argument("--grid_size", type=int, default=100,
                   help="Grid size for field evaluation (1D)")
    p.add_argument("--time_steps", type=int, default=3,
                   help="Number of evolution time steps")

    # 2-D specific
    p.add_argument("--grid_height", type=int, default=32)
    p.add_argument("--grid_width", type=int, default=32)
    p.add_argument("--evo_steps", type=int, default=5)

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        validate_kernel_evolution(args.kernel_type, args.evolution_type)
    except ValueError as e:
        print(f"[ConfigError] {e}")
        return

    device = torch.device(args.device)

    # ------------------- Data ------------------- #
    if args.dataset == "agnews":
        train_ds, val_ds, vocab_size = dl.load_agnews(seq_len=args.seq_len, vocab_size=args.vocab_size)
    elif args.dataset == "yelp":
        train_ds, val_ds, vocab_size = dl.load_yelp_full(seq_len=args.seq_len, vocab_size=args.vocab_size)
    else:
        train_ds, val_ds, vocab_size = dl.load_imdb(seq_len=args.seq_len, vocab_size=args.vocab_size)

    num_classes = int(torch.unique(train_ds.tensors[1]).numel())
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # ------------------- Model ------------------ #
    if args.model == "tfn1d":
        variants = create_tfn_variants(vocab_size, num_classes, embed_dim=args.embed_dim,
                                     kernel_type=args.kernel_type, evolution_type=args.evolution_type,
                                     grid_size=args.grid_size, time_steps=args.time_steps)
    else:
        variants = create_tfn2d_variants(
            num_classes=num_classes,
        )

    if args.variant not in variants:
        raise ValueError(f"Variant {args.variant} not found. Available: {list(variants)}")
    model = variants[args.variant].to(device)
    if args.num_layers and hasattr(model, "num_layers"):
        model.num_layers = args.num_layers  # type: ignore[attr-defined]

    print(f"🚀 Training variant '{args.variant}' | params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---------------- Optimiser & Scheduler ---------------- #
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = math.ceil(len(train_loader)) * args.epochs
    scheduler = WarmCosineLR(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    # CSV logging
    out_path = Path(f"long_run_{args.variant}_{int(time.time())}.csv")
    with out_path.open("w") as fcsv:
        fcsv.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc = _eval(model, val_loader, criterion, device)
            fcsv.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val {val_loss:.4f} acc {val_acc:.3f} | "
                f"time {time.time()-t0:.1f}s"
            )

    print(f"✅ Finished. Metrics saved to {out_path}")


if __name__ == "__main__":
    main() 