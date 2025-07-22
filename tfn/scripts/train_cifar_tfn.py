#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false
"""Train a 2-D TFN classifier on CIFAR-10 using patch tokens.

Dependencies: torchvision.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from tfn.model.tfn_pytorch import ImageTFN
from tfn.model.registry import validate_kernel_evolution

"""Train a 2-D TFN on CIFAR datasets.

This script may be executed from an arbitrary working directory
(e.g. Kaggle) where the *tfn* source folder is merely present in the
session and not yet installed as a site-package.  To make that scenario
work seamlessly we prepend the parent-of-parent directory of this file
to `sys.path` **before** importing from `tfn.*` – this has no effect when
the package is already properly installed (it will be ignored because
`import tfn` succeeds).
"""

import os, sys
# Ensure `tfn` is importable when the package hasn’t been pip-installed yet
if "tfn" not in sys.modules:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

class PatchEmbed(nn.Module):
    """Convert an image to a sequence of patch embeddings."""
    def __init__(self, in_channels: int = 3, patch_size: int = 4, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x

class VisionTFN(nn.Module):
    """Wrapper around ImageTFN for consistency with old training loop."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.tfn = ImageTFN(in_ch=3, num_classes=num_classes)

    def forward(self, images: torch.Tensor):
        return self.tfn(images)

def parse_args():
    p = argparse.ArgumentParser("TFN on CIFAR-10")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--tfn_layers", type=int, default=4)
    p.add_argument("--evo_steps", type=int, default=5)
    p.add_argument("--field_dropout", type=float, default=0.0)
    p.add_argument("--multiscale", action="store_true")
    p.add_argument("--kernel_mix", action="store_true")
    p.add_argument("--kernel_mix_scale", type=float, default=2.0)
    p.add_argument("--out_sigma_scale", type=float, default=2.0)
    # Add missing TFN parameters for full configurability
    p.add_argument("--kernel_type", type=str, default="rbf",
                   choices=["rbf", "compact", "fourier"],
                   help="Kernel type for field projection")
    p.add_argument("--evolution_type", type=str, default="cnn",
                   choices=["cnn", "pde"],
                   help="Evolution type for field dynamics")
    p.add_argument("--grid_size", type=int, default=64,
                   help="Grid size for field evaluation")
    p.add_argument("--time_steps", type=int, default=3,
                   help="Number of evolution time steps")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate")
    out_grp = p.add_mutually_exclusive_group()
    out_grp.add_argument("--learnable_out_sigma", dest="learnable_out_sigma", action="store_true")
    out_grp.add_argument("--no_learnable_out_sigma", dest="learnable_out_sigma", action="store_false")
    p.set_defaults(learnable_out_sigma=False)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="cifar_runs")
    return p.parse_args()

def main():
    args = parse_args()

    try:
        validate_kernel_evolution(args.kernel_type, args.evolution_type)
    except ValueError as e:
        print(f"[ConfigError] {e}");
        return

    device = torch.device(args.device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_ds = CIFAR10(root="data", train=True, download=True, transform=transform)
    test_ds = CIFAR10(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    grid_size = 32 // args.patch_size  # e.g., 8 for 4-pixel patches

    model = VisionTFN(10).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = Path(args.save_dir) / f"tfn_cifar_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best = 0.0
    for epoch in range(1, args.epochs+1):
        model.train(); total, correct, loss_sum = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward(); opt.step()
            loss_sum += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            total += imgs.size(0); correct += (preds == labels).sum().item()
        train_acc = correct/total; train_loss = loss_sum/total

        # eval
        model.eval(); total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss_sum += loss.item()*imgs.size(0)
                preds = logits.argmax(1)
                total += imgs.size(0); correct += (preds == labels).sum().item()
        val_acc = correct/total; val_loss = loss_sum/total
        print(f"Epoch {epoch}/{args.epochs} | train {train_loss:.3f} acc {train_acc:.3f} | val {val_loss:.3f} acc {val_acc:.3f}")
        if val_acc>best:
            best=val_acc
            torch.save(model.state_dict(), run_dir/"best.pt")
    with open(run_dir/"metrics.json","w") as f:
        json.dump({"best_acc":best},f)
    print("Best val acc",best)

if __name__=="__main__":
    main() 