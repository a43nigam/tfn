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

from tfn.model.tfn_2d import TFNClassifier2D

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
    def __init__(self, embed_dim: int = 256, num_classes: int = 10,
                 patch_size: int = 4, grid_size: int = 8,
                 tfn_layers: int = 4, evo_steps: int = 5,
                 field_dropout: float = 0.0, multiscale: bool = False,
                 kernel_mix: bool = False, kernel_mix_scale: float = 2.0,
                 out_sigma_scale: float = 2.0, learnable_out_sigma: bool = False):
        super().__init__()
        self.patch_embed = PatchEmbed(3, patch_size, embed_dim)
        self.tfn = TFNClassifier2D(
            vocab_size=1,  # dummy (we supply embeddings directly)
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_layers=tfn_layers,
            num_evolution_steps=evo_steps,
            grid_height=grid_size,
            grid_width=grid_size,
            use_dynamic_positions=False,
            learnable_sigma=True,
            learnable_out_sigma=learnable_out_sigma,
            out_sigma_scale=out_sigma_scale,
            field_dropout=field_dropout,
            multiscale=multiscale,
            kernel_mix=kernel_mix,
            kernel_mix_scale=kernel_mix_scale,
            use_global_context=False,
        )
        # overwrite token embedding with identity (we pass embeddings)
        self.tfn.embed = nn.Identity()

    def forward(self, images: torch.Tensor):
        tokens = self.patch_embed(images)  # (B,N,E)
        logits, _ = self.tfn(tokens)  # treat tokens as pre-embedded
        return logits

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
                   choices=["cnn", "spectral", "pde"],
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

    model = VisionTFN(args.embed_dim, 10, args.patch_size, grid_size,
                      args.tfn_layers, args.evo_steps,
                      field_dropout=args.field_dropout,
                      multiscale=args.multiscale,
                      kernel_mix=args.kernel_mix,
                      kernel_mix_scale=args.kernel_mix_scale,
                      out_sigma_scale=args.out_sigma_scale,
                      learnable_out_sigma=args.learnable_out_sigma).to(device)
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