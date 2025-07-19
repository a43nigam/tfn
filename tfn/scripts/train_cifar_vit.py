#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false
"""Train a tiny Vision Transformer on CIFAR-10 to compare with 2-D TFN.

This mirrors the training loop of `train_cifar_tfn.py` so results are
comparable (same transforms, optimiser, epochs).  Dependencies: torchvision.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

# -----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, patch_size: int = 4, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                   # (B,E,H/P,W/P)
        x = x.flatten(2).transpose(1, 2)   # (B,N,E)
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size: int = 32, patch_size: int = 4,
                 embed_dim: int = 256, depth: int = 8, n_heads: int = 8,
                 mlp_ratio: float = 4.0, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(3, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=n_heads,
                                                   dim_feedforward=int(embed_dim*mlp_ratio),
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor):
        x = self.patch_embed(images)               # (B,N,E)
        B, N, E = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)  # (B,1,E)
        x = torch.cat([cls_token, x], dim=1)          # (B,N+1,E)
        x = self.pos_drop(x + self.pos_embed[:, :N+1])
        x = self.encoder(x)
        x = self.norm(x[:, 0])                        # cls token
        return self.head(x)

# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("ViT on CIFAR-10 (baseline)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="cifar_vit_runs")
    return p.parse_args()

# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    train_ds = CIFAR10(root="data", train=True, download=True, transform=transform)
    test_ds  = CIFAR10(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = ViT(img_size=32, patch_size=args.patch_size, embed_dim=args.embed_dim,
                depth=args.depth, n_heads=args.heads, num_classes=10).to(device)
    print("Params:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = Path(args.save_dir) / f"vit_cifar_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best = 0.0
    for epoch in range(1, args.epochs+1):
        model.train(); tot, correct, loss_sum = 0,0,0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(); logits = model(imgs); loss = criterion(logits, labels)
            loss.backward(); opt.step()
            loss_sum += loss.item()*imgs.size(0)
            pred = logits.argmax(1); tot += imgs.size(0); correct += (pred==labels).sum().item()
        tr_acc = correct/tot; tr_loss = loss_sum/tot

        model.eval(); tot, correct, loss_sum = 0,0,0.0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs); loss = criterion(logits, labels)
                loss_sum += loss.item()*imgs.size(0)
                pred = logits.argmax(1); tot+=imgs.size(0); correct+=(pred==labels).sum().item()
        val_acc = correct/tot; val_loss = loss_sum/tot

        print(f"Epoch {epoch}/{args.epochs} | train {tr_loss:.3f} acc {tr_acc:.3f} | val {val_loss:.3f} acc {val_acc:.3f}")
        if val_acc>best:
            best=val_acc
            torch.save(model.state_dict(), run_dir/"best.pt")
    with open(run_dir/"metrics.json","w") as f:
        json.dump({"best_acc":best},f)
    print("Best val acc",best)

if __name__=="__main__":
    main() 