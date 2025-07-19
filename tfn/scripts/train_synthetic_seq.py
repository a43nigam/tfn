#!/usr/bin/env python3
"""train_synthetic_seq.py
Train 1-D TFN and baseline models on synthetic sequence-to-sequence tasks.

The script is intentionally minimal and intended for *unit-test scale* runs –
all data are generated on-the-fly and reside in memory.

Example:
    python -m tfn.scripts.train_synthetic_seq \
        --model tfn --task copy --seq_len 256 --epochs 3
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repository root is on PYTHONPATH when the script is executed directly.
# This makes `import tfn.*` work even if the user calls the script via an
# absolute path outside the project root (e.g. Kaggle input datasets layout).
# ---------------------------------------------------------------------------

import sys
import pathlib as _pl

_THIS_FILE = _pl.Path(__file__).resolve()

# Walk upwards until we find a directory that contains the `tfn` package.
_REPO_ROOT = None
for parent in _THIS_FILE.parents:
    if (parent / "tfn" / "__init__.py").is_file():
        _REPO_ROOT = parent
        break

if _REPO_ROOT and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# pylint: disable=relative-beyond-top-level
from tfn.utils.synthetic_sequence_tasks import get_synthetic_sequence_dataloaders
from tfn.model.seq_baselines import (
    TFNSeqModel,
    SimpleTransformerSeqModel,
    SimplePerformerSeqModel,
)

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def token_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute per-token accuracy from model logits."""
    return (pred.argmax(-1) == target).float().mean().item()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # ---------------- Dataset ---------------- #
    train_loader, val_loader, vocab_size = get_synthetic_sequence_dataloaders(
        task=args.task,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch,
        seed=args.seed,
    )

    # ---------------- Model selection ---------------- #
    if args.model == "tfn":
        model = TFNSeqModel(vocab_size=vocab_size, seq_len=args.seq_len, embed_dim=args.embed_dim)
    elif args.model == "transformer":
        model = SimpleTransformerSeqModel(
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    elif args.model == "performer":
        model = SimplePerformerSeqModel(
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            proj_dim=args.proj_dim,
        )
    else:
        raise ValueError(f"Unknown model '{args.model}'.")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🚀 Training {args.model.upper()} | params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        # -------- Training -------- #
        model.train()
        train_loss, train_acc, seen = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            train_loss += loss.item() * bsz
            train_acc += token_accuracy(logits, y) * bsz
            seen += bsz

        train_loss /= seen
        train_acc /= seen

        # -------- Validation -------- #
        model.eval()
        val_loss, val_acc, v_seen = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

                v_bsz = x.size(0)
                val_loss += loss.item() * v_bsz
                val_acc += token_accuracy(logits, y) * v_bsz
                v_seen += v_bsz

        val_loss /= v_seen
        val_acc /= v_seen
        tokens_per_sec = (seen * args.seq_len) / (time.time() - start)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.3f} | tokens/s {tokens_per_sec:.0f}"
        )

    # Save final metrics (optional – useful in CI logs)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
        f.write(f"{args.epochs},{train_loss:.6f},{val_loss:.6f},{train_acc:.4f},{val_acc:.4f}\n")
    print(f"✅ Finished training. Metrics saved to {out_path}")

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tfn", "transformer", "performer"], default="tfn")
    parser.add_argument("--task", choices=["copy", "reverse"], default="copy")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=100)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--val_samples", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--proj_dim", type=int, default=64, help="Projection dim for Performer")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="synthetic_seq_results.csv")

    args = parser.parse_args()
    train(args) 