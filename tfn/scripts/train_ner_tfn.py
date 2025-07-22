#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false
"""Train a Token Field Network on the CoNLL-2003 NER task.

Minimal dependencies: tfn_phase1_complete.py + ner_loader.py (or full repo).
This script keeps the code self-contained; it defines a simple 1-D TFNTagger
that outputs token-level logits (num_tags).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tfn.model.tfn_base import TrainableTFNLayer  # 1-D layer
import tfn.tfn_datasets.ner_loader as nl                          # load_conll2003
from tfn.model.tfn_2d import TrainableTFNLayer2D  # 2-D layer
from tfn.model.tfn_enhanced import create_enhanced_tfn_model  # Enhanced TFN
from tfn.model.registry import validate_kernel_evolution

# -----------------------------------------------------------------------------
class TFNTagger(nn.Module):
    """1-D TFN encoder + per-token linear head for NER."""

    def __init__(self, vocab_size: int, num_tags: int, embed_dim: int = 128,
                 num_layers: int = 2, grid_size: int = 100, evo_steps: int = 3,
                 kernel_type: str = "rbf", evolution_type: str = "cnn", time_steps: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TrainableTFNLayer(embed_dim=embed_dim,
                               grid_size=grid_size,
                               time_steps=time_steps,
                               kernel_type=kernel_type,
                               evolution_type=evolution_type)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_tags)

    def forward(self, input_ids: torch.Tensor):  # (B,L)
        B, L = input_ids.shape
        x = self.embed(input_ids)
        # positions 0..L-1 normalised
        pos = torch.arange(L, device=input_ids.device).float() / max(1, L-1)
        pos = pos.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        for layer in self.layers:
            x = layer(x, pos)
        x = self.dropout(x)
        logits = self.classifier(x)  # (B,L,T)
        return logits

# -----------------------------------------------------------------------------
class TFNTagger2D(nn.Module):
    """2-D TFN encoder for token-level tagging (row-major layout)."""

    def __init__(self, vocab_size: int, num_tags: int, embed_dim: int = 128,
                 num_layers: int = 4, grid_height: int = 16, grid_width: int = 16,
                 evo_steps: int = 5, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.H, self.W = grid_height, grid_width
        self.layers = nn.ModuleList([
            TrainableTFNLayer2D(
                embed_dim=embed_dim,
                field_dim=embed_dim,
                grid_height=grid_height,
                grid_width=grid_width,
                num_evolution_steps=evo_steps,
                channel_mixing=True,
                learnable_sigma=True,
                use_global_context=False,
            )
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_tags)

    def forward(self, input_ids: torch.Tensor):  # (B,L)
        B, L = input_ids.shape
        if L > self.H * self.W:
            raise ValueError(f"Sequence length {L} exceeds grid capacity {self.H*self.W}")

        x = self.embed(input_ids)  # (B,L,E)

        # Positions row-major
        idx = torch.arange(L, device=input_ids.device)
        pos_y = (idx // self.W).float()
        pos_x = (idx % self.W).float()
        positions = torch.stack([pos_y, pos_x], dim=-1)  # (L,2)
        positions = positions.unsqueeze(0).expand(B, -1, -1)  # (B,L,2)

        for layer in self.layers:
            x, _ = layer(x, positions)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

# -----------------------------------------------------------------------------
class EnhancedTFNTagger(nn.Module):
    """Enhanced TFN encoder + per-token linear head for NER."""

    def __init__(self, vocab_size: int, num_tags: int, embed_dim: int = 128,
                 num_layers: int = 2, grid_size: int = 100, evo_steps: int = 3,
                 kernel_type: str = "rbf", evolution_type: str = "cnn", time_steps: int = 3,
                 dropout: float = 0.1, interference_type: str = "standard",
                 propagator_type: str = "standard", operator_type: str = "standard",
                 pos_dim: int = 1, num_heads: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced TFN layers
        self.layers = nn.ModuleList([
            create_enhanced_tfn_model(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=1,  # Single layer per iteration
                pos_dim=pos_dim,
                kernel_type=kernel_type,
                evolution_type=evolution_type,
                interference_type=interference_type,
                propagator_type=propagator_type,
                operator_type=operator_type,
                grid_size=grid_size,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=512  # Default max sequence length
            )
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_tags)

    def forward(self, input_ids: torch.Tensor):  # (B,L)
        B, L = input_ids.shape
        x = self.embed(input_ids)
        
        # Process through Enhanced TFN layers
        for layer in self.layers:
            x = layer(input_ids)  # Enhanced TFN handles positions internally
        
        x = self.dropout(x)
        logits = self.classifier(x)  # (B,L,T)
        return logits

# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train TFN on CoNLL-2003 NER")
    p.add_argument("--tfn_type", type=str, choices=["1d", "2d", "enhanced"], default="1d",
                   help="Use 1d, 2d, or enhanced TFN encoder")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--grid_size", type=int, default=100)
    p.add_argument("--grid_height", type=int, default=16)
    p.add_argument("--grid_width", type=int, default=16)
    p.add_argument("--evo_steps", type=int, default=3)
    # Add missing TFN parameters for full configurability
    p.add_argument("--kernel_type", type=str, default="rbf",
                   choices=["rbf", "compact", "fourier"],
                   help="Kernel type for field projection")
    p.add_argument("--evolution_type", type=str, default="cnn",
                   choices=["cnn", "pde"],
                   help="Evolution type for field dynamics")
    p.add_argument("--time_steps", type=int, default=3,
                   help="Number of evolution time steps")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate")
    
    # Enhanced TFN specific parameters
    p.add_argument("--interference_type", type=str, default="standard",
                   choices=["standard", "causal", "multiscale", "physics"],
                   help="Field interference type for Enhanced TFN")
    p.add_argument("--propagator_type", type=str, default="standard",
                   choices=["standard", "adaptive", "causal"],
                   help="Field propagator type for Enhanced TFN")
    p.add_argument("--operator_type", type=str, default="standard",
                   choices=["standard", "fractal", "causal", "meta"],
                   help="Field interaction operator type for Enhanced TFN")
    p.add_argument("--pos_dim", type=int, default=1,
                   help="Position dimension (1 for 1D, 2 for 2D)")
    p.add_argument("--num_heads", type=int, default=8,
                   help="Number of attention heads for Enhanced TFN")
    p.add_argument("--use_physics_constraints", action="store_true",
                   help="Use physics constraints during training for Enhanced TFN")
    p.add_argument("--constraint_weight", type=float, default=0.1,
                   help="Weight for physics constraint loss")
    
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="ner_runs")
    return p.parse_args()

# -----------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimiser, device, use_physics_constraints=False, constraint_weight=0.1):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Add physics constraints if enabled
        if use_physics_constraints and hasattr(model, 'get_physics_constraints'):
            constraints = model.get_physics_constraints()
            if constraints:
                constraint_loss = sum(constraints.values())
                loss = loss + constraint_weight * constraint_loss
        
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            tot_loss += loss.item() * x.size(0)
            preds = logits.argmax(-1)
            mask = (y != nl.TAG2IDX['O'])  # consider non-O for accuracy
            correct += ((preds == y) & mask).sum().item()
            count += mask.sum().item()
    return tot_loss / len(loader.dataset), correct / max(1, count)

# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    try:
        validate_kernel_evolution(args.kernel_type, args.evolution_type)
    except ValueError as e:
        print(f"[ConfigError] {e}")
        return

    device = torch.device(args.device)

    train_ds, val_ds, test_ds, vocab_size, num_tags = nl.load_conll2003()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    if args.tfn_type == "1d":
        model = TFNTagger(vocab_size, num_tags, args.embed_dim, args.num_layers,
                          args.grid_size, args.evo_steps,
                          kernel_type=args.kernel_type,
                          evolution_type=args.evolution_type,
                          time_steps=args.time_steps).to(device)
    elif args.tfn_type == "enhanced":
        model = EnhancedTFNTagger(vocab_size, num_tags, args.embed_dim, args.num_layers,
                                 args.grid_size, args.evo_steps,
                                 kernel_type=args.kernel_type,
                                 evolution_type=args.evolution_type,
                                 time_steps=args.time_steps,
                                 dropout=args.dropout,
                                 interference_type=args.interference_type,
                                 propagator_type=args.propagator_type,
                                 operator_type=args.operator_type,
                                 pos_dim=args.pos_dim,
                                 num_heads=args.num_heads).to(device)
    else:
        model = TFNTagger2D(vocab_size, num_tags, args.embed_dim,
                            args.num_layers, args.grid_height, args.grid_width,
                            args.evo_steps).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=nl.TAG2IDX['O'])
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = Path(args.save_dir) / f"tfn_ner_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best = 0.0
    history = []
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, criterion, optimiser, device,
                               use_physics_constraints=args.use_physics_constraints,
                               constraint_weight=args.constraint_weight)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history.append((tr_loss, val_loss, val_acc))
        print(f"Epoch {epoch}/{args.epochs} | train {tr_loss:.4f} | val {val_loss:.4f} | F1-like acc {val_acc:.3f}")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), run_dir/"best.pt")

    with open(run_dir/"history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best val acc {best:.3f}. Model saved in {run_dir}.")


if __name__ == "__main__":
    main() 