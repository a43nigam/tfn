#!/usr/bin/env python3
"""PG-19 language-model training.

This script may be executed via `python tfn/scripts/train_pg19.py` (file path)
or `python -m tfn.scripts.train_pg19` (module).  To support the first form we
ensure the TFN repo root is on `sys.path` *before* importing from `tfn.*` and
replace old relative imports (`from ..foo import`) with absolute ones.
"""

import os, sys
# Make sure `tfn` is importable when running as a plain script
if "tfn" not in sys.modules:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

# Absolute imports instead of package-relative
from tfn.tfn_datasets.pg19_loader import (
    create_pg19_dataloader,
    compute_perplexity,
    measure_memory_usage,
)
# Models
from tfn.model.tfn_base import TrainableTFNLayer
from tfn.model.seq_baselines import (
    SimpleTransformerSeqModel,
    SimplePerformerSeqModel,
)
from tfn.core.grid_utils import compute_auto_grid_size, estimate_memory_usage, estimate_flops
from tfn.model.registry import validate_kernel_evolution


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class TFNLanguageModel(nn.Module):
    """TFN-based language model for PG-19."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        kernel_type: str = "rbf",
        evolution_type: str = "cnn",
        grid_size: Optional[int] = None,
        seq_len: int = 4096,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # Auto-compute grid size if not specified
        if grid_size is None:
            grid_size, _ = compute_auto_grid_size(seq_len, embed_dim)
        
        self.grid_size = int(grid_size)
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings
        self.pos_embeddings = nn.Embedding(seq_len, embed_dim)
        
        # TFN layers
        self.tfn_layers = nn.ModuleList([
            TrainableTFNLayer(
                embed_dim=embed_dim,
                kernel_type=kernel_type,
                evolution_type=evolution_type,
                grid_size=grid_size,
                time_steps=3,
                max_seq_len=seq_len,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.token_embeddings.weight, 0, 0.1)
        nn.init.normal_(self.pos_embeddings.weight, 0, 0.1)
        nn.init.normal_(self.output_projection.weight, 0, 0.1)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through TFN language model."""
        B, N = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_embeddings(input_ids)  # [B, N, D]
        
        # Position embeddings
        pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embeddings(pos_ids)  # [B, N, D]
        
        # Combine embeddings
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)
        
        # Create positions for TFN (normalized to [0, 1])
        positions = torch.arange(N, device=input_ids.device, dtype=torch.float32)
        positions = positions.unsqueeze(0).expand(B, -1).unsqueeze(-1) / float(N - 1)
        
        # TFN layers
        hidden_states = embeddings
        for layer in self.tfn_layers:
            hidden_states = layer(hidden_states, positions)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return logits


def create_model(
    model_type: str,
    vocab_size: int,
    embed_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    seq_len: int = 4096,
    grid_size: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """Create model based on type."""
    
    if model_type == "tfn":
        return TFNLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            seq_len=seq_len,
            grid_size=grid_size,
            **kwargs
        )
    
    elif model_type == "transformer":
        return SimpleTransformerSeqModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=seq_len,
            **kwargs
        )
    
    elif model_type == "performer":
        return SimplePerformerSeqModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=seq_len,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    for batch in train_loader:
        if isinstance(batch, torch.Tensor):
            input_ids = batch.to(device)
        else:
            input_ids = batch[0].to(device)
        
        # Shift for language modeling
        input_ids = input_ids[:, :-1]
        target_ids = batch[:, 1:]
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0  # Ignore padding
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        # Count non-padding tokens
        non_pad_mask = target_ids != 0
        num_tokens = non_pad_mask.sum().item()
        
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        num_batches += 1
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "train_loss": avg_loss,
        "train_perplexity": perplexity,
        "num_batches": num_batches
    }


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, torch.Tensor):
                input_ids = batch.to(device)
            else:
                input_ids = batch[0].to(device)
            
            # Shift for language modeling
            input_ids = input_ids[:, :-1]
            target_ids = batch[:, 1:]
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0  # Ignore padding
            )
            
            # Count non-padding tokens
            non_pad_mask = target_ids != 0
            num_tokens = non_pad_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity
    }


def measure_performance(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 50
) -> Dict[str, float]:
    """Measure model performance metrics."""
    model.eval()
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids)
    
    # Measure memory usage
    memory_info = measure_memory_usage(model, batch_size, seq_len, vocab_size, device)
    
    # Measure throughput
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_runs
    tokens_per_second = (batch_size * seq_len) / avg_time_per_batch
    
    return {
        **memory_info,
        "tokens_per_second": tokens_per_second,
        "avg_time_per_batch": avg_time_per_batch
    }


def main():
    parser = argparse.ArgumentParser(description="Train TFN on PG-19")
    parser.add_argument("--model_type", type=str, default="tfn", 
                       choices=["tfn", "transformer", "performer"],
                       help="Model type to train")
    parser.add_argument("--seq_len", type=int, default=4096,
                       help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=256,
                       help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--grid_size", type=int, default=None,
                       help="Grid size (auto-computed if None)")
    parser.add_argument("--kernel_type", type=str, default="rbf",
                       choices=["rbf", "compact", "fourier"],
                       help="Kernel type for TFN")
    parser.add_argument("--evolution_type", type=str, default="cnn",
                       choices=["cnn", "pde"],
                       help="Evolution type for TFN")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--max_train_chunks", type=int, default=1000,
                       help="Maximum training chunks per epoch")
    parser.add_argument("--max_val_chunks", type=int, default=100,
                       help="Maximum validation chunks per epoch")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--wandb_project", type=str, default="tfn-pg19",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Logging interval")
    
    args = parser.parse_args()

    # Validate kernel/evolution combo
    try:
        validate_kernel_evolution(args.kernel_type, args.evolution_type)
    except ValueError as e:
        print(f"[ConfigError] {e}")
        return
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating PG-19 dataloaders...")
    train_loader, val_loader, vocab_size = create_pg19_dataloader(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_train_chunks=args.max_train_chunks,
        max_val_chunks=args.max_val_chunks
    )
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Auto-compute grid size if needed
    if args.grid_size is None and args.model_type == "tfn":
        args.grid_size, _ = compute_auto_grid_size(args.seq_len, args.embed_dim)
        logger.info(f"Auto-computed grid size: {args.grid_size}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        grid_size=args.grid_size,
        kernel_type=args.kernel_type,
        evolution_type=args.evolution_type
    )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Measure initial performance
    logger.info("Measuring initial performance...")
    perf_metrics = measure_performance(
        model, args.batch_size, args.seq_len, vocab_size, device
    )
    logger.info(f"Initial performance: {perf_metrics}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # WandB logging
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model_type}-{args.seq_len}-{args.embed_dim}"
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_perplexity = float('inf')
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.clip_grad_norm
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        metrics = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            **train_metrics,
            **val_metrics
        }
        
        wandb.log(metrics)
        
        # Log to console
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch}: "
            f"Train PPL: {train_metrics['train_perplexity']:.2f}, "
            f"Val PPL: {val_metrics['val_perplexity']:.2f}, "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if val_metrics['val_perplexity'] < best_val_perplexity:
            best_val_perplexity = val_metrics['val_perplexity']
            torch.save(model.state_dict(), f"best_{args.model_type}_pg19.pt")
            logger.info(f"New best model saved! Val PPL: {best_val_perplexity:.2f}")
    
    # Final performance measurement
    logger.info("Measuring final performance...")
    final_perf = measure_performance(
        model, args.batch_size, args.seq_len, vocab_size, device
    )
    logger.info(f"Final performance: {final_perf}")
    
    wandb.log({"final_performance": final_perf})
    wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    import sys
    import os
    # Add the parent directory to the path so we can import tfn modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    main() 