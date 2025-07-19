#!/usr/bin/env python3
"""
Enhanced TFN Training Script

A comprehensive training script for the Enhanced TFN model with full CLI configurability.
Supports all field interference types, evolution types, and physics constraints.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from tfn.model.tfn_enhanced import EnhancedTFNModel, create_enhanced_tfn_model
from tfn.tfn_datasets import dataset_loaders as dl


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for Enhanced TFN training."""
    parser = argparse.ArgumentParser("Enhanced TFN Training")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="synthetic",
                       choices=["synthetic", "glue", "agnews", "yelp", "imdb"],
                       help="Dataset to use for training")
    parser.add_argument("--task", type=str, default="classification",
                       choices=["classification", "regression", "language_modeling"],
                       help="Task type")
    parser.add_argument("--seq_len", type=int, default=128,
                       help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    
    # Model architecture
    parser.add_argument("--embed_dim", type=int, default=256,
                       help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of TFN layers")
    parser.add_argument("--pos_dim", type=int, default=1,
                       help="Position dimension (1 for 1D, 2 for 2D)")
    parser.add_argument("--grid_size", type=int, default=100,
                       help="Grid size for field evaluation")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    
    # Field components
    parser.add_argument("--kernel_type", type=str, default="rbf",
                       choices=["rbf", "compact", "fourier"],
                       help="Kernel type for field projection")
    parser.add_argument("--evolution_type", type=str, default="diffusion",
                       choices=["diffusion", "wave", "schrodinger", "cnn", "spectral"],
                       help="Field evolution type")
    parser.add_argument("--interference_type", type=str, default="standard",
                       choices=["standard", "causal", "multiscale", "physics"],
                       help="Field interference type")
    parser.add_argument("--propagator_type", type=str, default="standard",
                       choices=["standard", "adaptive", "causal"],
                       help="Field propagator type")
    parser.add_argument("--operator_type", type=str, default="standard",
                       choices=["standard", "fractal", "causal", "meta"],
                       help="Field interaction operator type")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                       help="Gradient clipping norm")
    
    # Physics constraints
    parser.add_argument("--use_physics_constraints", action="store_true",
                       help="Use physics constraints during training")
    parser.add_argument("--constraint_weight", type=float, default=0.1,
                       help="Weight for physics constraint loss")
    
    # Output and logging
    parser.add_argument("--save_dir", type=str, default="enhanced_tfn_runs",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Logging interval")
    
    return parser.parse_args()


def create_synthetic_dataset(num_samples: int = 1000, seq_len: int = 128, 
                           vocab_size: int = 1000, num_classes: int = 4) -> tuple:
    """Create synthetic dataset for testing Enhanced TFN."""
    # Generate random sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Generate labels (simple pattern: sum of tokens mod num_classes)
    labels = torch.sum(input_ids, dim=1) % num_classes
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    train_inputs = input_ids[:train_size]
    train_labels = labels[:train_size]
    val_inputs = input_ids[train_size:]
    val_labels = labels[train_size:]
    
    return (train_inputs, train_labels), (val_inputs, val_labels), vocab_size, num_classes


def load_dataset(args: argparse.Namespace) -> tuple:
    """Load dataset based on arguments."""
    if args.dataset == "synthetic":
        train_data, val_data, vocab_size, num_classes = create_synthetic_dataset(
            num_samples=2000, seq_len=args.seq_len, vocab_size=1000, num_classes=4
        )
        return train_data, val_data, vocab_size, num_classes
    elif args.dataset == "glue":
        # Load GLUE dataset
        train_data, val_data, vocab_size, num_classes = dl.load_glue_dataset(
            task="sst2", seq_len=args.seq_len
        )
        return train_data, val_data, vocab_size, num_classes
    else:
        # Load text classification datasets
        train_data, val_data, vocab_size, num_classes = dl.load_text_classification_dataset(
            dataset=args.dataset, seq_len=args.seq_len
        )
        return train_data, val_data, vocab_size, num_classes


def create_model(args: argparse.Namespace, vocab_size: int, num_classes: int) -> EnhancedTFNModel:
    """Create Enhanced TFN model with specified configuration."""
    model = create_enhanced_tfn_model(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        pos_dim=args.pos_dim,
        kernel_type=args.kernel_type,
        evolution_type=args.evolution_type,
        interference_type=args.interference_type,
        propagator_type=args.propagator_type,
        operator_type=args.operator_type,
        grid_size=args.grid_size,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    return model


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                args: argparse.Namespace) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # Compute loss
        if args.task == "classification":
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
        else:
            loss = criterion(logits, labels)
        
        # Add physics constraints if enabled
        if args.use_physics_constraints:
            constraints = model.get_physics_constraints()
            if constraints:
                constraint_loss = sum(constraints.values())
                loss = loss + args.constraint_weight * constraint_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_samples += input_ids.size(0)
        
        # Logging
        if batch_idx % args.log_interval == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    metrics = {
        "loss": total_loss / len(train_loader),
        "accuracy": total_correct / total_samples if args.task == "classification" else 0.0
    }
    
    return metrics


def evaluate(model: nn.Module, 
             val_loader: DataLoader, 
             criterion: nn.Module,
             device: torch.device,
             args: argparse.Namespace) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss
            if args.task == "classification":
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                predictions = logits.argmax(dim=-1)
                correct = (predictions == labels).sum().item()
                total_correct += correct
            else:
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            total_samples += input_ids.size(0)
    
    metrics = {
        "loss": total_loss / len(val_loader),
        "accuracy": total_correct / total_samples if args.task == "classification" else 0.0
    }
    
    return metrics


def main():
    """Main training function."""
    args = parse_args()
    device = torch.device(args.device)
    
    print(f"🚀 Starting Enhanced TFN Training")
    print(f"📊 Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Task: {args.task}")
    print(f"  Model: Enhanced TFN ({args.interference_type} interference, {args.evolution_type} evolution)")
    print(f"  Architecture: {args.num_layers} layers, {args.embed_dim} dim, {args.num_heads} heads")
    print(f"  Field Components: {args.kernel_type} kernel, {args.propagator_type} propagator, {args.operator_type} operators")
    print(f"  Physics Constraints: {'Enabled' if args.use_physics_constraints else 'Disabled'}")
    
    # Load dataset
    print(f"\n📦 Loading dataset: {args.dataset}")
    train_data, val_data, vocab_size, num_classes = load_dataset(args)
    
    # Create data loaders
    train_dataset = TensorDataset(train_data[0], train_data[1])
    val_dataset = TensorDataset(val_data[0], val_data[1])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Num classes: {num_classes}")
    
    # Create model
    print(f"\n🏗️ Creating Enhanced TFN model...")
    model = create_model(args, vocab_size, num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss() if args.task == "classification" else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create output directory
    save_dir = Path(args.save_dir) / f"enhanced_tfn_{int(time.time())}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    config['vocab_size'] = vocab_size
    config['num_classes'] = num_classes
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print(f"\n🎯 Starting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, args)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device, args)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Log results
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  💾 New best model saved! (Val Acc: {best_val_acc:.4f})")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'lr': scheduler.get_last_lr()[0]
        })
    
    # Save final results
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Results saved to: {save_dir}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    
    return best_val_acc


if __name__ == "__main__":
    main() 