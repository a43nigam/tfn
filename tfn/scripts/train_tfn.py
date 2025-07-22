#!/usr/bin/env python3
"""
Training script for TFN models.

Usage:
    python train_tfn.py --model tfn_classifier --data synthetic --epochs 10
    python train_tfn.py --model tfn_regressor --data synthetic --epochs 20
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfn.model.tfn_unified import UnifiedTFN
from tfn.model.tfn_enhanced import create_enhanced_tfn_model  # For legacy enhanced path
from utils.data_utils import (
    create_synthetic_data, create_synthetic_regression_data,
    TextClassificationDataset, SequenceRegressionDataset,
    create_vocab, create_dataloader, collate_fn_classification,
    collate_fn_regression
)
from utils.metrics import evaluate_model, print_metrics
from utils.plot_utils import plot_training_curves

# Dataset registry & compatibility guard
from tfn.tfn_datasets.registry import get_dataset
from tfn.model.registry import validate_kernel_evolution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TFN models')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='tfn_classifier',
                       choices=['tfn_classifier', 'tfn_regressor', 'enhanced_tfn_classifier', 'enhanced_tfn_regressor'],
                       help='Model type to train')
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of TFN layers')
    parser.add_argument('--kernel_type', type=str, default='rbf',
                       choices=['rbf', 'compact', 'fourier'],
                       help='Kernel type')
    parser.add_argument('--evolution_type', type=str, default='cnn',
                       choices=['cnn', 'pde'],
                       help='Evolution type')
    parser.add_argument('--grid_size', type=int, default=100,
                       help='Grid size for field')
    parser.add_argument('--time_steps', type=int, default=3,
                       help='Number of evolution time steps')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Enhanced TFN specific parameters
    parser.add_argument('--interference_type', type=str, default='standard',
                       choices=['standard', 'causal', 'multiscale', 'physics'],
                       help='Field interference type for Enhanced TFN')
    parser.add_argument('--propagator_type', type=str, default='standard',
                       choices=['standard', 'adaptive', 'causal'],
                       help='Field propagator type for Enhanced TFN')
    parser.add_argument('--operator_type', type=str, default='standard',
                       choices=['standard', 'fractal', 'causal', 'meta'],
                       help='Field interaction operator type for Enhanced TFN')
    parser.add_argument('--pos_dim', type=int, default=1,
                       help='Position dimension (1 for 1D, 2 for 2D)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads for Enhanced TFN')
    parser.add_argument('--use_physics_constraints', action='store_true',
                       help='Use physics constraints during training for Enhanced TFN')
    parser.add_argument('--constraint_weight', type=float, default=0.1,
                       help='Weight for physics constraint loss')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='synthetic_copy',
                       help='Dataset key (see tfn/tfn_datasets/registry.py) or "synthetic" for legacy synthetic text')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--seq_len', type=int, default=50,
                       help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save model')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup device for training."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def create_model(args, device: torch.device):
    """Create model based on arguments using UnifiedTFN."""
    if args.model == 'tfn_classifier':
        model = UnifiedTFN.for_classification(
            vocab_size=args.embed_dim * 2,  # dummy vocab for synthetic
            num_classes=2,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            kernel_type=args.kernel_type,
            evolution_type=args.evolution_type,
            grid_size=args.grid_size,
            time_steps=args.time_steps,
            dropout=args.dropout,
        )
    elif args.model == 'tfn_regressor':
        model = UnifiedTFN.for_regression(
            input_dim=1,
            output_dim=1,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            kernel_type=args.kernel_type,
            evolution_type=args.evolution_type,
            grid_size=args.grid_size,
            time_steps=args.time_steps,
            dropout=args.dropout,
        )
    elif args.model == 'enhanced_tfn_classifier':
        # Keep using specialized enhanced model for now
        model = create_enhanced_tfn_model(
            vocab_size=args.embed_dim * 2,  # dummy vocab for synthetic
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
            dropout=args.dropout,
        )
    elif args.model == 'enhanced_tfn_regressor':
        # Enhanced regression via UnifiedTFN with enhanced layers
        model = UnifiedTFN.for_regression(
            input_dim=1,
            output_dim=1,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            kernel_type=args.kernel_type,
            evolution_type=args.evolution_type,
            grid_size=args.grid_size,
            time_steps=args.time_steps,
            dropout=args.dropout,
            use_enhanced=True,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model.to(device)
    return model


def create_data(args):
    """Create training and validation data."""
    if args.dataset == 'synthetic':
        if args.model in ['tfn_classifier', 'enhanced_tfn_classifier']:
            # Create synthetic text classification data
            train_texts, train_labels = create_synthetic_data(
                num_samples=args.num_samples,
                seq_len=args.seq_len,
                vocab_size=1000,
                num_classes=3
            )
            
            val_texts, val_labels = create_synthetic_data(
                num_samples=args.num_samples // 4,
                seq_len=args.seq_len,
                vocab_size=1000,
                num_classes=3
            )
            
            # Create vocabulary
            vocab = create_vocab(train_texts + val_texts, min_freq=2)
            
            # Create datasets
            train_dataset = TextClassificationDataset(train_texts, train_labels, vocab, args.seq_len)
            val_dataset = TextClassificationDataset(val_texts, val_labels, vocab, args.seq_len)
            
            # Create dataloaders
            train_loader = create_dataloader(
                train_dataset, args.batch_size, True, 0, collate_fn_classification
            )
            val_loader = create_dataloader(
                val_dataset, args.batch_size, False, 0, collate_fn_classification
            )
            
            return train_loader, val_loader, 'classification', 3
            
        elif args.model == 'tfn_regressor':
            # Create synthetic regression data
            train_sequences, train_targets = create_synthetic_regression_data(
                num_samples=args.num_samples,
                seq_len=args.seq_len,
                feature_dim=32,
                output_dim=8
            )
            
            val_sequences, val_targets = create_synthetic_regression_data(
                num_samples=args.num_samples // 4,
                seq_len=args.seq_len,
                feature_dim=32,
                output_dim=8
            )
            
            # Create datasets
            train_dataset = SequenceRegressionDataset(train_sequences, train_targets, args.seq_len)
            val_dataset = SequenceRegressionDataset(val_sequences, val_targets, args.seq_len)
            
            # Create dataloaders
            train_loader = create_dataloader(
                train_dataset, args.batch_size, True, 0, collate_fn_regression
            )
            val_loader = create_dataloader(
                val_dataset, args.batch_size, False, 0, collate_fn_regression
            )
            
            return train_loader, val_loader, 'regression', None
    else:
        # Use central dataset registry – expects it to return train & val tensors or DataLoaders
        train_ds, val_ds, _meta = get_dataset(
            args.dataset,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            vocab_size=1000,
        )

        # If loaders are returned already, just propagate; else wrap in loader
        if isinstance(train_ds, DataLoader):
            return train_ds, val_ds, _meta

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        return train_loader, val_loader, _meta


def train_epoch(model, train_loader, optimizer, criterion, device, task_type, log_interval, 
                use_physics_constraints=False, constraint_weight=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        if task_type == 'classification':
            input_ids = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits, targets)
            
        elif task_type == 'regression':
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            predictions = model(features)
            loss = criterion(predictions, targets)
        
        # Add physics constraints if enabled
        if use_physics_constraints and hasattr(model, 'get_physics_constraints'):
            constraints = model.get_physics_constraints()
            if constraints:
                constraint_loss = sum(constraints.values())
                loss = loss + constraint_weight * constraint_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def main():
    """Main training function."""
    args = parse_args()
    
    # Validate kernel/evolution combo early to fail-fast
    try:
        validate_kernel_evolution(args.kernel_type, args.evolution_type)
    except ValueError as e:
        print(f"[ConfigError] {e}");
        return

    # Setup device
    device = setup_device(args.device)
    
    # Create model
    model = create_model(args, device)
    print(f"Created {args.model} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data
    train_loader, val_loader, task_type, num_classes = create_data(args)
    print(f"Created {task_type} dataloaders")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"\n🚀 Starting training for {args.epochs} epochs")
    print("=" * 50)
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, task_type, args.log_interval,
                                 use_physics_constraints=args.use_physics_constraints,
                                 constraint_weight=args.constraint_weight)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion, device, task_type, num_classes)
        val_losses.append(val_metrics['loss'])
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        
        if task_type == 'classification':
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        else:
            print(f"  Val MSE: {val_metrics['mse']:.4f}")
        print()
    
    # Print final results
    print("🎉 Training completed!")
    print_metrics(val_metrics, task_type)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Save model
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")


if __name__ == '__main__':
    main()
