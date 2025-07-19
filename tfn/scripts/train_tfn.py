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

from model import TFNClassifier, TFNRegressor
from utils.data_utils import (
    create_synthetic_data, create_synthetic_regression_data,
    TextClassificationDataset, SequenceRegressionDataset,
    create_vocab, create_dataloader, collate_fn_classification,
    collate_fn_regression
)
from utils.metrics import evaluate_model, print_metrics
from utils.plot_utils import plot_training_curves


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TFN models')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='tfn_classifier',
                       choices=['tfn_classifier', 'tfn_regressor'],
                       help='Model type to train')
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of TFN layers')
    parser.add_argument('--kernel_type', type=str, default='rbf',
                       choices=['rbf', 'compact', 'fourier'],
                       help='Kernel type')
    parser.add_argument('--evolution_type', type=str, default='cnn',
                       choices=['cnn', 'spectral', 'pde'],
                       help='Evolution type')
    parser.add_argument('--grid_size', type=int, default=100,
                       help='Grid size for field')
    parser.add_argument('--time_steps', type=int, default=3,
                       help='Number of evolution time steps')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='synthetic',
                       choices=['synthetic'],
                       help='Dataset type')
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
    """Create model based on arguments."""
    if args.model == 'tfn_classifier':
        # For synthetic data
        vocab_size = 1000
        num_classes = 3
        
        model = TFNClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            kernel_type=args.kernel_type,
            evolution_type=args.evolution_type,
            grid_size=args.grid_size,
            time_steps=args.time_steps,
            dropout=args.dropout
        )
        
    elif args.model == 'tfn_regressor':
        input_dim = 32
        output_dim = 8
        
        model = TFNRegressor(
            input_dim=input_dim,
            embed_dim=args.embed_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            kernel_type=args.kernel_type,
            evolution_type=args.evolution_type,
            grid_size=args.grid_size,
            time_steps=args.time_steps,
            dropout=args.dropout
        )
    
    model = model.to(device)
    return model


def create_data(args):
    """Create training and validation data."""
    if args.data == 'synthetic':
        if args.model == 'tfn_classifier':
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


def train_epoch(model, train_loader, optimizer, criterion, device, task_type, log_interval):
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
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def main():
    """Main training function."""
    args = parse_args()
    
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
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, task_type, args.log_interval)
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
