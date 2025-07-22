#!/usr/bin/env python3
"""
Hyperparameter sweep script for TFN models.

Usage:
    python sweep_tfn.py --model tfn_classifier --sweep_type kernel
    python sweep_tfn.py --model tfn_regressor --sweep_type evolution
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import json
from typing import Dict, List, Any
import itertools

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TFNClassifier, TFNRegressor
from utils.data_utils import (
    create_synthetic_data, create_synthetic_regression_data,
    TextClassificationDataset, SequenceRegressionDataset,
    create_vocab, create_dataloader, collate_fn_classification,
    collate_fn_regression
)
from utils.metrics import evaluate_model, compare_models
from utils.plot_utils import plot_model_comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for TFN models')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='tfn_classifier',
                       choices=['tfn_classifier', 'tfn_regressor'],
                       help='Model type to sweep')
    parser.add_argument('--sweep_type', type=str, default='kernel',
                       choices=['kernel', 'evolution', 'architecture'],
                       help='Type of hyperparameter sweep')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs per configuration')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save results JSON')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup device for training."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def get_sweep_configurations(sweep_type: str, model_type: str) -> List[Dict[str, Any]]:
    """Get hyperparameter configurations to sweep over."""
    configs = []
    
    if sweep_type == 'kernel':
        # Sweep over kernel types
        kernel_types = ['rbf', 'compact', 'fourier']
        for kernel_type in kernel_types:
            configs.append({
                'kernel_type': kernel_type,
                'evolution_type': 'cnn',
                'embed_dim': 64,
                'num_layers': 2,
                'grid_size': 100,
                'time_steps': 3,
                'dropout': 0.1
            })
    
    elif sweep_type == 'evolution':
        # Sweep over evolution types
        evolution_types = ['cnn', 'pde']
        for evolution_type in evolution_types:
            configs.append({
                'kernel_type': 'rbf',
                'evolution_type': evolution_type,
                'embed_dim': 64,
                'num_layers': 2,
                'grid_size': 100,
                'time_steps': 3,
                'dropout': 0.1
            })
    
    elif sweep_type == 'architecture':
        # Sweep over architectural parameters
        embed_dims = [32, 64, 128]
        num_layers = [1, 2, 3]
        grid_sizes = [50, 100, 200]
        
        for embed_dim, num_layer, grid_size in itertools.product(embed_dims, num_layers, grid_sizes):
            configs.append({
                'kernel_type': 'rbf',
                'evolution_type': 'cnn',
                'embed_dim': embed_dim,
                'num_layers': num_layer,
                'grid_size': grid_size,
                'time_steps': 3,
                'dropout': 0.1
            })
    
    return configs


def create_model_from_config(config: Dict[str, Any], model_type: str, device: torch.device):
    """Create model from configuration."""
    if model_type == 'tfn_classifier':
        vocab_size = 1000
        num_classes = 3
        
        model = TFNClassifier(
            vocab_size=vocab_size,
            embed_dim=config['embed_dim'],
            num_classes=num_classes,
            num_layers=config['num_layers'],
            kernel_type=config['kernel_type'],
            evolution_type=config['evolution_type'],
            grid_size=config['grid_size'],
            time_steps=config['time_steps'],
            dropout=config['dropout'],
            task="classification"
        )
        
    elif model_type == 'tfn_regressor':
        input_dim = 32
        output_dim = 8
        
        model = TFNRegressor(
            input_dim=input_dim,
            embed_dim=config['embed_dim'],
            output_dim=output_dim,
            num_layers=config['num_layers'],
            kernel_type=config['kernel_type'],
            evolution_type=config['evolution_type'],
            grid_size=config['grid_size'],
            time_steps=config['time_steps'],
            dropout=config['dropout']
        )
    
    model = model.to(device)
    return model


def create_data_for_sweep(model_type: str, num_samples: int, batch_size: int):
    """Create data for hyperparameter sweep."""
    if model_type == 'tfn_classifier':
        # Create synthetic text classification data
        train_texts, train_labels = create_synthetic_data(
            num_samples=num_samples,
            seq_len=50,
            vocab_size=1000,
            num_classes=3
        )
        
        val_texts, val_labels = create_synthetic_data(
            num_samples=num_samples // 4,
            seq_len=50,
            vocab_size=1000,
            num_classes=3
        )
        
        # Create vocabulary
        vocab = create_vocab(train_texts + val_texts, min_freq=2)
        
        # Create datasets
        train_dataset = TextClassificationDataset(train_texts, train_labels, vocab, 50)
        val_dataset = TextClassificationDataset(val_texts, val_labels, vocab, 50)
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset, batch_size, True, 0, collate_fn_classification
        )
        val_loader = create_dataloader(
            val_dataset, batch_size, False, 0, collate_fn_classification
        )
        
        return train_loader, val_loader, 'classification', 3
        
    elif model_type == 'tfn_regressor':
        # Create synthetic regression data
        train_sequences, train_targets = create_synthetic_regression_data(
            num_samples=num_samples,
            seq_len=50,
            feature_dim=32,
            output_dim=8
        )
        
        val_sequences, val_targets = create_synthetic_regression_data(
            num_samples=num_samples // 4,
            seq_len=50,
            feature_dim=32,
            output_dim=8
        )
        
        # Create datasets
        train_dataset = SequenceRegressionDataset(train_sequences, train_targets, 50)
        val_dataset = SequenceRegressionDataset(val_sequences, val_targets, 50)
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset, batch_size, True, 0, collate_fn_regression
        )
        val_loader = create_dataloader(
            val_dataset, batch_size, False, 0, collate_fn_regression
        )
        
        return train_loader, val_loader, 'regression', None


def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                task_type, num_classes, epochs):
    """Train a single model configuration."""
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
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
            
            total_train_loss += loss.item()
        
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion, device, task_type, num_classes)
        val_losses.append(val_metrics['loss'])
    
    return val_metrics


def main():
    """Main hyperparameter sweep function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Get sweep configurations
    configs = get_sweep_configurations(args.sweep_type, args.model)
    print(f"Found {len(configs)} configurations to test")
    
    # Create data
    train_loader, val_loader, task_type, num_classes = create_data_for_sweep(
        args.model, args.num_samples, args.batch_size
    )
    
    # Setup training
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # Results storage
    results = {}
    
    print(f"\n🔍 Starting {args.sweep_type} sweep for {args.model}")
    print("=" * 60)
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}/{len(configs)}")
        print(f"Config: {config}")
        
        # Create model
        model = create_model_from_config(config, args.model, device)
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Train model
        final_metrics = train_model(
            model, train_loader, val_loader, optimizer, criterion, device,
            task_type, num_classes, args.epochs
        )
        
        # Store results
        config_name = f"{config['kernel_type']}_{config['evolution_type']}_{config['embed_dim']}d_{config['num_layers']}l"
        results[config_name] = final_metrics
        
        print(f"Final metrics: {final_metrics}")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print comparison
    print(f"\n🏆 Sweep Results ({args.sweep_type.upper()})")
    print("=" * 60)
    compare_models(results, task_type)
    
    # Plot comparison
    if task_type == 'classification':
        plot_model_comparison(results, 'accuracy')
    else:
        plot_model_comparison(results, 'mse')
    
    # Save results
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save_results}")
    
    print(f"\n🎉 {args.sweep_type} sweep completed!")


if __name__ == '__main__':
    main()
