#!/usr/bin/env python3
"""Benchmark PyTorch ImageTFN against ResNet and ViT baselines.

This script benchmarks the new PyTorch ImageTFN implementation optimized for 2D image processing,
distinct from the previous token-based TFN used for 1D time series.

Usage:
    python benchmark_tfn_pytorch.py --dataset cifar10 --epochs 50
    python benchmark_tfn_pytorch.py --dataset cifar100 --epochs 100 --models tfn resnet vit
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18, resnet50, vit_b_16, vit_l_16

from tfn.model.tfn_pytorch import ImageTFN


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark TFN against baselines')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    
    # Model arguments
    parser.add_argument('--models', nargs='+', 
                       default=['tfn', 'resnet18', 'resnet50', 'vit'],
                       choices=['tfn', 'resnet18', 'resnet50', 'vit'],
                       help='Models to benchmark')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='./benchmarks',
                       help='Directory to save results')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup device for training."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def get_dataset_info(dataset_name: str) -> Tuple[int, int, int]:
    """Get dataset information (num_classes, image_size, channels)."""
    if dataset_name == 'cifar10':
        return 10, 32, 3
    elif dataset_name == 'cifar100':
        return 100, 32, 3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_transforms(dataset_name: str, is_training: bool = True):
    """Create data transforms."""
    if dataset_name in ['cifar10', 'cifar100']:
        if is_training:
            return T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    num_classes, image_size, channels = get_dataset_info(args.dataset)
    
    train_transform = create_transforms(args.dataset, is_training=True)
    val_transform = create_transforms(args.dataset, is_training=False)
    
    if args.dataset == 'cifar10':
        train_dataset = CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = CIFAR10(root=args.data_dir, train=False, download=True, transform=val_transform)
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = CIFAR100(root=args.data_dir, train=False, download=True, transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, num_classes


def create_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """Create a model by name."""
    if model_name == 'tfn':
        model = ImageTFN(
            in_ch=3,  # RGB images
            num_classes=num_classes
        )
    elif model_name == 'resnet18':
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vit':
        model = vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
            device: torch.device, scaler=None):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def benchmark_model(model_name: str, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, device: torch.device, args) -> Dict:
    """Benchmark a single model."""
    print(f"\n{'='*50}")
    print(f"Benchmarking {model_name.upper()}")
    print(f"{'='*50}")
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    # Create optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Setup mixed precision
    scaler = None
    if args.mixed_precision and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, scaler)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    total_time = time.time() - start_time
    
    print(f"\nFinal Results for {model_name}:")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Final Val Acc: {val_acc:.4f}")
    print(f"Training Time: {total_time:.2f}s")
    print(f"Time per Epoch: {total_time/args.epochs:.2f}s")
    
    return {
        'model_name': model_name,
        'num_parameters': num_params,
        'best_val_acc': best_acc,
        'final_val_acc': val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'total_time': total_time,
        'time_per_epoch': total_time / args.epochs
    }


def main():
    args = parse_args()
    device = setup_device(args.device)
    
    # Create dataloaders
    train_loader, val_loader, num_classes = create_dataloaders(args)
    
    # Benchmark each model
    results = []
    
    for model_name in args.models:
        print(f"\nCreating {model_name} model...")
        model = create_model(model_name, num_classes, device)
        
        result = benchmark_model(model_name, model, train_loader, val_loader, device, args)
        results.append(result)
    
    # Create save directory
    save_dir = Path(args.save_dir) / f"benchmark_{args.dataset}_{int(time.time())}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'models': results
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Params':<12} {'Best Acc':<10} {'Final Acc':<10} {'Time/Epoch':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model_name']:<15} "
              f"{result['num_parameters']:<12,} "
              f"{result['best_val_acc']:<10.4f} "
              f"{result['final_val_acc']:<10.4f} "
              f"{result['time_per_epoch']:<12.2f}")
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main() 