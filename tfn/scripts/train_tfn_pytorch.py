#!/usr/bin/env python3
"""Train the PyTorch ImageTFN model on various datasets.

This script trains the new PyTorch ImageTFN implementation optimized for 2D image processing,
distinct from the previous token-based TFN used for 1D time series.

Usage:
    python train_tfn_pytorch.py --dataset cifar10 --epochs 50 --batch_size 128
    python train_tfn_pytorch.py --dataset cifar100 --epochs 100 --batch_size 64
"""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from tfn.model.tfn_pytorch import ImageTFN
from tfn.core.field_emitter import ImageFieldEmitter
from tfn.core.field_interference_block import ImageFieldInterference
from tfn.core.field_propagator import ImageFieldPropagator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PyTorch TFN model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'imagenet'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    
    # Model architecture arguments
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of TFN layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads (for interference)')
    parser.add_argument('--field_channels', type=int, default=64,
                       help='Number of field channels')
    parser.add_argument('--propagation_steps', type=int, default=3,
                       help='Number of field propagation steps')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='Kernel size for field operations')
    
    # Field-specific arguments
    parser.add_argument('--field_dropout', type=float, default=0.1,
                       help='Field dropout rate')
    parser.add_argument('--interference_rank', type=int, default=16,
                       help='Rank for low-rank interference')
    parser.add_argument('--propagation_type', type=str, default='diffusion',
                       choices=['diffusion', 'wave', 'schrodinger'],
                       help='Type of field propagation')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='Time step for field evolution')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Warmup epochs')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='./runs',
                       help='Directory to save results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, do not train')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save interval (epochs)')
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
    elif dataset_name == 'imagenet':
        return 1000, 224, 3
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
    elif dataset_name == 'imagenet':
        if is_training:
            return T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    elif args.dataset == 'imagenet':
        train_dataset = ImageFolder(root=f"{args.data_dir}/train", transform=train_transform)
        val_dataset = ImageFolder(root=f"{args.data_dir}/val", transform=val_transform)
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


def create_model(args, num_classes: int) -> ImageTFN:
    """Create the TFN model."""
    model = ImageTFN(
        in_ch=3,  # RGB images
        num_classes=num_classes
    )
    return model


def create_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Create optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )


def create_scheduler(optimizer: torch.optim.Optimizer, args, num_steps: int):
    """Create learning rate scheduler."""
    if args.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=num_steps)
    elif args.scheduler == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        return None


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, args, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
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
        
        if batch_idx % args.log_interval == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}: '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100. * pred.eq(target.view_as(pred)).sum().item() / data.size(0):.2f}%')
    
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


def main():
    args = parse_args()
    device = setup_device(args.device)
    
    # Create dataloaders
    train_loader, val_loader, num_classes = create_dataloaders(args)
    
    # Create model
    model = create_model(args, num_classes).to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    num_steps = len(train_loader) * args.epochs
    scheduler = create_scheduler(optimizer, args, num_steps)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup mixed precision
    scaler = None
    if args.mixed_precision and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch} with best acc {best_acc:.2f}")
    
    # Create save directory
    save_dir = Path(args.save_dir) / f"tfn_pytorch_{args.dataset}_{int(time.time())}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Training loop
    if not args.eval_only:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, args, scaler
            )
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device, scaler)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_acc': best_acc,
                }, save_dir / 'best.pth')
                print(f"New best accuracy: {best_acc:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_acc': best_acc,
                }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Final evaluation
    print("\nFinal evaluation:")
    val_loss, val_acc = validate(model, val_loader, criterion, device, scaler)
    print(f"Final Val Loss: {val_loss:.4f}, Final Val Acc: {val_acc:.4f}")
    print(f"Best Val Acc: {best_acc:.4f}")
    
    # Save final results
    results = {
        'final_val_loss': val_loss,
        'final_val_acc': val_acc,
        'best_val_acc': best_acc,
        'total_epochs': args.epochs
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main() 