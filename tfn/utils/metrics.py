"""
Evaluation metrics for TFN models.

Metrics for evaluating Token Field Network performance on various tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy."""
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


def precision_recall_f1(logits: torch.Tensor, targets: torch.Tensor, 
                       num_classes: int) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score for each class."""
    predictions = torch.argmax(logits, dim=-1)
    
    precision = []
    recall = []
    f1 = []
    
    for class_idx in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((predictions == class_idx) & (targets == class_idx)).float().sum()
        fp = ((predictions == class_idx) & (targets != class_idx)).float().sum()
        fn = ((predictions != class_idx) & (targets == class_idx)).float().sum()
        
        # Calculate metrics
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precision.append(p.item())
        recall.append(r.item())
        f1.append(f.item())
    
    return {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1': np.mean(f1),
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1
    }


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate Mean Squared Error."""
    return F.mse_loss(predictions, targets).item()


def mae_loss(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate Mean Absolute Error."""
    return F.l1_loss(predictions, targets).item()


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate R² score for regression."""
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, 
                   task_type: str = "classification", num_classes: Optional[int] = None) -> Dict[str, float]:
    """Compute appropriate metrics based on task type."""
    if task_type == "classification":
        metrics = {
            'accuracy': accuracy(logits, targets)
        }
        
        if num_classes is not None:
            prf1 = precision_recall_f1(logits, targets, num_classes)
            metrics.update(prf1)
        
        return metrics
    
    elif task_type == "regression":
        return {
            'mse': mse_loss(logits, targets),
            'mae': mae_loss(logits, targets),
            'r2': r2_score(logits, targets)
        }
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                  criterion: nn.Module, device: torch.device,
                  task_type: str = "classification", num_classes: Optional[int] = None) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if task_type == "classification":
                input_ids = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                
                # Forward pass
                logits = model(input_ids)
                
            elif task_type == "regression":
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)
                
                # Forward pass
                logits = model(features)
            
            # Compute loss
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions and targets
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_logits, all_targets, task_type, num_classes)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def print_metrics(metrics: Dict[str, float], task_type: str = "classification"):
    """Print metrics in a formatted way."""
    print(f"\n📊 Evaluation Results ({task_type.upper()})")
    print("=" * 50)
    
    if task_type == "classification":
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        if 'precision' in metrics:
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Loss: {metrics['loss']:.4f}")
    
    elif task_type == "regression":
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Loss: {metrics['loss']:.4f}")


def compare_models(model_results: Dict[str, Dict[str, float]], 
                  task_type: str = "classification") -> None:
    """Compare multiple models and print results."""
    print(f"\n🏆 Model Comparison ({task_type.upper()})")
    print("=" * 60)
    
    # Print header
    if task_type == "classification":
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Loss':<10}")
        print("-" * 70)
        
        for model_name, metrics in model_results.items():
            print(f"{model_name:<20} {metrics['accuracy']:<10.4f} "
                  f"{metrics.get('precision', 0):<10.4f} "
                  f"{metrics.get('recall', 0):<10.4f} "
                  f"{metrics.get('f1', 0):<10.4f} "
                  f"{metrics['loss']:<10.4f}")
    
    elif task_type == "regression":
        print(f"{'Model':<20} {'MSE':<10} {'MAE':<10} {'R²':<10} {'Loss':<10}")
        print("-" * 60)
        
        for model_name, metrics in model_results.items():
            print(f"{model_name:<20} {metrics['mse']:<10.4f} "
                  f"{metrics['mae']:<10.4f} "
                  f"{metrics['r2']:<10.4f} "
                  f"{metrics['loss']:<10.4f}")
