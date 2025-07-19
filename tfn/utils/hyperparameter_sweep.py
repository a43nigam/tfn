"""
Hyperparameter Sweep Utilities

Systematic hyperparameter search for fair model comparisons.
"""

import itertools
import json
import os
from typing import Dict, List, Any, Optional, Callable
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter sweep."""
    
    # Model architecture parameters
    embed_dim: int
    num_layers: int
    dropout: float
    
    # Training parameters
    learning_rate: float
    batch_size: int
    weight_decay: float
    
    # TFN-specific parameters (if applicable)
    kernel_type: Optional[str] = None
    evolution_type: Optional[str] = None
    grid_size: Optional[int] = None
    time_steps: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'kernel_type': self.kernel_type,
            'evolution_type': self.evolution_type,
            'grid_size': self.grid_size,
            'time_steps': self.time_steps
        }


class HyperparameterSweep:
    """
    Systematic hyperparameter sweep for fair model comparisons.
    """
    
    def __init__(self, 
                 model_factory: Callable,
                 train_function: Callable,
                 eval_function: Callable,
                 save_dir: str = "hyperparameter_sweeps"):
        """
        Initialize hyperparameter sweep.
        
        Args:
            model_factory: Function that creates model from config
            train_function: Function that trains model
            eval_function: Function that evaluates model
            save_dir: Directory to save results
        """
        self.model_factory = model_factory
        self.train_function = train_function
        self.eval_function = eval_function
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_configs(self, 
                        model_type: str,
                        dataset_name: str) -> List[HyperparameterConfig]:
        """
        Generate hyperparameter configurations for a specific model and dataset.
        
        Args:
            model_type: Type of model ('tfn', 'transformer', 'lstm', 'cnn')
            dataset_name: Name of dataset
            
        Returns:
            List of hyperparameter configurations
        """
        configs = []
        
        # Base hyperparameter ranges
        embed_dims = [64, 128, 256]
        num_layers = [1, 2, 3]
        learning_rates = [1e-4, 3e-4, 1e-3]
        batch_sizes = [16, 32, 64]
        dropouts = [0.1, 0.2]
        weight_decays = [1e-4, 1e-3]
        
        # Model-specific parameters
        if model_type == 'tfn':
            kernel_types = ['rbf', 'compact']
            evolution_types = ['cnn', 'spectral']
            grid_sizes = [32, 64]
            time_steps = [2, 3]
            
            # Generate TFN-specific configs
            for (embed_dim, num_layer, lr, batch_size, dropout, wd,
                 kernel_type, evolution_type, grid_size, time_step) in itertools.product(
                embed_dims, num_layers, learning_rates, batch_sizes, dropouts, weight_decays,
                kernel_types, evolution_types, grid_sizes, time_steps
            ):
                config = HyperparameterConfig(
                    embed_dim=embed_dim,
                    num_layers=num_layer,
                    dropout=dropout,
                    learning_rate=lr,
                    batch_size=batch_size,
                    weight_decay=wd,
                    kernel_type=kernel_type,
                    evolution_type=evolution_type,
                    grid_size=grid_size,
                    time_steps=time_step
                )
                configs.append(config)
        
        else:
            # Generate standard configs for other models
            for (embed_dim, num_layer, lr, batch_size, dropout, wd) in itertools.product(
                embed_dims, num_layers, learning_rates, batch_sizes, dropouts, weight_decays
            ):
                config = HyperparameterConfig(
                    embed_dim=embed_dim,
                    num_layers=num_layer,
                    dropout=dropout,
                    learning_rate=lr,
                    batch_size=batch_size,
                    weight_decay=wd
                )
                configs.append(config)
        
        # Limit number of configs for feasibility
        max_configs = 20
        if len(configs) > max_configs:
            # Sample evenly across the space
            step = len(configs) // max_configs
            configs = configs[::step][:max_configs]
        
        return configs
    
    def run_sweep(self,
                  model_type: str,
                  dataset_name: str,
                  num_epochs: int = 10,
                  device: str = "cpu") -> Dict[str, Any]:
        """
        Run hyperparameter sweep for a model and dataset.
        
        Args:
            model_type: Type of model to sweep
            dataset_name: Name of dataset
            num_epochs: Number of training epochs per config
            device: Device to run on
            
        Returns:
            Dictionary containing sweep results
        """
        self.logger.info(f"Starting hyperparameter sweep for {model_type} on {dataset_name}")
        
        # Generate configurations
        configs = self.generate_configs(model_type, dataset_name)
        self.logger.info(f"Generated {len(configs)} configurations")
        
        results = []
        best_config = None
        best_score = float('-inf')
        
        for i, config in enumerate(configs):
            self.logger.info(f"Testing config {i+1}/{len(configs)}: {config.to_dict()}")
            
            try:
                # Create model
                model = self.model_factory(config, model_type)
                model.to(device)
                
                # Train model
                train_metrics = self.train_function(
                    model, config, dataset_name, num_epochs, device
                )
                
                # Evaluate model
                eval_metrics = self.eval_function(model, dataset_name, device)
                
                # Combine results
                result = {
                    'config': config.to_dict(),
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics,
                    'model_type': model_type,
                    'dataset_name': dataset_name
                }
                
                results.append(result)
                
                # Track best configuration
                score = eval_metrics.get('accuracy', eval_metrics.get('f1_score', 0))
                if score > best_score:
                    best_score = score
                    best_config = config
                
                self.logger.info(f"Config {i+1} score: {score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error in config {i+1}: {str(e)}")
                continue
        
        # Save results
        sweep_results = {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'total_configs': len(configs),
            'successful_configs': len(results),
            'best_config': best_config.to_dict() if best_config else None,
            'best_score': best_score,
            'all_results': results
        }
        
        # Save to file
        filename = f"{model_type}_{dataset_name}_sweep.json"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(sweep_results, f, indent=2, default=str)
        
        self.logger.info(f"Sweep completed. Best score: {best_score:.4f}")
        self.logger.info(f"Results saved to {filepath}")
        
        return sweep_results
    
    def compare_models(self,
                      model_types: List[str],
                      dataset_name: str,
                      num_epochs: int = 10,
                      device: str = "cpu") -> Dict[str, Any]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            model_types: List of model types to compare
            dataset_name: Name of dataset
            num_epochs: Number of training epochs per config
            device: Device to run on
            
        Returns:
            Dictionary containing comparison results
        """
        self.logger.info(f"Starting model comparison on {dataset_name}")
        
        comparison_results = {}
        
        for model_type in model_types:
            self.logger.info(f"Running sweep for {model_type}")
            sweep_results = self.run_sweep(model_type, dataset_name, num_epochs, device)
            comparison_results[model_type] = sweep_results
        
        # Find best model overall
        best_model = None
        best_score = float('-inf')
        
        for model_type, results in comparison_results.items():
            score = results['best_score']
            if score > best_score:
                best_score = score
                best_model = model_type
        
        # Save comparison results
        comparison_summary = {
            'dataset_name': dataset_name,
            'model_types': model_types,
            'best_model': best_model,
            'best_score': best_score,
            'model_results': comparison_results
        }
        
        filename = f"comparison_{dataset_name}.json"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(comparison_summary, f, indent=2, default=str)
        
        self.logger.info(f"Comparison completed. Best model: {best_model} (score: {best_score:.4f})")
        self.logger.info(f"Results saved to {filepath}")
        
        return comparison_summary
    
    def print_sweep_summary(self, sweep_results: Dict[str, Any]) -> None:
        """
        Print summary of sweep results.
        
        Args:
            sweep_results: Results from hyperparameter sweep
        """
        print("\n" + "="*60)
        print(f"HYPERPARAMETER SWEEP SUMMARY")
        print("="*60)
        
        print(f"Model Type: {sweep_results['model_type']}")
        print(f"Dataset: {sweep_results['dataset_name']}")
        print(f"Total Configurations: {sweep_results['total_configs']}")
        print(f"Successful Configurations: {sweep_results['successful_configs']}")
        print(f"Best Score: {sweep_results['best_score']:.4f}")
        
        if sweep_results['best_config']:
            print(f"\nBest Configuration:")
            for key, value in sweep_results['best_config'].items():
                print(f"  {key}: {value}")
        
        # Show top 5 configurations
        if sweep_results['all_results']:
            print(f"\nTop 5 Configurations:")
            sorted_results = sorted(
                sweep_results['all_results'],
                key=lambda x: x['eval_metrics'].get('accuracy', x['eval_metrics'].get('f1_score', 0)),
                reverse=True
            )
            
            for i, result in enumerate(sorted_results[:5]):
                score = result['eval_metrics'].get('accuracy', result['eval_metrics'].get('f1_score', 0))
                print(f"  {i+1}. Score: {score:.4f}")
                print(f"     Config: {result['config']}")
        
        print("="*60)


def create_model_factory():
    """
    Create a model factory function for hyperparameter sweep.
    
    Returns:
        Function that creates models from configurations
    """
    def model_factory(config: HyperparameterConfig, model_type: str):
        """Create model from configuration."""
        if model_type == 'tfn':
            from ..model.tfn_classifiers import TFNClassifier
            return TFNClassifier(
                vocab_size=10000,  # Default vocab size
                embed_dim=config.embed_dim,
                num_classes=2,  # Default for binary classification
                num_layers=config.num_layers,
                kernel_type=config.kernel_type,
                evolution_type=config.evolution_type,
                grid_size=config.grid_size,
                time_steps=config.time_steps,
                dropout=config.dropout
            )
        elif model_type == 'transformer':
            from ..model.baseline_classifiers import TransformerClassifier
            return TransformerClassifier(
                vocab_size=10000,
                embed_dim=config.embed_dim,
                num_classes=2,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
        elif model_type == 'lstm':
            from ..model.baseline_classifiers import LSTMClassifier
            return LSTMClassifier(
                vocab_size=10000,
                embed_dim=config.embed_dim,
                num_classes=2,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
        elif model_type == 'cnn':
            from ..model.baseline_classifiers import CNNClassifier
            return CNNClassifier(
                vocab_size=10000,
                embed_dim=config.embed_dim,
                num_classes=2,
                dropout=config.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return model_factory 