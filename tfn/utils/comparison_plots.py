"""
Comparison Plot Utilities

Generate performance vs efficiency plots for model comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
from pathlib import Path


class ComparisonPlotter:
    """
    Utility for creating comparison plots between models.
    """
    
    def __init__(self, save_dir: str = "comparison_plots"):
        """
        Initialize comparison plotter.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_performance_efficiency_plot(self,
                                         results: Dict[str, Dict[str, Any]],
                                         dataset_name: str,
                                         efficiency_metric: str = "throughput_samples_per_sec",
                                         performance_metric: str = "accuracy") -> None:
        """
        Create performance vs efficiency scatter plot.
        
        Args:
            results: Dictionary of model results
            dataset_name: Name of dataset
            efficiency_metric: Metric to use for efficiency (x-axis)
            performance_metric: Metric to use for performance (y-axis)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data
        model_names = []
        efficiencies = []
        performances = []
        colors = []
        
        color_map = {
            'tfn': '#FF6B6B',
            'transformer': '#4ECDC4', 
            'lstm': '#45B7D1',
            'cnn': '#96CEB4',
            'performer': '#FFEAA7'
        }
        
        for model_name, model_results in results.items():
            if efficiency_metric in model_results and performance_metric in model_results:
                model_names.append(model_name)
                efficiencies.append(model_results[efficiency_metric])
                performances.append(model_results[performance_metric])
                colors.append(color_map.get(model_name, '#95A5A6'))
        
        # Create scatter plot
        scatter = ax.scatter(efficiencies, performances, 
                           c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Add labels
        for i, model_name in enumerate(model_names):
            ax.annotate(model_name, 
                       (efficiencies[i], performances[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        # Add quadrant lines
        eff_median = np.median(efficiencies)
        perf_median = np.median(performances)
        
        ax.axvline(eff_median, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(perf_median, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(0.1, 0.9, 'High Performance\nLow Efficiency', 
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.text(0.9, 0.9, 'High Performance\nHigh Efficiency', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        ax.text(0.1, 0.1, 'Low Performance\nLow Efficiency', 
                transform=ax.transAxes, ha='left', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        ax.text(0.9, 0.1, 'Low Performance\nHigh Efficiency', 
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Labels and title
        ax.set_xlabel(f'Efficiency ({efficiency_metric.replace("_", " ").title()})', fontsize=12)
        ax.set_ylabel(f'Performance ({performance_metric.title()})', fontsize=12)
        ax.set_title(f'Performance vs Efficiency Comparison\n{dataset_name}', fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Save plot
        filename = f"performance_efficiency_{dataset_name}.png"
        filepath = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved to {filepath}")
    
    def create_radar_plot(self,
                          results: Dict[str, Dict[str, Any]],
                          dataset_name: str,
                          metrics: List[str]) -> None:
        """
        Create radar plot comparing multiple metrics across models.
        
        Args:
            results: Dictionary of model results
            dataset_name: Name of dataset
            metrics: List of metrics to compare
        """
        # Prepare data
        model_names = list(results.keys())
        num_models = len(model_names)
        num_metrics = len(metrics)
        
        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Color map
        colors = plt.cm.Set3(np.linspace(0, 1, num_models))
        
        # Plot each model
        for i, model_name in enumerate(model_names):
            values = []
            for metric in metrics:
                value = results[model_name].get(metric, 0)
                # Normalize to [0, 1] range
                if metric in ['accuracy', 'f1_score']:
                    values.append(value)  # Already in [0, 1]
                else:
                    # Normalize other metrics
                    all_values = [results[m].get(metric, 0) for m in model_names]
                    max_val = max(all_values) if all_values else 1
                    values.append(value / max_val if max_val > 0 else 0)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Title
        ax.set_title(f'Model Comparison - {dataset_name}', fontsize=14, fontweight='bold', pad=20)
        
        # Save plot
        filename = f"radar_comparison_{dataset_name}.png"
        filepath = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Radar plot saved to {filepath}")
    
    def create_bar_comparison(self,
                             results: Dict[str, Dict[str, Any]],
                             dataset_name: str,
                             metrics: List[str]) -> None:
        """
        Create bar chart comparing multiple metrics.
        
        Args:
            results: Dictionary of model results
            dataset_name: Name of dataset
            metrics: List of metrics to compare
        """
        model_names = list(results.keys())
        num_metrics = len(metrics)
        
        # Set up the figure
        fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 8))
        if num_metrics == 1:
            axes = [axes]
        
        # Color map
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract values for this metric
            values = [results[model].get(metric, 0) for model in model_names]
            
            # Create bar plot
            bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontweight='bold')
            
            # Labels
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            # Grid
            ax.grid(True, alpha=0.3, axis='y')
        
        # Overall title
        fig.suptitle(f'Model Comparison - {dataset_name}', fontsize=16, fontweight='bold')
        
        # Save plot
        filename = f"bar_comparison_{dataset_name}.png"
        filepath = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Bar comparison saved to {filepath}")
    
    def create_heatmap_comparison(self,
                                 results: Dict[str, Dict[str, Any]],
                                 dataset_name: str,
                                 metrics: List[str]) -> None:
        """
        Create heatmap comparing all metrics across all models.
        
        Args:
            results: Dictionary of model results
            dataset_name: Name of dataset
            metrics: List of metrics to compare
        """
        # Prepare data for heatmap
        model_names = list(results.keys())
        data = []
        
        for model_name in model_names:
            row = []
            for metric in metrics:
                value = results[model_name].get(metric, 0)
                row.append(value)
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, index=model_names, columns=metrics)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Value'})
        
        plt.title(f'Model Comparison Heatmap - {dataset_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save plot
        filename = f"heatmap_comparison_{dataset_name}.png"
        filepath = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Heatmap saved to {filepath}")
    
    def create_comprehensive_comparison(self,
                                      results: Dict[str, Dict[str, Any]],
                                      dataset_name: str) -> None:
        """
        Create comprehensive comparison with multiple plot types.
        
        Args:
            results: Dictionary of model results
            dataset_name: Name of dataset
        """
        print(f"Creating comprehensive comparison for {dataset_name}")
        
        # Define metrics for comparison
        performance_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        efficiency_metrics = ['throughput_samples_per_sec', 'total_parameters', 
                            'estimated_gflops', 'peak_memory_mb']
        
        # Filter available metrics
        available_performance = [m for m in performance_metrics 
                               if any(m in results[model] for model in results)]
        available_efficiency = [m for m in efficiency_metrics 
                              if any(m in results[model] for model in results)]
        
        # Create plots
        if len(available_performance) >= 2:
            self.create_performance_efficiency_plot(
                results, dataset_name,
                efficiency_metric=available_efficiency[0] if available_efficiency else 'accuracy',
                performance_metric=available_performance[0]
            )
        
        if len(available_performance) >= 3:
            self.create_radar_plot(results, dataset_name, available_performance)
        
        if available_performance:
            self.create_bar_comparison(results, dataset_name, available_performance)
        
        if available_efficiency:
            self.create_bar_comparison(results, dataset_name, available_efficiency)
        
        # Create heatmap with all available metrics
        all_metrics = available_performance + available_efficiency
        if all_metrics:
            self.create_heatmap_comparison(results, dataset_name, all_metrics)
        
        print(f"Comprehensive comparison completed for {dataset_name}")


def create_comparison_from_sweep_results(sweep_results: Dict[str, Any],
                                       dataset_name: str,
                                       save_dir: str = "comparison_plots") -> None:
    """
    Create comparison plots from hyperparameter sweep results.
    
    Args:
        sweep_results: Results from hyperparameter sweep
        dataset_name: Name of dataset
        save_dir: Directory to save plots
    """
    plotter = ComparisonPlotter(save_dir)
    
    # Extract best results for each model
    best_results = {}
    
    for model_type, results in sweep_results['model_results'].items():
        if results['best_config'] and results['best_score'] > float('-inf'):
            best_results[model_type] = {
                'accuracy': results['best_score'],
                'config': results['best_config']
            }
    
    if best_results:
        plotter.create_comprehensive_comparison(best_results, dataset_name)
    else:
        print("No valid results found for comparison") 