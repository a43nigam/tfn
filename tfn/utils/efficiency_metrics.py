"""
Efficiency Metrics for TFN Models

Utilities for measuring model efficiency including:
- Throughput (samples/second)
- FLOPs (floating point operations)
- Memory usage
- Parameter count
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import psutil
import gc


class EfficiencyMetrics:
    """
    Utility class for measuring model efficiency metrics.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize efficiency metrics.
        
        Args:
            device: Device to run measurements on
        """
        self.device = device
        self.warmup_runs = 5
        self.measurement_runs = 10
    
    def measure_throughput(self, 
                          model: nn.Module, 
                          input_shape: Tuple[int, ...],
                          batch_size: int = 1) -> Dict[str, float]:
        """
        Measure model throughput in samples per second.
        
        Args:
            model: Model to measure
            input_shape: Shape of input tensor (excluding batch dimension)
            batch_size: Batch size for measurement
            
        Returns:
            Dictionary containing throughput metrics
        """
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        input_tensor = torch.randn(batch_size, *input_shape, device=self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize if using CUDA
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Measurement runs
        start_time = time.time()
        with torch.no_grad():
            for _ in range(self.measurement_runs):
                _ = model(input_tensor)
        
        # Synchronize if using CUDA
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_batch = total_time / self.measurement_runs
        throughput = batch_size / avg_time_per_batch
        
        return {
            "throughput_samples_per_sec": throughput,
            "avg_time_per_batch_ms": avg_time_per_batch * 1000,
            "total_time_sec": total_time,
            "measurement_runs": self.measurement_runs
        }
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: Model to count parameters for
            
        Returns:
            Dictionary containing parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params
        }
    
    def estimate_flops(self, 
                      model: nn.Module, 
                      input_shape: Tuple[int, ...],
                      batch_size: int = 1) -> Dict[str, int]:
        """
        Estimate FLOPs for a forward pass.
        
        Note: This is a simplified estimation. For more accurate results,
        use specialized libraries like fvcore or thop.
        
        Args:
            model: Model to estimate FLOPs for
            input_shape: Shape of input tensor (excluding batch dimension)
            batch_size: Batch size for estimation
            
        Returns:
            Dictionary containing FLOP estimates
        """
        # This is a simplified FLOP estimation
        # For more accurate results, consider using fvcore or thop
        
        total_flops = 0
        input_size = batch_size * torch.prod(torch.tensor(input_shape)).item()
        
        # Estimate FLOPs for common layer types
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # FLOPs for linear layer: 2 * input_features * output_features
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    total_flops += 2 * module.in_features * module.out_features * batch_size
            
            elif isinstance(module, nn.Conv1d):
                # FLOPs for conv1d: 2 * in_channels * out_channels * kernel_size * output_length
                if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                    # Estimate output length (simplified)
                    output_length = input_shape[0] - kernel_size + 1
                    total_flops += 2 * module.in_channels * module.out_channels * kernel_size * output_length * batch_size
            
            elif isinstance(module, nn.Embedding):
                # FLOPs for embedding: num_embeddings * embedding_dim
                if hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'):
                    total_flops += module.num_embeddings * module.embedding_dim * batch_size
        
        return {
            "estimated_flops": total_flops,
            "estimated_gflops": total_flops / 1e9,
            "input_size": input_size
        }
    
    def measure_memory_usage(self, 
                           model: nn.Module, 
                           input_shape: Tuple[int, ...],
                           batch_size: int = 1) -> Dict[str, float]:
        """
        Measure memory usage during forward pass.
        
        Args:
            model: Model to measure memory for
            input_shape: Shape of input tensor (excluding batch dimension)
            batch_size: Batch size for measurement
            
        Returns:
            Dictionary containing memory usage metrics
        """
        model.eval()
        model.to(self.device)
        
        # Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Get initial memory
        if self.device == "cuda":
            initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            initial_memory = psutil.Process().memory_info().rss / 1024**2  # MB
        
        # Create input and run forward pass
        input_tensor = torch.randn(batch_size, *input_shape, device=self.device)
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Get peak memory
        if self.device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            current_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            peak_memory = current_memory  # Simplified for CPU
        
        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "current_memory_mb": current_memory,
            "memory_increase_mb": peak_memory - initial_memory
        }
    
    def comprehensive_efficiency_report(self, 
                                     model: nn.Module,
                                     input_shape: Tuple[int, ...],
                                     batch_size: int = 1,
                                     model_name: str = "Model") -> Dict[str, Any]:
        """
        Generate comprehensive efficiency report.
        
        Args:
            model: Model to analyze
            input_shape: Shape of input tensor (excluding batch dimension)
            batch_size: Batch size for measurements
            model_name: Name of the model for reporting
            
        Returns:
            Comprehensive efficiency report
        """
        print(f"Generating efficiency report for {model_name}...")
        
        # Measure all metrics
        throughput_metrics = self.measure_throughput(model, input_shape, batch_size)
        parameter_metrics = self.count_parameters(model)
        flops_metrics = self.estimate_flops(model, input_shape, batch_size)
        memory_metrics = self.measure_memory_usage(model, input_shape, batch_size)
        
        # Combine all metrics
        report = {
            "model_name": model_name,
            "input_shape": input_shape,
            "batch_size": batch_size,
            "device": self.device,
            **throughput_metrics,
            **parameter_metrics,
            **flops_metrics,
            **memory_metrics
        }
        
        # Calculate efficiency ratios
        if parameter_metrics["total_parameters"] > 0:
            report["throughput_per_million_params"] = (
                throughput_metrics["throughput_samples_per_sec"] / 
                (parameter_metrics["total_parameters"] / 1e6)
            )
        
        if flops_metrics["estimated_flops"] > 0:
            report["throughput_per_gflop"] = (
                throughput_metrics["throughput_samples_per_sec"] / 
                flops_metrics["estimated_gflops"]
            )
        
        return report
    
    def print_efficiency_report(self, report: Dict[str, Any]) -> None:
        """
        Print formatted efficiency report.
        
        Args:
            report: Efficiency report dictionary
        """
        print("\n" + "="*60)
        print(f"EFFICIENCY REPORT: {report['model_name']}")
        print("="*60)
        
        print(f"\nInput Shape: {report['input_shape']}")
        print(f"Batch Size: {report['batch_size']}")
        print(f"Device: {report['device']}")
        
        print(f"\n--- THROUGHPUT ---")
        print(f"Samples per second: {report['throughput_samples_per_sec']:.2f}")
        print(f"Average time per batch: {report['avg_time_per_batch_ms']:.2f} ms")
        
        print(f"\n--- PARAMETERS ---")
        print(f"Total parameters: {report['total_parameters']:,}")
        print(f"Trainable parameters: {report['trainable_parameters']:,}")
        print(f"Non-trainable parameters: {report['non_trainable_parameters']:,}")
        
        print(f"\n--- COMPUTATION ---")
        print(f"Estimated FLOPs: {report['estimated_flops']:,}")
        print(f"Estimated GFLOPs: {report['estimated_gflops']:.3f}")
        
        print(f"\n--- MEMORY ---")
        print(f"Peak memory usage: {report['peak_memory_mb']:.2f} MB")
        print(f"Memory increase: {report['memory_increase_mb']:.2f} MB")
        
        if "throughput_per_million_params" in report:
            print(f"\n--- EFFICIENCY RATIOS ---")
            print(f"Throughput per million params: {report['throughput_per_million_params']:.2f}")
        
        if "throughput_per_gflop" in report:
            print(f"Throughput per GFLOP: {report['throughput_per_gflop']:.2f}")
        
        print("="*60)


def measure_model_efficiency(model: nn.Module,
                           input_shape: Tuple[int, ...],
                           batch_size: int = 1,
                           device: str = "cpu",
                           model_name: str = "Model") -> Dict[str, Any]:
    """
    Convenience function to measure model efficiency.
    
    Args:
        model: Model to measure
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_size: Batch size for measurements
        device: Device to run measurements on
        model_name: Name of the model for reporting
        
    Returns:
        Efficiency report dictionary
    """
    metrics = EfficiencyMetrics(device=device)
    report = metrics.comprehensive_efficiency_report(
        model, input_shape, batch_size, model_name
    )
    metrics.print_efficiency_report(report)
    return report 