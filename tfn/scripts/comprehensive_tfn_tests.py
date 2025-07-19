#!/usr/bin/env python3
"""
Comprehensive TFN Testing Script

Runs all TFN tests discussed in the strategy:
1. Long-sequence efficiency tests (PG-19)
2. Physics PDE evolution tests
3. Multimodal tests
4. Robustness and transfer tests
"""

import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Import TFN components
from ..datasets.pg19_loader import create_pg19_dataloader, compute_perplexity, measure_memory_usage
from ..datasets.physics_loader import create_physics_dataloader, compute_pde_metrics, visualize_pde_prediction
from ..model.tfn_base import TrainableTFNLayer
from ..model.seq_baselines import TransformerEncoder, PerformerEncoder
from ..core.grid_utils import compute_auto_grid_size, estimate_memory_usage, estimate_flops


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class TFNPhysicsModel(nn.Module):
    """TFN model for physics PDE evolution."""
    
    def __init__(
        self,
        input_steps: int = 10,
        output_steps: int = 40,
        grid_points: int = 128,
        embed_dim: int = 64,
        num_layers: int = 4,
        kernel_type: str = "rbf",
        evolution_type: str = "pde",
        grid_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.grid_points = grid_points
        self.embed_dim = embed_dim
        
        # Auto-compute grid size if not specified
        if grid_size is None:
            grid_size, _ = compute_auto_grid_size(grid_points, embed_dim)
        
        self.grid_size = int(grid_size)
        
        # Input projection
        self.input_projection = nn.Linear(1, embed_dim)
        
        # TFN layers
        self.tfn_layers = nn.ModuleList([
            TrainableTFNLayer(
                embed_dim=embed_dim,
                kernel_type=kernel_type,
                evolution_type=evolution_type,
                grid_size=self.grid_size,
                time_steps=3,
                max_seq_len=input_steps,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.input_projection.weight, 0, 0.1)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.output_projection.weight, 0, 0.1)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass for PDE evolution.
        
        Args:
            input_seq: Input sequence [batch, input_steps, grid_points]
        
        Returns:
            Predicted sequence [batch, output_steps, grid_points]
        """
        B, T, N = input_seq.shape
        
        # Reshape to [batch * input_steps, grid_points, 1]
        input_reshaped = input_seq.unsqueeze(-1).view(B * T, N, 1)
        
        # Project to embeddings
        embeddings = self.input_projection(input_reshaped)  # [B*T, N, D]
        
        # Create positions (normalized to [0, 1])
        positions = torch.arange(N, device=input_seq.device, dtype=torch.float32)
        positions = positions.unsqueeze(0).expand(B * T, -1).unsqueeze(-1) / float(N - 1)
        
        # TFN layers
        hidden_states = embeddings
        for layer in self.tfn_layers:
            hidden_states = layer(hidden_states, positions)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Output projection
        outputs = self.output_projection(hidden_states)  # [B*T, N, 1]
        
        # Reshape back to [batch, input_steps, grid_points]
        outputs = outputs.squeeze(-1).view(B, T, N)
        
        # For now, just repeat the last timestep for output_steps
        # In a full implementation, this would be autoregressive
        last_output = outputs[:, -1:, :]  # [B, 1, N]
        predictions = last_output.expand(-1, self.output_steps, -1)  # [B, output_steps, N]
        
        return predictions


def test_long_sequence_efficiency(
    seq_lens: List[int] = [1024, 2048, 4096, 8192],
    model_types: List[str] = ["tfn", "transformer", "performer"],
    batch_size: int = 4,
    embed_dim: int = 256,
    num_layers: int = 4,
    device: torch.device = None
) -> Dict[str, Any]:
    """Test long-sequence efficiency on PG-19."""
    logger = logging.getLogger(__name__)
    logger.info("Testing long-sequence efficiency...")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}
    
    for seq_len in seq_lens:
        logger.info(f"Testing sequence length: {seq_len}")
        results[seq_len] = {}
        
        for model_type in model_types:
            logger.info(f"  Testing model: {model_type}")
            
            try:
                # Create dataloader
                train_loader, val_loader, vocab_size = create_pg19_dataloader(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    max_train_chunks=100,
                    max_val_chunks=20
                )
                
                # Create model
                if model_type == "tfn":
                    from ..scripts.train_pg19 import TFNLanguageModel
                    model = TFNLanguageModel(
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        num_layers=num_layers,
                        seq_len=seq_len
                    )
                elif model_type == "transformer":
                    model = TransformerEncoder(
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        num_layers=num_layers,
                        max_seq_len=seq_len
                    )
                elif model_type == "performer":
                    model = PerformerEncoder(
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        num_layers=num_layers,
                        max_seq_len=seq_len
                    )
                
                model = model.to(device)
                
                # Measure performance
                perf_metrics = measure_memory_usage(model, batch_size, seq_len, vocab_size, device)
                
                # Measure throughput
                model.eval()
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(input_ids)
                
                # Measure throughput
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(50):
                        _ = model(input_ids)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                total_time = end_time - start_time
                tokens_per_second = (batch_size * seq_len * 50) / total_time
                
                # Count parameters
                num_params = sum(p.numel() for p in model.parameters())
                
                results[seq_len][model_type] = {
                    **perf_metrics,
                    "tokens_per_second": tokens_per_second,
                    "num_params": num_params,
                    "success": True
                }
                
                logger.info(f"    Success: {tokens_per_second:.0f} tokens/s, {perf_metrics['gpu_memory_mb']:.1f} MB GPU")
                
            except Exception as e:
                logger.error(f"    Failed: {e}")
                results[seq_len][model_type] = {
                    "success": False,
                    "error": str(e)
                }
    
    return results


def test_physics_pde_evolution(
    pde_types: List[str] = ["burgers", "wave", "heat"],
    grid_points: List[int] = [64, 128, 256],
    model_types: List[str] = ["tfn", "transformer"],
    batch_size: int = 8,
    embed_dim: int = 64,
    num_layers: int = 4,
    device: torch.device = None
) -> Dict[str, Any]:
    """Test physics PDE evolution."""
    logger = logging.getLogger(__name__)
    logger.info("Testing physics PDE evolution...")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}
    
    for pde_type in pde_types:
        logger.info(f"Testing PDE type: {pde_type}")
        results[pde_type] = {}
        
        for grid_point in grid_points:
            logger.info(f"  Testing grid points: {grid_point}")
            results[pde_type][grid_point] = {}
            
            for model_type in model_types:
                logger.info(f"    Testing model: {model_type}")
                
                try:
                    # Create dataloader
                    train_loader, val_loader = create_physics_dataloader(
                        dataset_type=pde_type,
                        batch_size=batch_size,
                        grid_points=grid_point,
                        num_samples=200,
                        time_steps=50,
                        input_steps=10,
                        output_steps=20
                    )
                    
                    # Create model
                    if model_type == "tfn":
                        model = TFNPhysicsModel(
                            input_steps=10,
                            output_steps=20,
                            grid_points=grid_point,
                            embed_dim=embed_dim,
                            num_layers=num_layers,
                            kernel_type="rbf",
                            evolution_type="pde"
                        )
                    else:
                        # For transformer, we'll use a simple sequence model
                        model = nn.Sequential(
                            nn.Linear(grid_point, embed_dim),
                            nn.ReLU(),
                            nn.Linear(embed_dim, embed_dim),
                            nn.ReLU(),
                            nn.Linear(embed_dim, grid_point * 20)  # output_steps * grid_points
                        )
                    
                    model = model.to(device)
                    
                    # Test on a batch
                    model.eval()
                    for batch in train_loader:
                        input_seq, target_seq = batch
                        input_seq = input_seq.to(device)
                        target_seq = target_seq.to(device)
                        
                        with torch.no_grad():
                            if model_type == "tfn":
                                pred_seq = model(input_seq)
                            else:
                                # Reshape for transformer
                                B, T, N = input_seq.shape
                                input_flat = input_seq.view(B, T * N)
                                output_flat = model(input_flat)
                                pred_seq = output_flat.view(B, 20, N)  # output_steps=20
                        
                        # Compute metrics
                        metrics = compute_pde_metrics(pred_seq, target_seq)
                        
                        # Measure memory and throughput
                        memory_info = measure_memory_usage(model, batch_size, grid_point, 1, device)
                        
                        results[pde_type][grid_point][model_type] = {
                            **metrics,
                            **memory_info,
                            "success": True
                        }
                        
                        logger.info(f"      Success: MSE={metrics['mse']:.4f}, GPU={memory_info['gpu_memory_mb']:.1f} MB")
                        break
                        
                except Exception as e:
                    logger.error(f"      Failed: {e}")
                    results[pde_type][grid_point][model_type] = {
                        "success": False,
                        "error": str(e)
                    }
    
    return results


def test_grid_size_heuristics(
    seq_lens: List[int] = [256, 512, 1024, 2048, 4096, 8192],
    embed_dims: List[int] = [128, 256, 512],
    heuristics: List[str] = ["sqrt", "linear", "log", "adaptive"]
) -> Dict[str, Any]:
    """Test grid size heuristics."""
    logger = logging.getLogger(__name__)
    logger.info("Testing grid size heuristics...")
    
    results = {}
    
    for seq_len in seq_lens:
        results[seq_len] = {}
        
        for embed_dim in embed_dims:
            results[seq_len][embed_dim] = {}
            
            for heuristic in heuristics:
                try:
                    grid_size, info = compute_auto_grid_size(
                        seq_len=seq_len,
                        embed_dim=embed_dim,
                        heuristic=heuristic
                    )
                    
                    # Estimate memory and FLOPs
                    memory_info = estimate_memory_usage(1, seq_len, grid_size, embed_dim)
                    flops_info = estimate_flops(seq_len, grid_size, embed_dim)
                    
                    results[seq_len][embed_dim][heuristic] = {
                        "grid_size": grid_size,
                        "memory_mb": memory_info["total_memory_mb"],
                        "flops_per_token": flops_info["flops_per_token"],
                        "success": True
                    }
                    
                except Exception as e:
                    results[seq_len][embed_dim][heuristic] = {
                        "success": False,
                        "error": str(e)
                    }
    
    return results


def test_multimodal_capabilities(
    device: torch.device = None
) -> Dict[str, Any]:
    """Test multimodal capabilities (placeholder for now)."""
    logger = logging.getLogger(__name__)
    logger.info("Testing multimodal capabilities...")
    
    # This is a placeholder for multimodal tests
    # In a full implementation, this would test:
    # - Text + Vision (CLIP-style)
    # - Audio + Text
    # - 3D Point Clouds
    
    results = {
        "text_vision": {"status": "not_implemented", "message": "CLIP-style multimodal training not yet implemented"},
        "audio_text": {"status": "not_implemented", "message": "Audio-text multimodal not yet implemented"},
        "point_clouds": {"status": "not_implemented", "message": "3D point cloud extension not yet implemented"}
    }
    
    return results


def test_robustness_and_transfer(
    device: torch.device = None
) -> Dict[str, Any]:
    """Test robustness and transfer learning capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("Testing robustness and transfer learning...")
    
    # This is a placeholder for robustness tests
    # In a full implementation, this would test:
    # - Domain transfer (books -> IMDB)
    # - Adversarial robustness
    # - Few-shot learning
    
    results = {
        "domain_transfer": {"status": "not_implemented", "message": "Domain transfer tests not yet implemented"},
        "adversarial_robustness": {"status": "not_implemented", "message": "Adversarial robustness tests not yet implemented"},
        "few_shot_learning": {"status": "not_implemented", "message": "Few-shot learning tests not yet implemented"}
    }
    
    return results


def run_comprehensive_tests(
    output_dir: str = "tfn_test_results",
    device: str = "auto",
    wandb_project: str = "tfn-comprehensive-tests",
    wandb_run_name: str = None
) -> Dict[str, Any]:
    """Run comprehensive TFN tests."""
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive TFN tests...")
    
    # Setup
    setup_logging()
    
    # Device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # WandB logging
    if wandb_run_name is None:
        wandb_run_name = f"comprehensive-tests-{int(time.time())}"
    
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={"device": str(device)}
    )
    
    all_results = {}
    
    # Test 1: Long-sequence efficiency
    logger.info("=" * 50)
    logger.info("TEST 1: Long-sequence efficiency")
    logger.info("=" * 50)
    
    long_seq_results = test_long_sequence_efficiency(device=device)
    all_results["long_sequence_efficiency"] = long_seq_results
    
    # Test 2: Physics PDE evolution
    logger.info("=" * 50)
    logger.info("TEST 2: Physics PDE evolution")
    logger.info("=" * 50)
    
    physics_results = test_physics_pde_evolution(device=device)
    all_results["physics_pde_evolution"] = physics_results
    
    # Test 3: Grid size heuristics
    logger.info("=" * 50)
    logger.info("TEST 3: Grid size heuristics")
    logger.info("=" * 50)
    
    grid_results = test_grid_size_heuristics()
    all_results["grid_size_heuristics"] = grid_results
    
    # Test 4: Multimodal capabilities
    logger.info("=" * 50)
    logger.info("TEST 4: Multimodal capabilities")
    logger.info("=" * 50)
    
    multimodal_results = test_multimodal_capabilities(device=device)
    all_results["multimodal_capabilities"] = multimodal_results
    
    # Test 5: Robustness and transfer
    logger.info("=" * 50)
    logger.info("TEST 5: Robustness and transfer")
    logger.info("=" * 50)
    
    robustness_results = test_robustness_and_transfer(device=device)
    all_results["robustness_and_transfer"] = robustness_results
    
    # Save results
    results_file = output_path / "comprehensive_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Log to WandB
    wandb.log({"all_results": all_results})
    wandb.finish()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    for test_name, results in all_results.items():
        logger.info(f"{test_name}: {len(results)} subtests completed")
    
    logger.info("Comprehensive tests completed!")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive TFN tests")
    parser.add_argument("--output_dir", type=str, default="tfn_test_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--wandb_project", type=str, default="tfn-comprehensive-tests",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    args = parser.parse_args()
    
    results = run_comprehensive_tests(
        output_dir=args.output_dir,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
    
    print("Comprehensive TFN tests completed successfully!")


if __name__ == "__main__":
    import sys
    import os
    # Add the parent directory to the path so we can import tfn modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    main() 