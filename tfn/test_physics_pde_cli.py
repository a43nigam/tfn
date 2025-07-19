#!/usr/bin/env python3
"""
Physics PDE evolution test script with command line flags.
Run with: python test_physics_pde_cli.py [options]
"""

import sys
import os
import json
import argparse
import torch

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_physics_pde_evolution(args):
    """Test physics PDE evolution with parameters."""
    try:
        from tfn_datasets.physics_loader import create_physics_dataloader, compute_pde_metrics
        from model.tfn_base import TrainableTFNLayer
        
        print("Testing physics PDE evolution...")
        
        # Test parameters
        pde_types = ["burgers", "wave", "heat"]
        grid_points = [64, 128]
        model_types = ["tfn", "transformer"]
        batch_size = args.batch_size
        embed_dim = args.embed_dim
        num_layers = args.num_layers
        
        if args.pde_type:
            pde_types = [args.pde_type]
        if args.grid_points:
            grid_points = [args.grid_points]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        results = {}
        
        for pde_type in pde_types:
            print(f"\nTesting PDE type: {pde_type}")
            results[pde_type] = {}
            
            for grid_point in grid_points:
                print(f"  Testing grid points: {grid_point}")
                results[pde_type][grid_point] = {}
                
                for model_type in model_types:
                    print(f"    Testing model: {model_type}")
                    
                    try:
                        # Create dataloader
                        train_loader, val_loader = create_physics_dataloader(
                            dataset_type=pde_type,
                            batch_size=batch_size,
                            grid_points=grid_point,
                            num_samples=50,
                            time_steps=30,
                            input_steps=5,
                            output_steps=10
                        )
                        
                        # Create model
                        if model_type == "tfn":
                            model = TrainableTFNLayer(
                                embed_dim=embed_dim,
                                kernel_type="rbf",
                                evolution_type="cnn",
                                grid_size=grid_point,
                                time_steps=3,
                                max_seq_len=grid_point
                            )
                        else:
                            # Simple MLP baseline
                            model = torch.nn.Sequential(
                                torch.nn.Linear(grid_point * 5, embed_dim),  # input_steps * grid_points
                                torch.nn.ReLU(),
                                torch.nn.Linear(embed_dim, embed_dim),
                                torch.nn.ReLU(),
                                torch.nn.Linear(embed_dim, grid_point * 10)  # output_steps * grid_points
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
                                    # TFN expects positions as well
                                    B, T, N = input_seq.shape
                                    positions = torch.linspace(0, 1, N, device=input_seq.device).view(1, 1, N).expand(B, T, -1)
                                    pred_seq = model(input_seq, positions)
                                else:
                                    # Reshape for MLP
                                    B, T, N = input_seq.shape
                                    input_flat = input_seq.view(B, T * N)
                                    output_flat = model(input_flat)
                                    pred_seq = output_flat.view(B, 10, N)  # output_steps=10
                            
                            # Compute metrics
                            metrics = compute_pde_metrics(pred_seq, target_seq)
                            
                            # Count parameters
                            num_params = sum(p.numel() for p in model.parameters())
                            
                            results[pde_type][grid_point][model_type] = {
                                **metrics,
                                "num_params": num_params,
                                "success": True
                            }
                            
                            print(f"      Success: MSE={metrics['mse']:.4f}, "
                                  f"MAE={metrics['mae']:.4f}, params={num_params:,}")
                            break
                            
                    except Exception as e:
                        results[pde_type][grid_point][model_type] = {
                            "success": False,
                            "error": str(e)
                        }
                        print(f"      Failed: {e}")
        
        print("\n✓ Physics PDE evolution test completed successfully!")
        return results
        
    except Exception as e:
        print(f"✗ Physics PDE evolution test failed: {e}")
        return {"error": str(e)}

def main():
    """Main test function with command line arguments."""
    parser = argparse.ArgumentParser(description="Physics PDE Test")
    
    # Model parameters
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    
    # Physics parameters
    parser.add_argument("--pde-type", choices=["burgers", "wave", "heat"], 
                       help="Specific PDE type to test")
    parser.add_argument("--grid-points", type=int, help="Specific grid points to test")
    
    # Output
    parser.add_argument("--output", type=str, default="physics_pde_test_results.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PHYSICS PDE EVOLUTION TEST")
    print("=" * 50)
    
    if args.verbose:
        print(f"Batch Size: {args.batch_size}")
        print(f"Embedding Dimension: {args.embed_dim}")
        print(f"Number of Layers: {args.num_layers}")
        print(f"PDE Type: {args.pde_type or 'all'}")
        print(f"Grid Points: {args.grid_points or 'all'}")
    
    results = test_physics_pde_evolution(args)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {args.output}")
    print("Physics PDE test completed!")

if __name__ == "__main__":
    main() 