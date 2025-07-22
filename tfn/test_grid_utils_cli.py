#!/usr/bin/env python3
"""
Grid utilities test script with command line flags.
Run with: python test_grid_utils_cli.py [options]
"""

import sys
import os
import json
import argparse

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_grid_size_heuristics(args):
    """Test grid size heuristics with parameters."""
    try:
        from tfn.core.grid_utils import compute_auto_grid_size, estimate_memory_usage, estimate_flops
        
        print("Testing grid size heuristics...")
        
        # Test different sequence lengths and embedding dimensions
        seq_lens = [256, 512, 1024, 2048, 4096]
        embed_dims = [128, 256, 512]
        heuristics = ["sqrt", "linear", "log", "adaptive"]
        
        if args.seq_len:
            seq_lens = [args.seq_len]
        if args.embed_dim:
            embed_dims = [args.embed_dim]
        if args.heuristic:
            heuristics = [args.heuristic]
        
        results = {}
        
        for seq_len in seq_lens:
            results[seq_len] = {}
            
            for embed_dim in embed_dims:
                results[seq_len][embed_dim] = {}
                
                for heuristic in heuristics:
                    try:
                        grid_size = compute_auto_grid_size(
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
                        
                        print(f"  seq_len={seq_len}, embed_dim={embed_dim}, heuristic={heuristic}: "
                              f"grid_size={grid_size}, memory={memory_info['total_memory_mb']:.1f}MB, "
                              f"flops={flops_info['flops_per_token']:.0f}")
                        
                    except Exception as e:
                        results[seq_len][embed_dim][heuristic] = {
                            "success": False,
                            "error": str(e)
                        }
                        print(f"  seq_len={seq_len}, embed_dim={embed_dim}, heuristic={heuristic}: FAILED - {e}")
        
        print("\n✓ Grid size heuristics test completed successfully!")
        return results
        
    except Exception as e:
        print(f"✗ Grid size heuristics test failed: {e}")
        return {"error": str(e)}

def main():
    """Main test function with command line arguments."""
    parser = argparse.ArgumentParser(description="Grid Utils Test")
    
    # Test parameters
    parser.add_argument("--seq-len", type=int, help="Specific sequence length to test")
    parser.add_argument("--embed-dim", type=int, help="Specific embedding dimension to test")
    parser.add_argument("--heuristic", choices=["sqrt", "linear", "log", "adaptive"], 
                       help="Specific heuristic to test")
    
    # Output
    parser.add_argument("--output", type=str, default="grid_utils_test_results.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("GRID UTILS TEST")
    print("=" * 50)
    
    if args.verbose:
        print(f"Sequence Length: {args.seq_len or 'all'}")
        print(f"Embedding Dimension: {args.embed_dim or 'all'}")
        print(f"Heuristic: {args.heuristic or 'all'}")
    
    results = test_grid_size_heuristics(args)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    print("Grid utils test completed!")

if __name__ == "__main__":
    main() 