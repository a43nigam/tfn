#!/usr/bin/env python3
"""
Simple models test script with command line flags.
Run with: python test_simple_models_cli.py [options]
"""

import sys
import os
import json
import argparse
import time
import torch

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_simple_models(args):
    """Test simple model creation and forward pass with parameters."""
    try:
        from tfn.model.tfn_base import TrainableTFNLayer
        from tfn.model.seq_baselines import SimpleTransformerSeqModel, SimplePerformerSeqModel
        
        print("Testing simple models...")
        
        # Test parameters
        seq_lens = [256, 512, 1024]
        model_types = ["tfn", "transformer", "performer"]
        batch_size = args.batch_size
        embed_dim = args.embed_dim
        vocab_size = args.vocab_size
        
        if args.seq_len:
            seq_lens = [args.seq_len]
        if args.model_type:
            model_types = [args.model_type]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        results = {}
        
        for seq_len in seq_lens:
            print(f"\nTesting sequence length: {seq_len}")
            results[seq_len] = {}
            
            for model_type in model_types:
                print(f"  Testing model: {model_type}")
                
                try:
                    # Create model
                    if model_type == "tfn":
                        model = torch.nn.Sequential(
                            torch.nn.Embedding(vocab_size, embed_dim),
                            TrainableTFNLayer(
                                embed_dim=embed_dim,
                                kernel_type="rbf",
                                evolution_type="cnn",
                                grid_size=32,
                                time_steps=3,
                                max_seq_len=seq_len
                            ),
                            torch.nn.Linear(embed_dim, vocab_size)
                        )
                    elif model_type == "transformer":
                        model = SimpleTransformerSeqModel(
                            vocab_size=vocab_size,
                            seq_len=seq_len,
                            embed_dim=embed_dim,
                            num_layers=args.num_layers
                        )
                    else:
                        model = SimplePerformerSeqModel(
                            vocab_size=vocab_size,
                            seq_len=seq_len,
                            embed_dim=embed_dim,
                            num_layers=args.num_layers
                        )
                    
                    model = model.to(device)
                    
                    # Test forward pass
                    model.eval()
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(3):
                            if model_type == "tfn":
                                # TFN needs positions
                                positions = torch.linspace(0, 1, seq_len, device=device).view(1, seq_len, 1).expand(batch_size, -1, -1)
                                _ = model[1](model[0](input_ids), positions)  # Just the TFN layer
                            else:
                                _ = model(input_ids)
                    
                    # Measure throughput
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(10):
                            if model_type == "tfn":
                                positions = torch.linspace(0, 1, seq_len, device=device).view(1, seq_len, 1).expand(batch_size, -1, -1)
                                _ = model[1](model[0](input_ids), positions)
                            else:
                                _ = model(input_ids)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    
                    total_time = end_time - start_time
                    tokens_per_second = (batch_size * seq_len * 10) / total_time
                    
                    # Count parameters
                    num_params = sum(p.numel() for p in model.parameters())
                    
                    results[seq_len][model_type] = {
                        "tokens_per_second": tokens_per_second,
                        "num_params": num_params,
                        "success": True
                    }
                    
                    print(f"    Success: {tokens_per_second:.0f} tokens/s, {num_params:,} params")
                    
                except Exception as e:
                    results[seq_len][model_type] = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"    Failed: {e}")
        
        print("\n✓ Simple models test completed successfully!")
        return results
        
    except Exception as e:
        print(f"✗ Simple models test failed: {e}")
        return {"error": str(e)}

def main():
    """Main test function with command line arguments."""
    parser = argparse.ArgumentParser(description="Simple Models Test")
    
    # Model parameters
    parser.add_argument("--seq-len", type=int, help="Specific sequence length to test")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--model-type", choices=["tfn", "transformer", "performer"], 
                       help="Specific model type to test")
    
    # Output
    parser.add_argument("--output", type=str, default="simple_models_test_results.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("SIMPLE MODELS TEST")
    print("=" * 50)
    
    if args.verbose:
        print(f"Sequence Length: {args.seq_len or 'all'}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Embedding Dimension: {args.embed_dim}")
        print(f"Number of Layers: {args.num_layers}")
        print(f"Vocabulary Size: {args.vocab_size}")
        print(f"Model Type: {args.model_type or 'all'}")
    
    results = test_simple_models(args)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {args.output}")
    print("Simple models test completed!")

if __name__ == "__main__":
    main() 