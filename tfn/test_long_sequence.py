#!/usr/bin/env python3
"""
Test script for long-sequence efficiency.
Run with: python test_long_sequence.py
"""

import sys
import os
import json
import time
import torch

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_long_sequence_efficiency():
    """Test long-sequence efficiency on PG-19."""
    try:
        from datasets.pg19_loader import create_pg19_dataloader, measure_memory_usage
        from tfn.model.seq_baselines import SimpleTransformerSeqModel, SimplePerformerSeqModel
        from tfn.model.tfn_base import TrainableTFNLayer
        
        print("Testing long-sequence efficiency...")
        
        # Test parameters (smaller for faster testing)
        seq_lens = [1024, 2048]
        model_types = ["tfn", "transformer"]
        batch_size = 2
        embed_dim = 128
        num_layers = 2
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        results = {}
        
        for seq_len in seq_lens:
            print(f"\nTesting sequence length: {seq_len}")
            results[seq_len] = {}
            
            for model_type in model_types:
                print(f"  Testing model: {model_type}")
                
                try:
                    # Create dataloader
                    train_loader, val_loader, vocab_size = create_pg19_dataloader(
                        seq_len=seq_len,
                        batch_size=batch_size,
                        max_train_chunks=20,
                        max_val_chunks=5
                    )
                    
                    # Create model
                    if model_type == "tfn":
                        # Create a simple TFN model for testing
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
                            num_layers=num_layers
                        )
                    else:
                        model = SimplePerformerSeqModel(
                            vocab_size=vocab_size,
                            seq_len=seq_len,
                            embed_dim=embed_dim,
                            num_layers=num_layers
                        )
                    
                    model = model.to(device)
                    
                    # Measure performance
                    perf_metrics = measure_memory_usage(model, batch_size, seq_len, vocab_size, device)
                    
                    # Measure throughput
                    model.eval()
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(5):
                            _ = model(input_ids)
                    
                    # Measure throughput
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(20):
                            _ = model(input_ids)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    
                    total_time = end_time - start_time
                    tokens_per_second = (batch_size * seq_len * 20) / total_time
                    
                    # Count parameters
                    num_params = sum(p.numel() for p in model.parameters())
                    
                    results[seq_len][model_type] = {
                        **perf_metrics,
                        "tokens_per_second": tokens_per_second,
                        "num_params": num_params,
                        "success": True
                    }
                    
                    print(f"    Success: {tokens_per_second:.0f} tokens/s, "
                          f"{perf_metrics['gpu_memory_mb']:.1f} MB GPU, "
                          f"{num_params:,} params")
                    
                except Exception as e:
                    results[seq_len][model_type] = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"    Failed: {e}")
        
        print("\n✓ Long-sequence efficiency test completed successfully!")
        return results
        
    except Exception as e:
        print(f"✗ Long-sequence efficiency test failed: {e}")
        return {"error": str(e)}

def main():
    """Main test function."""
    print("=" * 50)
    print("LONG-SEQUENCE EFFICIENCY TEST")
    print("=" * 50)
    
    results = test_long_sequence_efficiency()
    
    # Save results
    with open("long_sequence_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: long_sequence_test_results.json")
    print("Long-sequence efficiency test completed!")

if __name__ == "__main__":
    main() 