#!/usr/bin/env python3
"""
PG-19 dataset test script with command line flags.
Run with: python test_pg19_cli.py [options]
"""

import sys
import os
import json
import argparse
import time
import logging
import torch

# Suppress verbose download bars from HuggingFace datasets
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_pg19_loader(args):
    """Test PG-19 dataloader with parameters."""
    try:
        # Use absolute import to avoid conflicts
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        # Try to import HuggingFace datasets
        try:
            from datasets import load_dataset
            print("✓ HuggingFace datasets library available")
        except ImportError:
            print("✗ HuggingFace datasets library not available")
            print("Please install with: pip install datasets")
            return {"error": "datasets library not available"}
        
        # Import our PG-19 loader
        from tfn.tfn_datasets.pg19_loader import create_pg19_dataloader
        
        print("Testing PG-19 dataloader...")
        
        # Create dataloader
        train_loader, val_loader, vocab_size = create_pg19_dataloader(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_train_chunks=args.max_train_chunks,
            max_val_chunks=args.max_val_chunks,
            vocab_size=args.vocab_size
        )
        
        print(f"Vocabulary size: {vocab_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"Batch shape: {batch.shape}")
            print(f"Batch dtype: {batch.dtype}")
            break
        
        # Measure throughput
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create a simple model for throughput test
        from tfn.model.tfn_base import TrainableTFNLayer
        
        model = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, args.embed_dim),
            TrainableTFNLayer(
                embed_dim=args.embed_dim,
                kernel_type="rbf",
                evolution_type="cnn",
                grid_size=32,
                time_steps=3,
                max_seq_len=args.seq_len
            ),
            torch.nn.Linear(args.embed_dim, vocab_size)
        ).to(device)
        
        # Test throughput
        model.eval()
        for batch in train_loader:
            batch = batch.to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    positions = torch.linspace(0, 1, args.seq_len, device=device).view(1, args.seq_len, 1).expand(args.batch_size, -1, -1)
                    _ = model[1](model[0](batch), positions)
            
            # Measure throughput
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    positions = torch.linspace(0, 1, args.seq_len, device=device).view(1, args.seq_len, 1).expand(args.batch_size, -1, -1)
                    _ = model[1](model[0](batch), positions)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            total_time = end_time - start_time
            tokens_per_second = (args.batch_size * args.seq_len * 10) / total_time
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            results = {
                "vocab_size": vocab_size,
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "tokens_per_second": tokens_per_second,
                "num_params": num_params,
                "success": True
            }
            
            print(f"✓ PG-19 loader test passed")
            print(f"  Throughput: {tokens_per_second:.0f} tokens/s")
            print(f"  Parameters: {num_params:,}")
            return results
            
    except Exception as e:
        print(f"✗ PG-19 loader test failed: {e}")
        return {"error": str(e)}

def main():
    """Main test function with command line arguments."""
    parser = argparse.ArgumentParser(description="PG-19 Dataset Test")
    
    # Dataset parameters
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--vocab-size", type=int, default=50000, help="Vocabulary size")
    parser.add_argument("--max-train-chunks", type=int, default=100, help="Max train chunks")
    parser.add_argument("--max-val-chunks", type=int, default=20, help="Max val chunks")
    
    # Model parameters
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    
    # Output
    parser.add_argument("--output", type=str, default="pg19_test_results.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PG-19 DATASET TEST")
    print("=" * 50)
    
    if args.verbose:
        print(f"Sequence Length: {args.seq_len}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Vocabulary Size: {args.vocab_size}")
        print(f"Max Train Chunks: {args.max_train_chunks}")
        print(f"Max Val Chunks: {args.max_val_chunks}")
        print(f"Embedding Dimension: {args.embed_dim}")
    
    results = test_pg19_loader(args)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {args.output}")
    print("PG-19 test completed!")

if __name__ == "__main__":
    main() 