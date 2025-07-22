#!/usr/bin/env python3
"""
Simple test script to verify TFN implementation.

Tests the core components we've implemented:
1. PG-19 dataloader
2. Physics dataloader  
3. Grid size heuristics
4. Basic TFN training
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pg19_loader():
    """Test PG-19 dataloader."""
    logger.info("Testing PG-19 dataloader...")
    
    try:
        from tfn.tfn_datasets.pg19_loader import create_pg19_dataloader
        
        # Create small dataloader for testing
        train_loader, val_loader, vocab_size = create_pg19_dataloader(
            seq_len=512,
            batch_size=2,
            max_train_chunks=10,
            max_val_chunks=5
        )
        
        logger.info(f"Vocabulary size: {vocab_size}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        
        # Test a batch
        for batch in train_loader:
            logger.info(f"Batch shape: {batch.shape}")
            logger.info(f"Batch dtype: {batch.dtype}")
            break
            
        logger.info("✓ PG-19 loader test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ PG-19 loader test failed: {e}")
        return False


def test_physics_loader():
    """Test physics dataloader."""
    logger.info("Testing physics dataloader...")
    
    try:
        from tfn.tfn_datasets.physics_loader import create_physics_dataloader
        
        # Create small dataloader for testing
        train_loader, val_loader = create_physics_dataloader(
            dataset_type="burgers",
            batch_size=4,
            num_samples=50,
            grid_points=64,
            time_steps=30,
            input_steps=5,
            output_steps=10
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        
        # Test a batch
        for batch in train_loader:
            input_seq, target_seq = batch
            logger.info(f"Input shape: {input_seq.shape}")
            logger.info(f"Target shape: {target_seq.shape}")
            break
            
        logger.info("✓ Physics loader test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Physics loader test failed: {e}")
        return False


def test_grid_utils():
    """Test grid size utilities."""
    logger.info("Testing grid size utilities...")
    
    try:
        from tfn.core.grid_utils import compute_auto_grid_size, estimate_memory_usage, estimate_flops
        
        # Test grid size computation
        seq_lens = [256, 512, 1024, 2048]
        embed_dims = [128, 256]
        
        for seq_len in seq_lens:
            for embed_dim in embed_dims:
                grid_size = compute_auto_grid_size(seq_len, embed_dim)
                memory_info = estimate_memory_usage(1, seq_len, grid_size, embed_dim)
                flops_info = estimate_flops(seq_len, grid_size, embed_dim)
                
                logger.info(f"seq_len={seq_len}, embed_dim={embed_dim}: grid_size={grid_size}, "
                          f"memory={memory_info['total_memory_mb']:.1f}MB, "
                          f"flops={flops_info['flops_per_token']:.0f}")
        
        logger.info("✓ Grid utils test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Grid utils test failed: {e}")
        return False


def test_tfn_model():
    """Test TFN model creation and forward pass."""
    logger.info("Testing TFN model...")
    
    try:
        from tfn.model.tfn_base import TrainableTFNLayer
        
        # Create TFN layer
        model = TrainableTFNLayer(
            embed_dim=64,
            kernel_type="rbf",
            evolution_type="cnn",
            grid_size=32,
            time_steps=3,
            max_seq_len=128
        )
        
        # Test forward pass
        batch_size, seq_len, embed_dim = 2, 64, 64
        embeddings = torch.randn(batch_size, seq_len, embed_dim)
        positions = torch.rand(batch_size, seq_len, 1)
        
        outputs = model(embeddings, positions)
        
        logger.info(f"Input shape: {embeddings.shape}")
        logger.info(f"Output shape: {outputs.shape}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        logger.info("✓ TFN model test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TFN model test failed: {e}")
        return False


def test_baseline_models():
    """Test baseline models."""
    logger.info("Testing baseline models...")
    
    try:
        from tfn.model.seq_baselines import SimpleTransformerSeqModel, SimplePerformerSeqModel
        
        # Test Transformer
        transformer = SimpleTransformerSeqModel(
            vocab_size=1000,
            seq_len=256,
            embed_dim=128,
            num_layers=2
        )
        
        # Test Performer
        performer = SimplePerformerSeqModel(
            vocab_size=1000,
            seq_len=256,
            embed_dim=128,
            num_layers=2
        )
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 64))
        
        transformer_output = transformer(input_ids)
        performer_output = performer(input_ids)
        
        logger.info(f"Transformer output shape: {transformer_output.shape}")
        logger.info(f"Performer output shape: {performer_output.shape}")
        
        logger.info("✓ Baseline models test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Baseline models test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting TFN implementation tests...")
    
    tests = [
        ("PG-19 Loader", test_pg19_loader),
        ("Physics Loader", test_physics_loader),
        ("Grid Utils", test_grid_utils),
        ("TFN Model", test_tfn_model),
        ("Baseline Models", test_baseline_models),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! TFN implementation is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    import os
    # Add the current directory to the path so we can import tfn modules
    sys.path.insert(0, os.path.dirname(__file__))
    success = main()
    exit(0 if success else 1) 