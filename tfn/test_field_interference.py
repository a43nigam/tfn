"""
Comprehensive Test Script for Field Interference Mechanisms.

This script tests all components of the field interference implementation:
- Token field interference
- Dynamic field propagation  
- Novel interaction operators
- Enhanced TFN integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import time
import matplotlib.pyplot as plt

# Import our modules
from tfn.core.field_interference import (
    TokenFieldInterference, 
    CausalFieldInterference,
    MultiScaleFieldInterference,
    PhysicsConstrainedInterference,
    create_field_interference
)

from tfn.core.field_evolution import (
    DynamicFieldPropagator,
    AdaptiveFieldPropagator,
    CausalFieldPropagator,
    create_field_evolver
)

from tfn.core.interaction_operators import (
    FieldInteractionOperators,
    FractalInteractionOperators,
    CausalInteractionOperators,
    MetaInteractionOperators,
    create_interaction_operators
)

from tfn.model.tfn_enhanced import EnhancedTFNModel, create_enhanced_tfn_model


def test_field_interference():
    """Test field interference mechanisms."""
    print("🧪 Testing Field Interference Mechanisms...")
    
    # Test parameters
    batch_size = 4
    num_tokens = 32
    embed_dim = 256
    num_heads = 8
    
    # Create test data
    token_fields = torch.randn(batch_size, num_tokens, embed_dim)
    positions = torch.rand(batch_size, num_tokens, 1)
    
    # Test different interference types
    interference_types = ["standard", "causal", "multiscale", "physics"]
    
    for interference_type in interference_types:
        print(f"\n📊 Testing {interference_type} interference...")
        
        # Create interference module
        interference = create_field_interference(
            interference_type=interference_type,
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        # Test forward pass
        start_time = time.time()
        enhanced_fields = interference(token_fields, positions)
        forward_time = time.time() - start_time
        
        # Verify output shape
        assert enhanced_fields.shape == token_fields.shape, f"Shape mismatch: {enhanced_fields.shape} vs {token_fields.shape}"
        
        # Test physics constraints if applicable
        if hasattr(interference, 'get_constraint_loss'):
            constraint_loss = interference.get_constraint_loss()
            print(f"  Physics constraint loss: {constraint_loss.item():.6f}")
        
        print(f"  ✅ {interference_type} interference: {forward_time:.4f}s")
        print(f"  Output shape: {enhanced_fields.shape}")
        print(f"  Output norm: {enhanced_fields.norm().item():.4f}")


def test_dynamic_propagation():
    """Test dynamic field propagation."""
    print("\n🧪 Testing Dynamic Field Propagation...")
    
    # Test parameters
    batch_size = 4
    num_tokens = 32
    embed_dim = 256
    pos_dim = 1
    
    # Create test data
    token_fields = torch.randn(batch_size, num_tokens, embed_dim)
    positions = torch.rand(batch_size, num_tokens, pos_dim)
    grid_points = torch.rand(batch_size, 100, pos_dim)
    
    # Test different propagator types
    propagator_types = ["standard", "adaptive", "causal"]
    evolution_types = ["diffusion", "wave"]
    
    for propagator_type in propagator_types:
        for evolution_type in evolution_types:
            print(f"\n📊 Testing {propagator_type} propagator with {evolution_type} evolution...")
            
            # Set correct interference_type for causal propagator
            if propagator_type == "causal":
                interference_type = "causal"
            else:
                interference_type = "standard"
            
            # Create propagator
            propagator = create_field_evolver(
                propagator_type=propagator_type,
                embed_dim=embed_dim,
                pos_dim=pos_dim,
                evolution_type=evolution_type,
                interference_type=interference_type
            )
            
            # Test forward pass
            start_time = time.time()
            evolved_fields = propagator(token_fields, grid_points=grid_points, time_steps=1, positions=positions)
            forward_time = time.time() - start_time
            
            # Verify output shape
            assert evolved_fields.shape == token_fields.shape, f"Shape mismatch: {evolved_fields.shape} vs {token_fields.shape}"
            
            print(f"  ✅ {propagator_type} + {evolution_type}: {forward_time:.4f}s")
            print(f"  Output shape: {evolved_fields.shape}")
            print(f"  Output norm: {evolved_fields.norm().item():.4f}")


def test_interaction_operators():
    """Test novel interaction operators."""
    print("\n🧪 Testing Novel Interaction Operators...")
    
    # Test parameters
    batch_size = 4
    num_fields = 16
    embed_dim = 256
    num_heads = 8
    
    # Create test data
    fields = torch.randn(batch_size, num_fields, embed_dim)
    positions = torch.rand(batch_size, num_fields, 1)
    
    # Test different operator types
    operator_types = ["standard", "fractal", "causal", "meta"]
    
    for operator_type in operator_types:
        print(f"\n📊 Testing {operator_type} interaction operators...")
        
        # Create operator
        operator = create_interaction_operators(
            operator_type=operator_type,
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        # Test forward pass
        start_time = time.time()
        enhanced_fields = operator(fields, positions)
        forward_time = time.time() - start_time
        
        # Verify output shape
        assert enhanced_fields.shape == fields.shape, f"Shape mismatch: {enhanced_fields.shape} vs {fields.shape}"
        
        print(f"  ✅ {operator_type} operators: {forward_time:.4f}s")
        print(f"  Output shape: {enhanced_fields.shape}")
        print(f"  Output norm: {enhanced_fields.norm().item():.4f}")


def test_enhanced_tfn_integration():
    """Test enhanced TFN integration."""
    print("\n🧪 Testing Enhanced TFN Integration...")
    
    # Test parameters
    vocab_size = 1000
    embed_dim = 256
    num_layers = 2
    batch_size = 4
    seq_len = 32
    
    # Create test data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test different interference types
    interference_types = ["standard", "causal", "multiscale", "physics"]
    
    for interference_type in interference_types:
        print(f"\n📊 Testing Enhanced TFN with {interference_type} interference...")
        
        # Create enhanced TFN model
        model = create_enhanced_tfn_model(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            interference_type=interference_type
        )
        
        # Test forward pass
        start_time = time.time()
        logits = model(input_ids)
        forward_time = time.time() - start_time
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, vocab_size)
        assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} vs {expected_shape}"
        
        # Test physics constraints
        constraints = model.get_physics_constraints()
        if constraints:
            total_constraint_loss = sum(constraints.values())
            print(f"  Total physics constraint loss: {total_constraint_loss.item():.6f}")
        
        print(f"  ✅ Enhanced TFN ({interference_type}): {forward_time:.4f}s")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output norm: {logits.norm().item():.4f}")


def test_performance_benchmark():
    """Benchmark performance of field interference mechanisms."""
    print("\n🧪 Performance Benchmarking...")
    
    # Test configurations
    configs = [
        {"batch_size": 4, "seq_len": 64, "embed_dim": 256},
        {"batch_size": 8, "seq_len": 128, "embed_dim": 512},
        {"batch_size": 16, "seq_len": 256, "embed_dim": 1024},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n📊 Benchmarking config: {config}")
        
        # Create test data
        token_fields = torch.randn(config["batch_size"], config["seq_len"], config["embed_dim"])
        positions = torch.rand(config["batch_size"], config["seq_len"], 1)
        
        # Test different interference types
        for interference_type in ["standard", "causal", "multiscale", "physics"]:
            # Create interference module
            interference = create_field_interference(
                interference_type=interference_type,
                embed_dim=config["embed_dim"],
                num_heads=8
            )
            
            # Warmup
            for _ in range(3):
                _ = interference(token_fields, positions)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(10):
                _ = interference(token_fields, positions)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            throughput = config["batch_size"] * config["seq_len"] / avg_time
            
            key = f"{interference_type}_{config['batch_size']}_{config['seq_len']}_{config['embed_dim']}"
            results[key] = {
                "avg_time": avg_time,
                "throughput": throughput,
                "config": config,
                "interference_type": interference_type
            }
            
            print(f"  {interference_type}: {avg_time:.4f}s ({throughput:.0f} tokens/s)")
    
    return results


def test_physics_constraints():
    """Test physics constraint mechanisms."""
    print("\n🧪 Testing Physics Constraints...")
    
    # Test parameters
    batch_size = 4
    num_tokens = 32
    embed_dim = 256
    
    # Create test data
    token_fields = torch.randn(batch_size, num_tokens, embed_dim)
    positions = torch.rand(batch_size, num_tokens, 1)
    
    # Test physics-constrained interference
    physics_interference = PhysicsConstrainedInterference(
        embed_dim=embed_dim,
        energy_weight=0.1,
        symmetry_weight=0.1
    )
    
    # Test forward pass
    enhanced_fields = physics_interference(token_fields, positions)
    
    # Get constraint losses
    constraint_losses = physics_interference.get_constraint_loss()
    
    print(f"  Energy conservation loss: {constraint_losses.item():.6f}")
    print(f"  Original field norm: {token_fields.norm().item():.4f}")
    print(f"  Enhanced field norm: {enhanced_fields.norm().item():.4f}")
    
    # Test that constraints are reasonable
    assert constraint_losses.item() >= 0, "Constraint loss should be non-negative"
    assert enhanced_fields.shape == token_fields.shape, "Shape should be preserved"


def test_causal_mechanisms():
    """Test causal mechanisms for time-series applications."""
    print("\n🧪 Testing Causal Mechanisms...")
    
    # Test parameters
    batch_size = 4
    num_tokens = 32
    embed_dim = 256
    
    # Create test data
    token_fields = torch.randn(batch_size, num_tokens, embed_dim)
    positions = torch.rand(batch_size, num_tokens, 1)
    
    # Test causal interference
    causal_interference = CausalFieldInterference(embed_dim=embed_dim)
    causal_enhanced = causal_interference(token_fields, positions)
    
    # Test causal propagator
    causal_propagator = CausalFieldPropagator(embed_dim=embed_dim, pos_dim=1)
    causal_propagated = causal_propagator(token_fields, positions)
    
    # Test causal operators
    causal_operators = CausalInteractionOperators(embed_dim=embed_dim)
    causal_operated = causal_operators(token_fields, positions)
    
    print(f"  Causal interference shape: {causal_enhanced.shape}")
    print(f"  Causal propagation shape: {causal_propagated.shape}")
    print(f"  Causal operators shape: {causal_operated.shape}")
    
    # Verify all outputs have correct shape
    assert causal_enhanced.shape == token_fields.shape
    assert causal_propagated.shape == token_fields.shape
    assert causal_operated.shape == token_fields.shape


def test_multi_scale_mechanisms():
    """Test multi-scale mechanisms."""
    print("\n🧪 Testing Multi-Scale Mechanisms...")
    
    # Test parameters
    batch_size = 4
    num_tokens = 64
    embed_dim = 256
    
    # Create test data
    token_fields = torch.randn(batch_size, num_tokens, embed_dim)
    positions = torch.rand(batch_size, num_tokens, 1)
    
    # Test multi-scale interference
    multiscale_interference = MultiScaleFieldInterference(
        embed_dim=embed_dim,
        scales=4
    )
    multiscale_enhanced = multiscale_interference(token_fields, positions)
    
    # Test fractal operators
    fractal_operators = FractalInteractionOperators(
        embed_dim=embed_dim,
        scales=4
    )
    fractal_enhanced = fractal_operators(token_fields, positions)
    
    print(f"  Multi-scale interference shape: {multiscale_enhanced.shape}")
    print(f"  Fractal operators shape: {fractal_enhanced.shape}")
    
    # Verify outputs have correct shape
    assert multiscale_enhanced.shape == token_fields.shape
    assert fractal_enhanced.shape == token_fields.shape


def test_token_field_interference_gradient():
    """Test that TokenFieldInterference is differentiable and gradients flow to input."""
    embed_dim = 32
    batch_size = 2
    num_tokens = 8
    x = torch.randn(batch_size, num_tokens, embed_dim, requires_grad=True)
    module = TokenFieldInterference(embed_dim=embed_dim)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow to input!"
    assert torch.isfinite(x.grad).all(), "Gradient contains non-finite values!"
    print("\u2705 TokenFieldInterference gradient flow test passed.")

def test_dynamic_field_propagator_gradient():
    """Test that DynamicFieldPropagator is differentiable and gradients flow to input."""
    embed_dim = 32
    pos_dim = 1
    batch_size = 2
    num_tokens = 8
    x = torch.randn(batch_size, num_tokens, embed_dim, requires_grad=True)
    positions = torch.rand(batch_size, num_tokens, pos_dim)
    module = DynamicFieldPropagator(embed_dim=embed_dim, pos_dim=pos_dim)
    out = module(x, positions)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow to input!"
    assert torch.isfinite(x.grad).all(), "Gradient contains non-finite values!"
    print("\u2705 DynamicFieldPropagator gradient flow test passed.")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive Field Interference Tests")
    print("=" * 60)
    
    try:
        # Test core mechanisms
        test_field_interference()
        test_dynamic_propagation()
        test_interaction_operators()
        
        # Test integration
        test_enhanced_tfn_integration()
        
        # Test specialized mechanisms
        test_physics_constraints()
        test_causal_mechanisms()
        test_multi_scale_mechanisms()
        
        # Performance benchmarking
        results = test_performance_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
        # Print summary
        print("\n📊 Performance Summary:")
        for key, result in results.items():
            print(f"  {key}: {result['avg_time']:.4f}s ({result['throughput']:.0f} tokens/s)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run tests
    success = run_comprehensive_tests()
    
    if success:
        print("\n🎉 Field interference implementation is working correctly!")
    else:
        print("\n💥 Field interference implementation has issues that need to be fixed.") 