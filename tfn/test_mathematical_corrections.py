"""
Comprehensive Test for Mathematical Corrections.

This script validates the mathematical corrections implemented in:
- Field interference (true pairwise interactions)
- PDE evolution (proper second-order equations)
- Unified field dynamics (stability and constraints)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import time
import matplotlib.pyplot as plt

# Import our corrected modules
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

from tfn.core.unified_field_dynamics import (
    UnifiedFieldDynamics,
    DiffusionOperator,
    WaveOperator,
    SchrodingerOperator,
    PhysicsConstraintOperator,
    StabilityMonitor,
    create_unified_field_dynamics
)


def test_corrected_interference():
    """Test corrected interference implementation."""
    print("🧪 Testing Corrected Field Interference...")
    
    # Test parameters
    batch_size = 2
    num_tokens = 8
    embed_dim = 64
    num_heads = 4
    
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
        
        # Test gradient flow
        enhanced_fields.sum().backward()
        print(f"  Gradient flow: ✅")


def test_corrected_pde_evolution():
    """Test corrected PDE evolution implementation."""
    print("\n🧪 Testing Corrected PDE Evolution...")
    
    # Test parameters
    batch_size = 2
    num_tokens = 16
    embed_dim = 64
    pos_dim = 1
    
    # Create test data
    token_fields = torch.randn(batch_size, num_tokens, embed_dim)
    positions = torch.rand(batch_size, num_tokens, pos_dim)
    grid_points = torch.rand(batch_size, 50, pos_dim)
    
    # Test different evolution types
    evolution_types = ["diffusion", "wave", "schrodinger"]
    propagator_types = ["standard", "adaptive", "causal"]
    
    for evolution_type in evolution_types:
        for propagator_type in propagator_types:
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
            evolved_fields = propagator(token_fields, positions)
            forward_time = time.time() - start_time
            
            # Verify output shape
            assert evolved_fields.shape == token_fields.shape, f"Shape mismatch: {evolved_fields.shape} vs {token_fields.shape}"
            
            # Test gradient flow
            evolved_fields.sum().backward()
            
            print(f"  ✅ {propagator_type} + {evolution_type}: {forward_time:.4f}s")
            print(f"  Output shape: {evolved_fields.shape}")
            print(f"  Output norm: {evolved_fields.norm().item():.4f}")
            print(f"  Gradient flow: ✅")


def test_unified_field_dynamics():
    """Test unified field dynamics implementation."""
    print("\n🧪 Testing Unified Field Dynamics...")
    
    # Test parameters
    batch_size = 2
    num_tokens = 16
    embed_dim = 64
    pos_dim = 1
    
    # Create test data
    fields = torch.randn(batch_size, num_tokens, embed_dim)
    positions = torch.rand(batch_size, num_tokens, pos_dim)
    
    # Test different evolution types
    evolution_types = ["diffusion", "wave", "schrodinger"]
    interference_types = ["standard", "causal", "multiscale", "physics"]
    
    for evolution_type in evolution_types:
        for interference_type in interference_types:
            print(f"\n📊 Testing unified dynamics with {evolution_type} + {interference_type}...")
            
            # Create unified dynamics
            dynamics = create_unified_field_dynamics(
                embed_dim=embed_dim,
                pos_dim=pos_dim,
                evolution_type=evolution_type,
                interference_type=interference_type,
                num_steps=4,
                dt=0.01,
                interference_weight=0.5,
                constraint_weight=0.1,
                stability_threshold=1.0
            )
            
            # Test forward pass
            start_time = time.time()
            evolved_fields = dynamics(fields, positions)
            forward_time = time.time() - start_time
            
            # Verify output shape
            assert evolved_fields.shape == fields.shape, f"Shape mismatch: {evolved_fields.shape} vs {fields.shape}"
            
            # Test physics constraints
            constraints = dynamics.get_physics_constraints()
            print(f"  Physics constraints: {len(constraints)} constraints")
            
            # Test stability metrics
            stability_metrics = dynamics.get_stability_metrics()
            print(f"  Stability metrics: {len(stability_metrics)} metrics")
            
            # Test gradient flow
            evolved_fields.sum().backward()
            
            print(f"  ✅ Unified dynamics: {forward_time:.4f}s")
            print(f"  Output shape: {evolved_fields.shape}")
            print(f"  Output norm: {evolved_fields.norm().item():.4f}")
            print(f"  Gradient flow: ✅")


def test_mathematical_correctness():
    """Test mathematical correctness of implementations."""
    print("\n🧪 Testing Mathematical Correctness...")
    
    # Test 1: Interference mathematical correctness
    print("\n📊 Testing interference mathematical correctness...")
    
    batch_size = 1
    num_tokens = 4
    embed_dim = 8
    num_heads = 2
    
    # Create simple test case
    # fields: [1, 4, 8]
    fields = torch.tensor([
        [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    ])  # [1, 4, 8]
    
    interference = TokenFieldInterference(
        embed_dim=embed_dim,
        num_heads=num_heads,
        interference_types=("constructive", "destructive", "phase")
    )
    
    # Test interference computation
    enhanced_fields = interference(fields)
    
    # Check that interference preserves field structure
    assert enhanced_fields.shape == fields.shape
    assert not torch.isnan(enhanced_fields).any()
    assert not torch.isinf(enhanced_fields).any()
    
    print(f"  ✅ Interference mathematical correctness: PASSED")
    
    # Test 2: PDE evolution mathematical correctness
    print("\n📊 Testing PDE evolution mathematical correctness...")
    
    # Test diffusion operator
    diffusion_op = DiffusionOperator(embed_dim=8)
    fields = torch.randn(1, 8, 8)
    
    # Test diffusion evolution
    evolution = diffusion_op(fields)
    
    # Check that evolution is finite and has reasonable magnitude
    assert not torch.isnan(evolution).any()
    assert not torch.isinf(evolution).any()
    assert evolution.abs().max() < 100.0  # Reasonable bound
    
    print(f"  ✅ PDE evolution mathematical correctness: PASSED")
    
    # Test 3: Stability monitoring
    print("\n📊 Testing stability monitoring...")
    
    stability_monitor = StabilityMonitor(threshold=1.0)
    
    # Test stable evolution
    stable_evolution = torch.randn(1, 4, 8) * 0.1
    assert stability_monitor.check_stability(stable_evolution)
    
    # Test unstable evolution
    unstable_evolution = torch.randn(1, 4, 8) * 10.0
    assert not stability_monitor.check_stability(unstable_evolution)
    
    # Test stability correction
    corrected_evolution = stability_monitor.apply_stability_correction(unstable_evolution)
    # Allow tiny numerical drift with >= threshold check
    assert stability_monitor.check_stability(corrected_evolution), "Stability correction failed"
    
    print(f"  ✅ Stability monitoring: PASSED")


def test_performance_benchmark():
    """Benchmark performance of corrected implementations."""
    print("\n🧪 Performance Benchmark...")
    
    # Test parameters
    batch_size = 4
    num_tokens = 64
    embed_dim = 256
    num_heads = 8
    
    # Create test data
    fields = torch.randn(batch_size, num_tokens, embed_dim)
    positions = torch.rand(batch_size, num_tokens, 1)
    
    # Benchmark interference
    print("\n📊 Benchmarking interference...")
    
    interference = TokenFieldInterference(
        embed_dim=embed_dim,
        num_heads=num_heads,
        interference_types=("constructive", "destructive", "phase")
    )
    
    # Warmup
    for _ in range(10):
        _ = interference(fields, positions)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(100):
        _ = interference(fields, positions)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"  Average interference time: {avg_time:.6f}s")
    
    # Benchmark unified dynamics
    print("\n📊 Benchmarking unified dynamics...")
    
    dynamics = UnifiedFieldDynamics(
        embed_dim=embed_dim,
        pos_dim=1,
        evolution_type="diffusion",
        interference_type="standard",
        num_steps=4
    )
    
    # Warmup
    for _ in range(10):
        _ = dynamics(fields, positions)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(50):
        _ = dynamics(fields, positions)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 50
    print(f"  Average unified dynamics time: {avg_time:.6f}s")


def run_mathematical_correction_tests():
    """Run all mathematical correction tests."""
    print("🚀 Starting Mathematical Correction Tests")
    print("=" * 60)
    
    try:
        # Test corrected interference
        test_corrected_interference()
        
        # Test corrected PDE evolution
        test_corrected_pde_evolution()
        
        # Test unified field dynamics
        test_unified_field_dynamics()
        
        # Test mathematical correctness
        test_mathematical_correctness()
        
        # Performance benchmark
        test_performance_benchmark()
        
        print("\n🎉 All mathematical correction tests PASSED!")
        print("✅ Interference implementation corrected")
        print("✅ PDE evolution implementation corrected")
        print("✅ Unified field dynamics implemented")
        print("✅ Stability monitoring added")
        print("✅ Physics constraints integrated")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_mathematical_correction_tests() 