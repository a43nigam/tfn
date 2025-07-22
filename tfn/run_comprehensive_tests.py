#!/usr/bin/env python3
"""
Comprehensive TFN testing suite with command line flags.
Run with: python run_comprehensive_tests.py [options]
"""

import sys
import os
import argparse
import subprocess
import json
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def run_test(test_name, test_file, timeout=300):
    """Run a single test and return results."""
    print(f"\n{'='*50}")
    print(f"RUNNING {test_name.upper()} TEST")
    print(f"{'='*50}")
    
    try:
        # Resolve absolute path to test file
        test_file_path = os.path.join(os.path.dirname(__file__), test_file)
        
        result = subprocess.run(
            [sys.executable, test_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✓ {test_name} test PASSED")
            return True, result.stdout
        else:
            print(f"✗ {test_name} test FAILED")
            print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"✗ {test_name} test TIMEOUT")
        return False, "Test timed out"
    except Exception as e:
        print(f"✗ {test_name} test ERROR: {e}")
        return False, str(e)

def run_basic_imports_test():
    """Test basic imports work."""
    print(f"\n{'='*50}")
    print("RUNNING BASIC IMPORTS TEST")
    print(f"{'='*50}")
    
    try:
        from tfn.tfn_datasets.physics_loader import create_physics_dataloader
        print("✓ Physics loader import successful")
        
        from tfn.core.grid_utils import compute_auto_grid_size
        print("✓ Grid utils import successful")
        
        from tfn.model.tfn_base import TrainableTFNLayer
        print("✓ TFN model import successful")
        
        from tfn.model.seq_baselines import SimpleTransformerSeqModel
        print("✓ Baseline models import successful")
        
        print("✓ All basic imports successful!")
        return True, "All imports successful"
        
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False, str(e)

def main():
    """Main test runner with command line arguments."""
    parser = argparse.ArgumentParser(description="TFN Comprehensive Test Suite")
    
    # Test selection
    parser.add_argument("--test", choices=[
        "all", "basic", "grid", "physics", "models", "simple", "implementation"
    ], default="all", help="Which test to run")
    
    # Model parameters
    parser.add_argument("--seq-len", type=int, default=1024, 
                       help="Sequence length for testing")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for testing")
    parser.add_argument("--embed-dim", type=int, default=128,
                       help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of layers")
    
    # Physics parameters
    parser.add_argument("--pde-type", choices=["burgers", "wave", "heat"], 
                       default="burgers", help="PDE type for physics tests")
    parser.add_argument("--grid-points", type=int, default=64,
                       help="Grid points for physics tests")
    
    # Grid parameters
    parser.add_argument("--heuristic", choices=["sqrt", "linear", "log", "adaptive"],
                       default="sqrt", help="Grid size heuristic")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="test_results",
                       help="Directory to save results")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout for each test in seconds")
    
    args = parser.parse_args()
    
    print("TFN COMPREHENSIVE TEST SUITE")
    print("=" * 50)
    print(f"Test: {args.test}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Embedding Dimension: {args.embed_dim}")
    print(f"Number of Layers: {args.num_layers}")
    print(f"PDE Type: {args.pde_type}")
    print(f"Grid Points: {args.grid_points}")
    print(f"Heuristic: {args.heuristic}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Timeout: {args.timeout}s")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define tests
    tests = {
        "basic": ("Basic Imports", run_basic_imports_test),
        "grid": ("Grid Utils", "test_grid_utils_cli.py"),
        "physics": ("Physics PDE", "test_physics_pde_cli.py"),
        "models": ("Simple Models", "test_simple_models_cli.py"),
        "simple": ("Simple Models", "test_simple_models_cli.py"),
        "implementation": ("Implementation", "test_implementation.py")
    }
    
    results = {}
    passed = 0
    total = 0
    
    if args.test == "all":
        test_list = list(tests.keys())
    else:
        test_list = [args.test]
    
    for test_key in test_list:
        if test_key in tests:
            test_name, test_func = tests[test_key]
            total += 1
            
            print(f"\nRunning {test_name} test...")
            
            if callable(test_func):
                success, output = test_func()
            else:
                success, output = run_test(test_name, test_func, args.timeout)
            
            results[test_name] = {
                "success": success,
                "output": output
            }
            
            if success:
                passed += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*50}")
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result["success"] else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Save results
    results_file = os.path.join(args.output_dir, "comprehensive_test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    if passed == total:
        print("🎉 All tests passed! TFN implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 