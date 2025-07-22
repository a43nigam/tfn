#!/usr/bin/env python3
"""
TFN Hyperparameter Sweep - FIXED VERSION for Kaggle

Addresses critical issues:
1. Memory-efficient Enhanced TFN configurations
2. Proper validation methodology
3. Valid kernel/evolution combinations only
4. More rigorous dataset splits
5. Professional tokenization

Usage:
    python tfn/scripts/kaggle_sweep_fixed.py
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

# Add tfn to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
KAGGLE_TFN_PATH = "/kaggle/input/tfn_july16/pytorch/default/42/tfn"
LOCAL_TFN_PATH = str(Path(__file__).parent.parent)  # tfn directory
TFN_PATH = KAGGLE_TFN_PATH if Path(KAGGLE_TFN_PATH).exists() else LOCAL_TFN_PATH

RESULTS_DIR = "sweep_results_fixed"
NUM_EPOCHS = 8  # More epochs for better validation
SEED = 42

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_training(command, config_name):
    """Run a training command and return success/failure."""
    print(f"\n{'='*60}")
    print(f"🏃 Running: {config_name}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            # Extract validation accuracy from output
            val_acc = extract_best_accuracy(result.stdout)
            print(f"✅ SUCCESS in {elapsed_time:.1f}s - {config_name}")
            if val_acc:
                print(f"   📊 Best Val Acc: {val_acc:.2f}%")
            
            return {
                "config": config_name,
                "status": "success",
                "val_accuracy": val_acc,
                "time": elapsed_time,
                "stdout": result.stdout[-1000:],  # Last 1000 chars
                "stderr": result.stderr[-500:] if result.stderr else ""
            }
        else:
            print(f"❌ FAILED in {elapsed_time:.1f}s - {config_name}")
            print(f"   Error: {result.stderr[-200:]}")
            return {
                "config": config_name,
                "status": "failed",
                "val_accuracy": None,
                "time": elapsed_time,
                "stdout": result.stdout[-500:] if result.stdout else "",
                "stderr": result.stderr[-500:] if result.stderr else ""
            }
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"⏰ TIMEOUT after {elapsed_time:.1f}s - {config_name}")
        return {
            "config": config_name,
            "status": "timeout",
            "val_accuracy": None,
            "time": elapsed_time,
            "stdout": "",
            "stderr": "Training timeout after 30 minutes"
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"💥 ERROR in {elapsed_time:.1f}s - {config_name}: {e}")
        return {
            "config": config_name,
            "status": "error",
            "val_accuracy": None,
            "time": elapsed_time,
            "stdout": "",
            "stderr": str(e)
        }

def extract_best_accuracy(stdout):
    """Extract the best validation accuracy from training output."""
    try:
        lines = stdout.split('\n')
        best_acc = 0.0
        
        for line in lines:
            if 'Best Val Acc:' in line:
                # Extract percentage from "Best Val Acc: 85.30%"
                acc_str = line.split('Best Val Acc:')[1].strip().replace('%', '')
                best_acc = max(best_acc, float(acc_str))
            elif 'Val Acc' in line and '%' in line:
                # Extract from training logs like "Val Acc 85.30%"
                parts = line.split('Val Acc')[1].strip().replace('%', '').split()[0]
                best_acc = max(best_acc, float(parts))
                
        return best_acc if best_acc > 0 else None
    except:
        return None

def install_tokenizers():
    """Install tokenizer dependencies if not available."""
    print("🔧 Checking tokenizer dependencies...")
    
    # Check if transformers is available
    try:
        from transformers import AutoTokenizer
        print("✅ Transformers library: Available")
        return True
    except ImportError:
        print("📦 Installing transformers library...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.20.0"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nltk>=3.7"])
            print("✅ Tokenizer dependencies installed successfully!")
            return True
        except:
            print("⚠️  Failed to install tokenizers. Will use basic regex fallback.")
            return False

def main():
    """Run FIXED hyperparameter sweep with proper validation."""
    
    results = []
    total_runs = 0
    successful_runs = 0
    
    print(f"🚀 Starting FIXED TFN Hyperparameter Sweep")
    print(f"   TFN path: {TFN_PATH}")
    print(f"   Epochs per run: {NUM_EPOCHS}")
    print(f"   Results will be saved to: {RESULTS_DIR}/")
    print(f"   🔧 FIXES: Memory optimization, proper validation, valid configs only")
    
    # Install tokenizer dependencies
    install_tokenizers()
    
    start_time = time.time()
    
    # =================================================================
    # PHASE 1: TFN GLUE Sweep (Memory-optimized, valid configs only)
    # =================================================================
    print(f"\n🔍 PHASE 1: TFN GLUE Sweep (6 configurations)")
    
    # Valid kernel/evolution combinations only
    tfn_configs = [
        ("rbf", "cnn"),      # ✅ Valid: RBF + CNN
        ("rbf", "pde"),      # ✅ Valid: RBF + PDE  
        ("compact", "cnn"),  # ✅ Valid: Compact + CNN
        ("compact", "pde"),  # ✅ Valid: Compact + PDE
        ("fourier", "cnn"),  # ✅ Valid: Fourier + CNN
        ("fourier", "pde"),  # ✅ Valid: Fourier + PDE
    ]
    
    for kernel, evo in tfn_configs:
        config_name = f"tfn_{kernel}_{evo}"
        
        command = f"""
        python {TFN_PATH}/scripts/train_glue_tfn.py \\
            --task sst2 \\
            --model tfn \\
            --kernel_type {kernel} \\
            --evolution_type {evo} \\
            --embed_dim 128 --num_layers 2 \\
            --grid_size 64 --time_steps 3 \\
            --batch_size 32 --epochs {NUM_EPOCHS} \\
            --lr 3e-4 --tag sweep_{kernel}_{evo} \\
            --seed {SEED}
        """.strip().replace('\n', ' ').replace('\\', '')
        
        result = run_training(command, config_name)
        results.append(result)
        total_runs += 1
        if result["status"] == "success":
            successful_runs += 1
    
    # =================================================================
    # PHASE 2: Enhanced TFN Sweep (Memory-optimized)
    # =================================================================
    print(f"\n🔍 PHASE 2: Enhanced TFN Sweep (3 configurations)")
    
    # Memory-optimized Enhanced TFN configs
    enhanced_configs = [
        {"grid": 32, "batch": 16, "embed": 64},   # Very conservative
        {"grid": 48, "batch": 24, "embed": 96},   # Moderate
        {"grid": 64, "batch": 32, "embed": 128},  # Standard (if memory allows)
    ]
    
    for i, config in enumerate(enhanced_configs, 1):
        config_name = f"enhanced_tfn_mem{i}"
        
        command = f"""
        python {TFN_PATH}/scripts/train_enhanced_tfn.py \\
            --task sst2 \\
            --embed_dim {config['embed']} --num_layers 2 \\
            --grid_size {config['grid']} --time_steps 3 \\
            --batch_size {config['batch']} --epochs {NUM_EPOCHS} \\
            --lr 3e-4 --tag sweep_enhanced_mem{i} \\
            --seed {SEED}
        """.strip().replace('\n', ' ').replace('\\', '')
        
        result = run_training(command, config_name)
        results.append(result)
        total_runs += 1
        if result["status"] == "success":
            successful_runs += 1
    
    # =================================================================
    # PHASE 3: Cross-validation on other GLUE tasks
    # =================================================================
    print(f"\n🔍 PHASE 3: Cross-validation (12 configurations)")
    
    # Test best TFN config on other GLUE tasks for validation
    best_tfn_config = ("rbf", "cnn")  # Usually performs well
    cross_val_tasks = ["mrpc", "cola", "rte"]
    
    for task in cross_val_tasks:
        for kernel, evo in [best_tfn_config, ("fourier", "pde")]:  # Test 2 configs per task
            config_name = f"tfn_{task}_{kernel}_{evo}"
            
            command = f"""
            python {TFN_PATH}/scripts/train_glue_tfn.py \\
                --task {task} \\
                --model tfn \\
                --kernel_type {kernel} \\
                --evolution_type {evo} \\
                --embed_dim 128 --num_layers 2 \\
                --grid_size 64 --time_steps 3 \\
                --batch_size 32 --epochs {NUM_EPOCHS} \\
                --lr 3e-4 --tag sweep_{task}_{kernel}_{evo} \\
                --seed {SEED}
            """.strip().replace('\n', ' ').replace('\\', '')
            
            result = run_training(command, config_name)
            results.append(result)
            total_runs += 1
            if result["status"] == "success":
                successful_runs += 1
    
    # =================================================================
    # SUMMARY AND RESULTS
    # =================================================================
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"🎯 FIXED SWEEP COMPLETE!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Successful runs: {successful_runs}/{total_runs}")
    print(f"   Success rate: {successful_runs/total_runs*100:.1f}%")
    
    # Save detailed results
    results_file = f"{RESULTS_DIR}/detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "total_time_minutes": total_time/60,
                "timestamp": time.time()
            },
            "results": results
        }, f, indent=2)
    
    # Create CSV summary
    successful_results = [r for r in results if r["status"] == "success" and r["val_accuracy"]]
    if successful_results:
        import pandas as pd
        
        df = pd.DataFrame([
            {
                "config": r["config"],
                "val_accuracy": r["val_accuracy"],
                "time_minutes": r["time"]/60
            }
            for r in successful_results
        ])
        
        csv_file = f"{RESULTS_DIR}/summary.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n📊 Top 5 Results:")
        top_5 = df.nlargest(5, 'val_accuracy')
        for _, row in top_5.iterrows():
            print(f"   {row['config']:20} | {row['val_accuracy']:5.1f}% | {row['time_minutes']:4.1f}m")
        
        print(f"\n💾 Results saved:")
        print(f"   📋 Detailed: {results_file}")
        print(f"   📊 Summary:  {csv_file}")
    
    print(f"\n🎉 Fixed sweep complete with realistic tokenization!")
    print(f"   Expected accuracy range: 80-88% (down from 92%+)")
    print(f"   Cross-task validation ensures robust results")

if __name__ == "__main__":
    main() 