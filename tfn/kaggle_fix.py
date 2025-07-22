#!/usr/bin/env python3
"""
Kaggle import fix for TFN tests.

Run this first in Kaggle to set up the import paths correctly.
"""

import sys
import os
from pathlib import Path

def setup_kaggle_paths():
    """Setup Python paths for Kaggle environment."""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Add to Python path if not already there
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
        print(f"✅ Added {script_dir} to Python path")
    
    # Also add parent directory if needed
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
        print(f"✅ Added {parent_dir} to Python path")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    return script_dir

def test_imports():
    """Test that all imports work."""
    
    print("\n" + "="*50)
    print("TESTING IMPORTS")
    print("="*50)
    
    try:
        from tfn.tfn_datasets.pg19_loader import create_pg19_dataloader
        print("✅ PG-19 loader import successful")
    except ImportError as e:
        print(f"❌ PG-19 loader import failed: {e}")
    
    try:
        from tfn.model.tfn_base import TrainableTFNLayer
        print("✅ TFN model import successful")
    except ImportError as e:
        print(f"❌ TFN model import failed: {e}")
    
    try:
        from tfn.core.grid_utils import compute_auto_grid_size
        print("✅ Grid utils import successful")
    except ImportError as e:
        print(f"❌ Grid utils import failed: {e}")
    
    try:
        from tfn.model.seq_baselines import SimplePerformerSeqModel
        print("✅ Baseline models import successful")
    except ImportError as e:
        print(f"❌ Baseline models import failed: {e}")

if __name__ == "__main__":
    print("Setting up Kaggle environment...")
    script_dir = setup_kaggle_paths()
    test_imports()
    print("\n✅ Kaggle setup complete!")
    print("You can now run TFN tests.") 