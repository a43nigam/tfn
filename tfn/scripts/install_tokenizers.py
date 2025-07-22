#!/usr/bin/env python3
"""
Install tokenizer dependencies for TFN experiments.

Run this script in Kaggle before running the hyperparameter sweep
to ensure proper tokenization.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install required tokenizer packages."""
    print("🔧 Installing tokenizer dependencies for TFN...")
    
    # List of packages to install
    packages = [
        "transformers>=4.20.0",  # For BERT/DistilBERT tokenizers
        "nltk>=3.7",             # For NLTK tokenizer fallback
        "tokenizers>=0.13.0",    # Fast tokenizers backend
    ]
    
    success_count = 0
    for package in packages:
        print(f"\n📦 Installing {package}...")
        if install_package(package):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"📊 Installation Summary:")
    print(f"   ✅ Successful: {success_count}/{len(packages)}")
    print(f"   ❌ Failed: {len(packages) - success_count}/{len(packages)}")
    
    if success_count == len(packages):
        print(f"🎉 All tokenizer dependencies installed successfully!")
        print(f"   You can now run the TFN hyperparameter sweep.")
    else:
        print(f"⚠️  Some packages failed to install.")
        print(f"   The sweep will fall back to basic regex tokenization.")
    
    # Test tokenizer availability
    print(f"\n🧪 Testing tokenizer availability...")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        print(f"✅ DistilBERT tokenizer: Available")
    except:
        print(f"❌ DistilBERT tokenizer: Not available")
    
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        # Download required data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("📥 Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        print(f"✅ NLTK tokenizer: Available")
    except:
        print(f"❌ NLTK tokenizer: Not available")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Run the fixed hyperparameter sweep:")
    print(f"      python tfn/scripts/kaggle_sweep_fixed.py")
    print(f"   2. Compare results with/without proper tokenization")
    print(f"   3. Expect more realistic accuracy numbers!")

if __name__ == "__main__":
    main() 