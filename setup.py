#!/usr/bin/env python3
"""
Setup script for FinAI Contest 2025 Task 1
Creates necessary directories and validates setup.
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    dirs = [
        "data",
        "trained_models", 
        "factor_mining",
        "logs"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}/")

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'gymnasium', 'stable_baselines3', 'transformers',
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def validate_data_files():
    """Check if data files exist."""
    data_files = ["data/news_train.csv", "data/BTC_1min.csv"]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ Found data file: {file_path}")
        else:
            print(f"⚠️  Missing data file: {file_path}")
            print(f"   Please place your data file at: {file_path}")

def main():
    """Main setup function."""
    print("=== FinAI Contest 2025 Task 1 Setup ===\n")
    
    # Create directories
    print("Creating directories...")
    create_directories()
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    # Validate data files
    print("Checking data files...")
    validate_data_files()
    print()
    
    # Setup instructions
    print("=== Setup Complete ===")
    if deps_ok:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Place your data files in the data/ directory")
        print("2. Set up HuggingFace token (if using LLM models)")
        print("3. Run training: python task1_ensemble.py")
        print("4. Run evaluation: python task1_eval.py")
    else:
        print("❌ Please install missing dependencies and run setup again")

if __name__ == "__main__":
    main()