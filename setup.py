#!/usr/bin/env python3
"""
Setup script for Gemma3N-Hermes training environment.

This script:
1. Installs dependencies
2. Downloads required datasets
3. Sets up the environment
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd: list, description: str, check: bool = True):
    """Run command with error handling."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        print("✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("Installing dependencies...")

    # Install from requirements.txt
    if os.path.exists("requirements.txt"):
        success = run_command([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], "Installing Python packages")
        return success
    else:
        print("requirements.txt not found!")
        return False

def download_datasets():
    """Download and cache the Hermes datasets."""
    print("Downloading Hermes datasets...")

    try:
        from datasets import load_dataset

        # Download Hermes-3-Dataset
        print("Downloading Hermes-3-Dataset...")
        hermes3 = load_dataset("NousResearch/Hermes-3-Dataset", split="train[:1000]")
        print(f"Hermes-3-Dataset: {len(hermes3)} samples")

        # Download hermes-function-calling-v1
        print("Downloading hermes-function-calling-v1...")
        hermes_fc = load_dataset("NousResearch/hermes-function-calling-v1", split="train[:1000]")
        print(f"hermes-function-calling-v1: {len(hermes_fc)} samples")

        print("✓ Datasets downloaded successfully!")
        return True

    except ImportError:
        print("datasets library not installed. Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"Dataset download failed: {e}")
        return False

def setup_unsloth():
    """Setup Unsloth environment."""
    print("Setting up Unsloth...")

    try:
        import unsloth
        print(f"Unsloth version: {unsloth.__version__}")
        print("✓ Unsloth setup complete!")
        return True
    except ImportError:
        print("Installing Unsloth...")
        success = run_command([
            sys.executable, "-m", "pip", "install", "unsloth"
        ], "Installing Unsloth")
        return success
    except Exception as e:
        print(f"Unsloth setup failed: {e}")
        return False

def setup_llama_cpp():
    """Setup llama.cpp for GGUF conversion."""
    print("Setting up llama.cpp...")

    if os.path.exists("llama.cpp"):
        print("llama.cpp already exists")
        return True

    # Clone llama.cpp
    success = run_command([
        "git", "clone", "https://github.com/ggerganov/llama.cpp"
    ], "Cloning llama.cpp")

    if success:
        # Build llama.cpp
        os.chdir("llama.cpp")
        success = run_command(["make"], "Building llama.cpp")
        os.chdir("..")

    return success

def create_directories():
    """Create necessary directories."""
    directories = ["data", "models", "scripts", "notebooks"]

    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def verify_setup():
    """Verify the setup is complete."""
    print("Verifying setup...")

    checks = []

    # Check Python packages
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ PyTorch not installed")
        checks.append(False)

    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ Transformers not installed")
        checks.append(False)

    try:
        import datasets
        print(f"✓ Datasets: {datasets.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ Datasets not installed")
        checks.append(False)

    try:
        import unsloth
        print(f"✓ Unsloth: {unsloth.__version__}")
        checks.append(True)
    except ImportError:
        print("✗ Unsloth not installed")
        checks.append(False)

    # Check directories
    for dir_name in ["data", "models", "scripts", "notebooks"]:
        if os.path.exists(dir_name):
            print(f"✓ Directory: {dir_name}")
            checks.append(True)
        else:
            print(f"✗ Directory missing: {dir_name}")
            checks.append(False)

    # Check llama.cpp
    if os.path.exists("llama.cpp"):
        print("✓ llama.cpp directory")
        checks.append(True)
    else:
        print("✗ llama.cpp missing")
        checks.append(False)

    return all(checks)

def main():
    parser = argparse.ArgumentParser(description="Setup Gemma3N-Hermes training environment")
    parser.add_argument("--skip-deps", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--skip-datasets", action="store_true",
                       help="Skip dataset download")
    parser.add_argument("--skip-llama-cpp", action="store_true",
                       help="Skip llama.cpp setup")

    args = parser.parse_args()

    print("🚀 Setting up Gemma3N-Hermes training environment")
    print("=" * 50)

    # Create directories
    create_directories()

    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("❌ Dependency installation failed")
            sys.exit(1)

    # Setup Unsloth
    if not setup_unsloth():
        print("❌ Unsloth setup failed")
        sys.exit(1)

    # Setup llama.cpp
    if not args.skip_llama_cpp:
        if not setup_llama_cpp():
            print("❌ llama.cpp setup failed")
            sys.exit(1)

    # Download datasets
    if not args.skip_datasets:
        if not download_datasets():
            print("❌ Dataset download failed")
            sys.exit(1)

    # Verify setup
    if verify_setup():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run training: python train_gemma3n_hermes.py")
        print("2. Or use notebook: jupyter notebook notebooks/train_gemma3n_hermes.ipynb")
    else:
        print("\n❌ Setup verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

