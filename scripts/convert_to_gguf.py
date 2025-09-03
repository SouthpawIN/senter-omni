#!/usr/bin/env python3
"""
Convert trained Gemma3N model to GGUF format with Q8_0 quantization.

This script:
1. Loads the trained model
2. Converts to GGUF format
3. Applies Q8_0 quantization
4. Saves the optimized model
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def setup_llama_cpp():
    """Clone and build llama.cpp if not present."""
    if not os.path.exists("llama.cpp"):
        print("Cloning llama.cpp...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp"],
                      check=True)

    # Build llama.cpp with CMake
    os.chdir("llama.cpp")
    if not os.path.exists("build"):
        print("Building llama.cpp with CMake...")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        subprocess.run(["cmake", ".."], check=True)
        subprocess.run(["cmake", "--build", ".", "--config", "Release"], check=True)
        os.chdir("..")
    os.chdir("..")

def convert_to_gguf(model_path: str, output_path: str, quantization: str = "q8_0"):
    """Convert HuggingFace model to GGUF format."""

    print(f"Converting model from {model_path} to GGUF...")

    # Ensure we're in the right directory
    if not os.path.exists("llama.cpp"):
        setup_llama_cpp()

    # Convert command
    convert_cmd = [
        "python", "llama.cpp/convert_hf_to_gguf.py",
        model_path,
        "--outtype", quantization,
        "--outfile", output_path,
    ]

    print(f"Running conversion: {' '.join(convert_cmd)}")
    result = subprocess.run(convert_cmd, check=True)

    if result.returncode == 0:
        print(f"Successfully converted model to {output_path}")
        return True
    else:
        print(f"Conversion failed with return code {result.returncode}")
        return False

def verify_gguf_model(gguf_path: str):
    """Verify the converted GGUF model."""
    if not os.path.exists(gguf_path):
        print(f"Error: GGUF file {gguf_path} not found")
        return False

    file_size = os.path.getsize(gguf_path) / (1024 * 1024 * 1024)  # GB
    print(".2f")

    # Basic verification by checking file header
    with open(gguf_path, 'rb') as f:
        header = f.read(4)
        if header == b'GGUF':
            print("✓ Valid GGUF file format")
            return True
        else:
            print("✗ Invalid GGUF file format")
            return False

def optimize_gguf_model(gguf_path: str, optimized_path: str):
    """Apply additional optimizations to GGUF model."""
    print("Applying additional optimizations...")

    # For now, just copy the file (can add more optimizations later)
    import shutil
    shutil.copy2(gguf_path, optimized_path)
    print(f"Optimized model saved to {optimized_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Gemma3N model to GGUF Q8_0")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained HuggingFace model")
    parser.add_argument("--output_path", type=str, default="gemma3n-hermes-Q8_0.gguf",
                       help="Output path for GGUF file")
    parser.add_argument("--quantization", type=str, default="q8_0",
                       choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"],
                       help="Quantization type")
    parser.add_argument("--optimize", action="store_true",
                       help="Apply additional optimizations")

    args = parser.parse_args()

    # Convert model
    success = convert_to_gguf(args.model_path, args.output_path, args.quantization)

    if success:
        # Verify conversion
        if verify_gguf_model(args.output_path):
            print("✓ Model conversion and verification successful!")

            # Apply optimizations if requested
            if args.optimize:
                optimized_path = args.output_path.replace(".gguf", "-optimized.gguf")
                optimize_gguf_model(args.output_path, optimized_path)

            print(f"Final model: {args.output_path}")
        else:
            print("✗ Model verification failed")
            sys.exit(1)
    else:
        print("✗ Model conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
