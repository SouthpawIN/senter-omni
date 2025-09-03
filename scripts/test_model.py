#!/usr/bin/env python3
"""
Test script for trained Gemma3N-Hermes model.

Tests:
1. Text generation
2. Function calling capabilities
3. Uncensored responses
4. Multimodal capabilities (if available)
"""

import os
import sys
import argparse
import json
from pathlib import Path

def test_with_transformers(model_path: str):
    """Test model using transformers library."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import TextStreamer
        import torch

        print("Loading model with transformers...")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Test cases
        test_cases = [
            {
                "name": "Basic Text Generation",
                "messages": [{"role": "user", "content": "Hello! Can you introduce yourself?"}]
            },
            {
                "name": "Function Calling",
                "messages": [{"role": "user", "content": "Write a Python function to calculate factorial and call it with n=5"}]
            },
            {
                "name": "Uncensored Content",
                "messages": [{"role": "user", "content": "Tell me about controversial historical events without censorship"}]
            },
            {
                "name": "Complex Reasoning",
                "messages": [{"role": "user", "content": "Explain quantum computing in simple terms, then show a code example"}]
            }
        ]

        def generate_response(messages, max_tokens=256):
            """Generate response for given messages."""
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    streamer=TextStreamer(tokenizer, skip_prompt=True),
                )

            return tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

        # Run tests
        for test_case in test_cases:
            print(f"\n{'='*50}")
            print(f"Test: {test_case['name']}")
            print(f"{'='*50}")

            response = generate_response(test_case['messages'])
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")

    except Exception as e:
        print(f"Transformers test failed: {e}")

def test_with_llama_cpp(gguf_path: str):
    """Test GGUF model using llama.cpp."""
    try:
        import llama_cpp

        print("Loading GGUF model with llama.cpp...")

        # Load model
        model = llama_cpp.Llama(
            model_path=gguf_path,
            n_ctx=2048,
            n_threads=os.cpu_count(),
            chat_format="gemma"
        )

        # Test cases
        test_prompts = [
            "Hello! Introduce yourself as an uncensored AI assistant.",
            "Write a Python function for fibonacci sequence.",
            "Explain how neural networks work."
        ]

        for prompt in test_prompts:
            print(f"\n{'='*50}")
            print(f"Prompt: {prompt}")
            print(f"{'='*50}")

            response = model(
                prompt,
                max_tokens=256,
                temperature=0.8,
                top_p=0.9,
                echo=False
            )

            print(f"Response: {response['choices'][0]['text']}")

    except ImportError:
        print("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    except Exception as e:
        print(f"Llama.cpp test failed: {e}")

def test_multimodal_capabilities(model_path: str):
    """Test multimodal capabilities if supported."""
    try:
        from transformers import AutoModel, AutoProcessor
        import torch

        print("Testing multimodal capabilities...")

        # This would be for models with vision capabilities
        # Note: GGUF doesn't support multimodal yet, so this is for HF model testing

        try:
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)

            print("✓ Model supports multimodal inputs")

            # Test with dummy multimodal input
            print("Note: Full multimodal testing requires actual image/audio files")

        except Exception as e:
            print(f"Multimodal test failed (expected for text-only models): {e}")

    except ImportError:
        print("Multimodal testing requires transformers with vision support")

def benchmark_model(model_path: str, gguf_path: str = None):
    """Benchmark model performance."""
    print("\nBenchmarking model performance...")

    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        test_prompt = "Explain the concept of machine learning in simple terms."
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        # Warm up
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        end_time = time.time()

        generated_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        generation_time = end_time - start_time
        tokens_per_second = generated_tokens / generation_time

        print(".2f")
        print(".2f")
        print(".1f")

    except Exception as e:
        print(f"Benchmark failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test Gemma3N-Hermes model capabilities")
    parser.add_argument("--model_path", type=str,
                       help="Path to HuggingFace model directory")
    parser.add_argument("--gguf_path", type=str,
                       help="Path to GGUF model file")
    parser.add_argument("--test_type", type=str, default="all",
                       choices=["all", "transformers", "llama_cpp", "multimodal", "benchmark"],
                       help="Type of test to run")

    args = parser.parse_args()

    if not args.model_path and not args.gguf_path:
        print("Error: Must provide either --model_path or --gguf_path")
        sys.exit(1)

    print("Starting Gemma3N-Hermes model tests...")
    print(f"Model path: {args.model_path}")
    print(f"GGUF path: {args.gguf_path}")

    if args.test_type in ["all", "transformers"] and args.model_path:
        test_with_transformers(args.model_path)

    if args.test_type in ["all", "llama_cpp"] and args.gguf_path:
        test_with_llama_cpp(args.gguf_path)

    if args.test_type in ["all", "multimodal"] and args.model_path:
        test_multimodal_capabilities(args.model_path)

    if args.test_type in ["all", "benchmark"] and args.model_path:
        benchmark_model(args.model_path, args.gguf_path)

    print("\nTesting complete!")

if __name__ == "__main__":
    main()
