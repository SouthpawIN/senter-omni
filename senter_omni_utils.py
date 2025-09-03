#!/usr/bin/env python3
"""
Senter-Omni Utility Library

Easy-to-use functions for integrating Senter-Omni into your projects.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
from pathlib import Path

class SenterOmniAI:
    """
    Senter-Omni AI Assistant Wrapper

    Easy integration of Senter-Omni into your Python projects.

    Usage:
        ai = SenterOmniAI()
        response = ai.ask("Hello, what's your name?")
    """

    def __init__(self, model_path="models/huggingface/senter-omni-lora", device=None):
        """
        Initialize Senter-Omni AI

        Args:
            model_path (str): Path to model ("lora" or "merged")
            device (str): Device to use ("cuda", "cpu", or auto-detect)
        """
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load the Senter-Omni model"""
        print(f"🤖 Loading Senter-Omni from {model_path}...")

        try:
            if "lora" in model_path:
                # Load LoRA model
                base_model = AutoModelForCausalLM.from_pretrained(
                    "unsloth/gemma-3n-E4B-it",
                    torch_dtype=torch.float16,
                    device_map={"": self.device} if "cuda" in self.device else self.device,
                    trust_remote_code=True
                )
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3n-E4B-it")
            else:
                # Load merged model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map={"": self.device} if "cuda" in self.device else self.device,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            print(f"✅ Senter-Omni loaded on {self.device}!")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Make sure you're in the senter-omni directory and models exist.")
            sys.exit(1)

    def ask(self, question, max_tokens=256, temperature=0.8, show_prompt=False):
        """
        Ask Senter-Omni a question

        Args:
            question (str): Your question or prompt
            max_tokens (int): Maximum response length
            temperature (float): Creativity (0.1=consistent, 1.0=creative)
            show_prompt (bool): Whether to print the prompt

        Returns:
            str: AI response
        """
        if show_prompt:
            print(f"❓ {question}")

        inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1
            )

        response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        return response

    def chat(self, messages, max_tokens=256):
        """
        Chat with conversation history

        Args:
            messages (list): List of message dicts with 'role' and 'content'
            max_tokens (int): Maximum response length

        Returns:
            str: AI response
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9
            )

        response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        return response

    def solve_math(self, problem):
        """Solve a math problem"""
        prompt = f"Solve this step by step: {problem}"
        return self.ask(prompt, temperature=0.1)  # Low temperature for accuracy

    def generate_code(self, description):
        """Generate code based on description"""
        prompt = f"Write clean, well-documented code for: {description}"
        return self.ask(prompt, max_tokens=512, temperature=0.7)

    def explain_concept(self, concept):
        """Explain a concept in simple terms"""
        prompt = f"Explain {concept} in simple, easy-to-understand terms."
        return self.ask(prompt, temperature=0.3)  # Low temperature for clarity

# Convenience functions for quick use
def quick_ask(question, model="lora"):
    """Quick one-off question without loading class"""
    model_path = f"models/huggingface/senter-omni-{model}"
    ai = SenterOmniAI(model_path)
    return ai.ask(question)

def math_solver(problem):
    """Quick math problem solver"""
    ai = SenterOmniAI()
    return ai.solve_math(problem)

def code_generator(description):
    """Quick code generator"""
    ai = SenterOmniAI()
    return ai.generate_code(description)

def concept_explainer(concept):
    """Quick concept explainer"""
    ai = SenterOmniAI()
    return ai.explain_concept(concept)

# Example usage
if __name__ == "__main__":
    print("🧪 Testing Senter-Omni Utils")
    print("=" * 40)

    # Test basic functionality
    ai = SenterOmniAI()

    print("\n1. Basic Question:")
    response = ai.ask("Hello! What's your name?")
    print(f"🤖 {response}")

    print("\n2. Math Problem:")
    math_result = ai.solve_math("What is 15 × 23 + 7?")
    print(f"🤖 {math_result}")

    print("\n3. Code Generation:")
    code = ai.generate_code("a Python function to check if a string is a palindrome")
    print(f"🤖 {code[:200]}...")

    print("\n4. Concept Explanation:")
    explanation = ai.explain_concept("machine learning")
    print(f"🤖 {explanation[:200]}...")

    print("\n✅ All tests completed!")

