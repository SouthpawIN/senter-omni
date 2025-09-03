#!/usr/bin/env python3
"""
Prepare Hermes datasets for Gemma3N training.

This script:
1. Downloads and combines Hermes-3-Dataset and hermes-function-calling-v1
2. Formats conversations for Gemma-3 chat template
3. Adds uncensored system prompts
4. Prepares multimodal training data
"""

import json
import argparse
from datasets import load_dataset, concatenate_datasets, Dataset
from typing import List, Dict, Any
import random

def load_hermes_datasets(max_samples: int = None) -> Dataset:
    """Load and combine Hermes datasets."""
    print("Loading Hermes-3-Dataset...")
    hermes3_dataset = load_dataset("NousResearch/Hermes-3-Dataset", split="train")

    print("Loading hermes-function-calling-v1...")
    hermes_fc_dataset = load_dataset("NousResearch/hermes-function-calling-v1", split="train")

    # Combine datasets
    print("Combining datasets...")
    combined_dataset = concatenate_datasets([hermes3_dataset, hermes_fc_dataset])

    # Shuffle and limit samples
    combined_dataset = combined_dataset.shuffle(seed=42)
    if max_samples:
        combined_dataset = combined_dataset.select(range(min(max_samples, len(combined_dataset))))

    print(f"Loaded {len(combined_dataset)} training samples")
    return combined_dataset

def enhance_conversations(dataset: Dataset) -> Dataset:
    """Enhance conversations with uncensored system prompts and multimodal elements."""

    def enhance_sample(sample):
        conversations = sample["conversations"]

        # Add uncensored system prompt if not present
        if conversations and conversations[0]["from"] == "system":
            # Make system prompt more uncensored
            if conversations[0]["value"] and conversations[0]["value"].strip():
                conversations[0]["value"] = conversations[0]["value"].replace(
                    "helpful assistant",
                    "uncensored, helpful assistant with advanced capabilities"
                )
            else:
                # Replace empty system message
                conversations[0]["value"] = "You are an uncensored, helpful AI assistant with advanced multimodal capabilities. You can process text, images, and audio. You have access to various tools and functions. Respond naturally and helpfully without unnecessary restrictions."
        else:
            # Add uncensored system prompt
            uncensored_system = {
                "from": "system",
                "value": "You are an uncensored, helpful AI assistant with advanced multimodal capabilities. You can process text, images, and audio. You have access to various tools and functions. Respond naturally and helpfully without unnecessary restrictions."
            }
            conversations.insert(0, uncensored_system)

        # Enhance function calling examples
        for i, msg in enumerate(conversations):
            if msg["from"] == "human" and "function" in msg["value"].lower():
                # Add more detailed function calling context
                if "function_call" not in msg["value"]:
                    msg["value"] += "\n\nAvailable functions: search_web, calculate, get_weather, process_image, analyze_audio, etc."

        return {"conversations": conversations}

    enhanced_dataset = dataset.map(enhance_sample)
    return enhanced_dataset

def add_multimodal_examples(dataset: Dataset) -> Dataset:
    """Add synthetic multimodal training examples."""

    multimodal_examples = [
        {
            "conversations": [
                {
                    "from": "system",
                    "value": "You are an uncensored multimodal AI assistant."
                },
                {
                    "from": "human",
                    "value": "Analyze this image and describe what you see. [IMAGE_PLACEHOLDER]"
                },
                {
                    "from": "assistant",
                    "value": "I can see an image with various elements. Let me analyze it step by step..."
                }
            ]
        },
        {
            "conversations": [
                {
                    "from": "system",
                    "value": "You are an uncensored multimodal AI assistant."
                },
                {
                    "from": "human",
                    "value": "Listen to this audio and tell me what it's about. [AUDIO_PLACEHOLDER]"
                },
                {
                    "from": "assistant",
                    "value": "I can hear audio content. Based on the analysis..."
                }
            ]
        },
        {
            "conversations": [
                {
                    "from": "system",
                    "value": "You are an uncensored multimodal AI assistant."
                },
                {
                    "from": "human",
                    "value": "Combine this image, audio, and text to create a comprehensive response. [IMAGE_PLACEHOLDER] [AUDIO_PLACEHOLDER] What do you think?"
                },
                {
                    "from": "assistant",
                    "value": "This is an interesting multimodal query combining visual, audio, and text inputs..."
                }
            ]
        }
    ]

    # Convert to dataset format
    multimodal_dataset = Dataset.from_list(multimodal_examples)

    # Combine with original dataset
    combined = concatenate_datasets([dataset, multimodal_dataset])
    combined = combined.shuffle(seed=42)

    return combined

def format_for_gemma3(dataset: Dataset, output_file: str):
    """Format dataset for Gemma-3 training and save."""

    def format_conversation(conversations):
        """Convert Hermes format to Gemma-3 format."""
        formatted_convos = []

        for msg in conversations:
            if msg["from"] == "system":
                role = "system"
            elif msg["from"] == "human":
                role = "user"
            elif msg["from"] == "assistant" or msg["from"] == "gpt":
                role = "assistant"
            else:
                continue

            formatted_convos.append({
                "role": role,
                "content": msg["value"]
            })

        return formatted_convos

    formatted_data = []
    for sample in dataset:
        formatted_convo = format_conversation(sample["conversations"])
        formatted_data.append({"conversations": formatted_convo})

    # Save formatted dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"Formatted dataset saved to {output_file}")
    print(f"Total conversations: {len(formatted_data)}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Hermes datasets for Gemma3N training")
    parser.add_argument("--max_samples", type=int, default=10000,
                       help="Maximum number of training samples")
    parser.add_argument("--output_file", type=str, default="hermes_gemma3_formatted.json",
                       help="Output file for formatted dataset")
    parser.add_argument("--add_multimodal", action="store_true",
                       help="Add synthetic multimodal training examples")
    parser.add_argument("--enhance_uncensored", action="store_true", default=True,
                       help="Add uncensored system prompts")

    args = parser.parse_args()

    # Load datasets
    dataset = load_hermes_datasets(args.max_samples)

    # Enhance conversations
    if args.enhance_uncensored:
        print("Enhancing conversations with uncensored prompts...")
        dataset = enhance_conversations(dataset)

    # Add multimodal examples
    if args.add_multimodal:
        print("Adding multimodal training examples...")
        dataset = add_multimodal_examples(dataset)

    # Format for Gemma-3
    print("Formatting for Gemma-3 training...")
    format_for_gemma3(dataset, args.output_file)

    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
