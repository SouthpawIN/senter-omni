#!/usr/bin/env python3
"""
🎭 Senter-Omni Unified API

Unified interface for chat and embedding operations across modalities.
"""

import torch
import re
import json
import os
from typing import Dict, Any, List, Optional, Union, Iterator
from pathlib import Path

# Import our models
try:
    from senter_omni import SenterOmniChat
    CHAT_AVAILABLE = True
except ImportError:
    CHAT_AVAILABLE = False
    print("⚠️ Chat model not available")

try:
    from senter_embed.core import SenterEmbedder
    EMBED_AVAILABLE = True
except ImportError:
    EMBED_AVAILABLE = False
    print("⚠️ Embedding model not available")

# Training imports
try:
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
    from datasets import load_dataset, concatenate_datasets, Dataset
    from trl import SFTTrainer, SFTConfig
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("⚠️ Training dependencies not available. Install: pip install unsloth trl datasets")

class OmniClient:
    """
    Unified client for Senter-Omni chat and embedding operations
    """

    def __init__(self, chat_device="auto", embed_device="auto"):
        """
        Initialize the Omni client

        Args:
            chat_device: Device for chat model ('auto', 'cuda:0', 'cpu')
            embed_device: Device for embedding model ('auto', 'cuda:1', 'cpu')
        """
        self.chat_model = None
        self.embed_model = None

        # Initialize models if available
        if CHAT_AVAILABLE:
            try:
                self.chat_model = SenterOmniChat(device=chat_device)
                print("✅ Chat model initialized")
            except Exception as e:
                print(f"❌ Failed to initialize chat model: {e}")

        if EMBED_AVAILABLE:
            try:
                # Try different devices if the first one fails
                devices_to_try = [embed_device]
                if embed_device == "auto":
                    devices_to_try = ["cuda:1", "cuda:0", "cpu"]

                for device in devices_to_try:
                    try:
                        self.embed_model = SenterEmbedder(device=device)
                        print(f"✅ Embedding model initialized on {device}")
                        break
                    except Exception as e:
                        print(f"⚠️ Failed to initialize embedding model on {device}: {e}")
                        continue

                if self.embed_model is None:
                    print("❌ Failed to initialize embedding model on any device")

            except Exception as e:
                print(f"❌ Failed to initialize embedding model: {e}")

    def chat(self,
             messages: Union[str, List[Dict]],
             max_tokens: int = 256,
             temperature: float = 0.8,
             top_p: float = 0.9,
             top_k: int = 50,
             stream: bool = False,
             stop_sequences: Optional[List[str]] = None,
             **kwargs) -> Union[str, Iterator[str]]:
        """
        Generate chat completions with configurable parameters

        Args:
            messages: Input messages (string or list of message dicts)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter
            stream: Whether to stream the response
            stop_sequences: Custom stop sequences
            **kwargs: Additional generation parameters

        Returns:
            Generated response (string or iterator if streaming)
        """
        if not self.chat_model:
            raise RuntimeError("Chat model not available. Please install senter-omni.")

        # Prepare generation parameters
        generation_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0.0,
            "stream": stream,
            **kwargs
        }

        # Add custom stop sequences if provided
        if stop_sequences:
            # Convert to token IDs and add to generation params
            stop_token_ids = []
            for seq in stop_sequences:
                try:
                    tokens = self.chat_model.tokenizer.encode(seq, add_special_tokens=False)
                    stop_token_ids.extend(tokens)
                except:
                    pass
            if stop_token_ids:
                generation_params["stop_token_ids"] = stop_token_ids

        # Generate response
        if stream:
            return self.chat_model.generate_streaming([messages], generation_params)
        else:
            return self.chat_model.generate_streaming([messages], generation_params)

    def embed(self,
              input_content: str,
              operation: str = "embed",
              similarity_threshold: float = 0.0,
              top_k: int = 5,
              **kwargs) -> Dict[str, Any]:
        """
        Process multimodal embeddings and similarity search

        Args:
            input_content: XML-formatted input with <text>, <image>, <audio> tags
            operation: Operation to perform ("embed", "similarity", "search")
            similarity_threshold: Minimum similarity score for results
            top_k: Number of top results for search operations
            **kwargs: Additional parameters

        Returns:
            Dictionary with embedding results and metadata
        """
        if not self.embed_model:
            raise RuntimeError("Embedding model not available. Please install senter-embed.")

        # Parse XML content
        modalities = self._parse_multimodal_content(input_content)

        result = {
            "operation": operation,
            "modalities": list(modalities.keys()),
            "embeddings": {},
            "metadata": {}
        }

        # Generate embeddings for each modality
        embeddings = {}

        if "text" in modalities:
            for i, text in enumerate(modalities["text"]):
                embeddings[f"text_{i}"] = {
                    "content": text,
                    "embedding": self.embed_model.get_text_embedding(text)
                }

        if "image" in modalities:
            for i, image_path in enumerate(modalities["image"]):
                if Path(image_path).exists():
                    embeddings[f"image_{i}"] = {
                        "content": image_path,
                        "embedding": self.embed_model.get_image_embedding(image_path)
                    }
                else:
                    print(f"⚠️ Image not found: {image_path}")

        if "audio" in modalities:
            for i, audio_path in enumerate(modalities["audio"]):
                if Path(audio_path).exists():
                    embeddings[f"audio_{i}"] = {
                        "content": audio_path,
                        "embedding": self.embed_model.get_audio_embedding(audio_path)
                    }
                else:
                    print(f"⚠️ Audio not found: {audio_path}")

        result["embeddings"] = embeddings

        # Perform similarity operations
        if operation in ["similarity", "search"] and len(embeddings) > 1:
            similarities = self._compute_cross_modal_similarities(embeddings)
            result["similarities"] = similarities

            # Filter by threshold
            if similarity_threshold > 0:
                filtered_similarities = {}
                for pair, score in similarities.items():
                    if score >= similarity_threshold:
                        filtered_similarities[pair] = score
                result["similarities_filtered"] = filtered_similarities

        return result

    def _parse_multimodal_content(self, content: str) -> Dict[str, List[str]]:
        """
        Parse XML-formatted content for multimodal inputs

        Args:
            content: XML string with <text>, <image>, <audio> tags

        Returns:
            Dictionary with modality content lists
        """
        modalities = {
            "text": [],
            "image": [],
            "audio": []
        }

        # Parse text content
        text_matches = re.findall(r'<text>(.*?)</text>', content, re.DOTALL)
        modalities["text"].extend([text.strip() for text in text_matches])

        # Parse image paths
        image_matches = re.findall(r'<image>(.*?)</image>', content, re.DOTALL)
        modalities["image"].extend([img.strip() for img in image_matches])

        # Parse audio paths
        audio_matches = re.findall(r'<audio>(.*?)</audio>', content, re.DOTALL)
        modalities["audio"].extend([audio.strip() for audio in audio_matches])

        return modalities

    def _compute_cross_modal_similarities(self, embeddings: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute similarities between all pairs of embeddings

        Args:
            embeddings: Dictionary of embeddings with metadata

        Returns:
            Dictionary of similarity scores
        """
        similarities = {}
        embedding_items = list(embeddings.items())

        for i, (key1, data1) in enumerate(embedding_items):
            for j, (key2, data2) in enumerate(embedding_items):
                if i < j:  # Only compute each pair once
                    similarity = self.embed_model.compute_similarity(
                        data1["embedding"],
                        data2["embedding"]
                    )
                    pair_key = f"{key1}_vs_{key2}"
                    similarities[pair_key] = similarity

        return similarities

    def train(self,
              dataset_name_or_path: Union[str, List[str]],
              output_dir: str = "models/senter-omni-trained",
              max_samples: int = 10000,
              num_epochs: int = 3,
              learning_rate: float = 2e-4,
              batch_size: int = 2,
              gradient_accumulation_steps: int = 4,
              lora_rank: int = 16,
              save_steps: int = 500,
              **kwargs) -> Dict[str, Any]:
        """
        Train a new Senter-Omni multimodal chat model

        Args:
            dataset_name_or_path: Dataset name(s) on HuggingFace or local path(s)
            output_dir: Directory to save the trained model
            max_samples: Maximum number of training samples
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            lora_rank: LoRA rank for fine-tuning
            save_steps: Save model every N steps
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training statistics and model paths
        """
        if not TRAINING_AVAILABLE:
            raise RuntimeError("Training dependencies not available. Install: pip install unsloth trl datasets")

        print("🚀 Starting Senter-Omni Training Pipeline")
        print("=" * 60)

        # Step 1: Load and prepare datasets
        print("📚 Step 1: Loading datasets...")
        datasets = []

        if isinstance(dataset_name_or_path, str):
            dataset_name_or_path = [dataset_name_or_path]

        for dataset_source in dataset_name_or_path:
            if os.path.exists(dataset_source):
                # Load from local path
                print(f"Loading local dataset: {dataset_source}")
                try:
                    dataset = Dataset.from_json(dataset_source)
                except:
                    # Try loading as JSON lines
                    with open(dataset_source, 'r') as f:
                        data = [json.loads(line) for line in f]
                    dataset = Dataset.from_list(data)
            else:
                # Load from HuggingFace
                print(f"Loading HuggingFace dataset: {dataset_source}")
                try:
                    dataset = load_dataset(dataset_source, split="train")
                except:
                    print(f"⚠️ Could not load {dataset_source}, skipping...")
                    continue

            datasets.append(dataset)

        if not datasets:
            raise ValueError("No valid datasets found")

        # Combine datasets
        if len(datasets) > 1:
            print("Combining multiple datasets...")
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]

        # Shuffle and limit samples
        combined_dataset = combined_dataset.shuffle(seed=42)
        dataset = combined_dataset.select(range(min(max_samples, len(combined_dataset))))
        print(f"📊 Training on {len(dataset)} samples")

        # Step 2: Initialize model
        print("🤖 Step 2: Initializing Gemma3N model...")
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3n-E4B-it",
            dtype=None,
            max_seq_length=2048,
            load_in_4bit=True,
            full_finetuning=False,
        )

        # Step 3: Configure LoRA
        print("🔧 Step 3: Configuring LoRA...")
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.05,
            bias="none",
            random_state=3407,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        # Step 4: Prepare data
        print("📝 Step 4: Preparing training data...")
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        dataset = standardize_data_formats(dataset)

        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>') for convo in convos]
            return {"text": texts}

        dataset = dataset.map(formatting_prompts_func, batched=True)

        # Step 5: Configure trainer
        print("⚙️ Step 5: Configuring trainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=None,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=10,
                num_train_epochs=num_epochs,
                max_steps=None,
                learning_rate=learning_rate,
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,
                report_to="none",
                save_steps=save_steps,
                save_total_limit=3,
                fp16=True,
                gradient_checkpointing=True,
                **kwargs
            ),
        )

        # Optimize training
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

        # Step 6: Train!
        print("🏃 Step 6: Training model...")
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer.train()

        # Step 7: Save model
        print("💾 Step 7: Saving model...")
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA adapters
        lora_path = f"{output_dir}/lora"
        model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)

        # Merge and save full model
        merged_path = f"{output_dir}/merged"
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        # Final memory stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        print("🎉 Training Complete!")
        print(f"Training time: {trainer_stats.metrics['train_runtime']} seconds")
        print(f"Peak memory: {used_memory} GB ({used_percentage}%)")

        training_result = {
            "status": "completed",
            "training_stats": trainer_stats.metrics,
            "model_paths": {
                "lora": lora_path,
                "merged": merged_path
            },
            "memory_stats": {
                "start_memory": start_gpu_memory,
                "peak_memory": used_memory,
                "max_memory": max_memory,
                "memory_percentage": used_percentage
            },
            "dataset_info": {
                "samples_used": len(dataset),
                "datasets": dataset_name_or_path
            },
            "training_config": {
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "lora_rank": lora_rank
            }
        }

        # Save training summary
        with open(f"{output_dir}/training_summary.json", "w") as f:
            json.dump(training_result, f, indent=2, default=str)

        print(f"📄 Training summary saved to: {output_dir}/training_summary.json")
        print(f"🤖 Model saved to: {merged_path}")

        return training_result

# Global instance for easy access
_omni_client = None

def get_omni_client(chat_device="auto", embed_device="auto") -> OmniClient:
    """Get or create the global Omni client instance"""
    global _omni_client
    if _omni_client is None:
        _omni_client = OmniClient(chat_device, embed_device)
    return _omni_client

def chat(messages: Union[str, List[Dict]],
         max_tokens: int = 256,
         temperature: float = 0.8,
         top_p: float = 0.9,
         top_k: int = 50,
         stream: bool = False,
         **kwargs) -> Union[str, Iterator[str]]:
    """
    Generate chat completions (llama.cpp style)

    Args:
        messages: Input messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling
        top_k: Top-k sampling
        stream: Enable streaming
        **kwargs: Additional parameters

    Returns:
        Generated response
    """
    client = get_omni_client()
    return client.chat(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stream=stream,
        **kwargs
    )

def embed(input_content: str,
          operation: str = "embed",
          similarity_threshold: float = 0.0,
          **kwargs) -> Dict[str, Any]:
    """
    Process multimodal embeddings with XML tags

    Args:
        input_content: XML with <text>, <image>, <audio> tags
        operation: "embed", "similarity", or "search"
        similarity_threshold: Minimum similarity score
        **kwargs: Additional parameters

    Returns:
        Dictionary with embedding results
    """
    client = get_omni_client()
    return client.embed(
        input_content=input_content,
        operation=operation,
        similarity_threshold=similarity_threshold,
        **kwargs
    )

def train(dataset_name_or_path: Union[str, List[str]],
          output_dir: str = "models/senter-omni-trained",
          max_samples: int = 10000,
          num_epochs: int = 3,
          learning_rate: float = 2e-4,
          batch_size: int = 2,
          gradient_accumulation_steps: int = 4,
          lora_rank: int = 16,
          save_steps: int = 500,
          **kwargs) -> Dict[str, Any]:
    """
    Train a new Senter-Omni multimodal chat model

    Args:
        dataset_name_or_path: Dataset name(s) on HuggingFace or local path(s)
        output_dir: Directory to save the trained model
        max_samples: Maximum number of training samples
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        lora_rank: LoRA rank for fine-tuning
        save_steps: Save model every N steps
        **kwargs: Additional training parameters

    Returns:
        Dictionary with training statistics and model paths

    Example:
        # Train with HuggingFace datasets
        result = omni.train([
            "NousResearch/Hermes-3-Dataset",
            "NousResearch/hermes-function-calling-v1"
        ])

        # Train with local dataset
        result = omni.train("my_dataset.jsonl", max_samples=5000)
    """
    client = OmniClient()  # Create new client for training
    return client.train(
        dataset_name_or_path=dataset_name_or_path,
        output_dir=output_dir,
        max_samples=max_samples,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_rank=lora_rank,
        save_steps=save_steps,
        **kwargs
    )

# Convenience functions
def create_chat_completion(**kwargs):
    """Alias for chat() to match OpenAI API style"""
    return chat(**kwargs)

def generate(**kwargs):
    """Alias for chat() to match llama.cpp style"""
    return chat(**kwargs)

# Example usage functions
def example_chat():
    """Example of chat functionality"""
    print("🤖 Chat Example:")
    response = chat(
        messages="<user>Hello! Tell me about multimodal AI.</user>",
        max_tokens=100,
        temperature=0.7
    )
    print(f"Response: {response}")
    return response

def example_embed():
    """Example of embedding functionality"""
    print("\\n🔍 Embedding Example:")
    result = embed(
        input_content="""
        <text>Artificial intelligence and machine learning</text>
        <image>test_assets/real_test_image.jpg</image>
        <audio>test_assets/pure_tone_440hz.wav</audio>
        """,
        operation="similarity"
    )

    print(f"Modalities found: {result['modalities']}")
    print(f"Embeddings generated: {len(result['embeddings'])}")
    if 'similarities' in result:
        print("Similarities:")
        for pair, score in result['similarities'].items():
            print(f"  {pair}: {score:.3f}")

    return result

if __name__ == "__main__":
    print("🎭 Senter-Omni Unified API")
    print("=" * 60)
    print("Available functions:")
    print("• omni.train() - Train new multimodal chat models")
    print("• omni.chat() - Chat completions with parameters")
    print("• omni.embed() - Multimodal embeddings with XML tags")
    print("• omni.create_chat_completion() - OpenAI-style API")
    print("• omni.generate() - llama.cpp-style generation")

    print("\\n📖 Usage Examples:")
    print("```python")
    print("# Train a new model")
    print("result = omni.train(['NousResearch/Hermes-3-Dataset'], max_samples=5000)")
    print("")
    print("# Chat")
    print("response = omni.chat('<user>Hello!</user>', max_tokens=100, temperature=0.7)")
    print("")
    print("# Embeddings")
    print("result = omni.embed('<text>Hello</text><image>photo.jpg</image>')")
    print("```")
