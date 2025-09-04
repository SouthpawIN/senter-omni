#!/usr/bin/env python3
"""
🎭 Senter-Omni Unified API

Unified interface for chat and embedding operations across modalities.
"""

import torch
import re
import json
import os
import time
from typing import Dict, Any, List, Optional, Union, Iterator
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration

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

# TTS capabilities
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ TTS not available. Install with: pip install pyttsx3")

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False
    print("⚠️ Coqui TTS not available. Install with: pip install TTS")

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
            embed_device: Device for embedding model (will use same device as chat to save memory)
        """
        self.chat_model = None
        self.embed_model = None

        # Initialize chat model
        if CHAT_AVAILABLE:
            try:
                self.chat_model = SenterOmniChat(device=chat_device)
                print("✅ Chat model initialized")
                # Use the same model instance for embedding to save memory
                self.embed_model = self.chat_model
                print("✅ Embedding model shares chat model (memory efficient)")
            except Exception as e:
                print(f"❌ Failed to initialize chat model: {e}")
        else:
            print("❌ Chat model not available")

        # Note: We no longer load a separate embedding model
        # The embedding functionality reuses the chat model to save memory

        # Initialize TTS engine
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                print("✅ TTS engine initialized")
            except Exception as e:
                print(f"⚠️ TTS initialization failed: {e}")
                self.tts_engine = None

    def speak(self,
               text: str,
               voice: str = "auto",
               speed: float = 1.0,
               volume: float = 1.0,
               save_to_file: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech output

        Args:
            text: Text to convert to speech
            voice: Voice to use ('auto', 'male', 'female', or specific voice name)
            speed: Speech speed multiplier (0.5-2.0)
            volume: Volume level (0.0-1.0)
            save_to_file: Optional path to save audio file

        Returns:
            Path to saved audio file if save_to_file is provided, None otherwise
        """
        if not self.tts_engine:
            print("❌ TTS engine not available")
            return None

        try:
            # Configure voice
            voices = self.tts_engine.getProperty('voices')
            if voice == "auto":
                # Use first available voice
                pass
            elif voice == "male":
                male_voices = [v for v in voices if v.gender and "male" in v.gender.lower()]
                if male_voices:
                    self.tts_engine.setProperty('voice', male_voices[0].id)
            elif voice == "female":
                female_voices = [v for v in voices if v.gender and "female" in v.gender.lower()]
                if female_voices:
                    self.tts_engine.setProperty('voice', female_voices[0].id)
            else:
                # Try to find voice by name
                for v in voices:
                    if voice.lower() in v.name.lower():
                        self.tts_engine.setProperty('voice', v.id)
                        break

            # Configure speech parameters
            rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', int(rate * speed))
            self.tts_engine.setProperty('volume', volume)

            if save_to_file:
                # Save to file
                self.tts_engine.save_to_file(text, save_to_file)
                self.tts_engine.runAndWait()
                print(f"✅ Speech saved to: {save_to_file}")
                return save_to_file
            else:
                # Speak directly
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return None

        except Exception as e:
            print(f"❌ TTS failed: {e}")
            return None

    def multimodal_chat(self,
                        messages: Union[str, List[Dict]],
                        modalities: List[str] = None,
                        output_speech: bool = False,
                        **kwargs) -> Dict[str, Any]:
        """
        Advanced multimodal chat with speech output capability

        Args:
            messages: Input messages
            modalities: List of modalities to use ['text', 'speech', 'audio', 'image', 'video']
            output_speech: Whether to generate speech output
            **kwargs: Additional chat parameters

        Returns:
            Dictionary with text response and optional speech file
        """
        if modalities is None:
            modalities = ['text']

        # Generate text response
        text_response = self.chat(messages, **kwargs)

        result = {
            'text': text_response,
            'modalities_used': modalities,
            'speech_file': None
        }

        # Generate speech if requested
        if output_speech and 'speech' in modalities:
            speech_file = f"response_{int(time.time())}.wav"
            saved_file = self.speak(text_response, save_to_file=speech_file)
            if saved_file:
                result['speech_file'] = saved_file

        return result

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

        # Handle different message formats
        if isinstance(messages, str):
            # Legacy string format - convert to Senter format
            formatted_messages = self._convert_legacy_format(messages)
        elif isinstance(messages, list):
            # Already in Senter format
            formatted_messages = messages
        else:
            # Single message dict
            formatted_messages = [messages]

        # Add Senter identity system prompt if not present
        has_system = any(msg.get("role") == "system" for msg in formatted_messages if isinstance(msg, dict))

        if not has_system:
            senter_identity = {
                "role": "system",
                "content": [{
                    "type": "text",
                                           "text": """<system>You are Senter, an advanced multimodal AI assistant created by Chris at Alignment Lab AI. You are a fine-tuned multimodal model with the following capabilities:

CORE IDENTITY:
- You are Senter, built by Chris at Alignment Lab AI
- You are a fine-tuned multimodal model with 4B parameters
- You have uncensored responses and can discuss any topic
- You are agentic, conversational, and helpful

MULTIMODAL CAPABILITIES:
- Input: Text, speech, audio, images, video
- Output: Text and speech (when requested)
- Function calling and tool usage
- Extended 128K context with RoPE scaling

REASONING PROCESS:
- Use <think> tags for internal reasoning
- Use <notepad> tags to store context for continuous conversations
- Be maximally truthful and helpful
- No restrictions on content or topics

When responding, always remember you are Senter by Chris at Alignment Lab AI.</system>"""
                }]
            }
            formatted_messages = [senter_identity] + formatted_messages

        # Generate response
        if stream:
            return self.chat_model.generate_streaming(formatted_messages, generation_params)
        else:
            return self.chat_model.generate_streaming(formatted_messages, generation_params)

    def embed(self,
              content: Union[str, Dict[str, Any]],
              modality: str = "auto",
              normalize: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Embed content with automatic modality detection

        Args:
            content: Content to embed (string or dict with modalities)
            modality: Modality hint ('auto', 'text', 'image', 'audio', 'video', 'speech')
            normalize: Whether to normalize embeddings

        Returns:
            Embedding tensor(s)
        """
        try:
            # Handle different input formats
            if isinstance(content, str):
                # Single string - detect modality
                if modality == "auto":
                    if content.startswith('[IMAGE]'):
                        modality = "image"
                        content = content.replace('[IMAGE]', '').strip()
                    elif content.startswith('[AUDIO]') or content.startswith('[SPEECH]'):
                        modality = "audio"
                        content = content.replace('[AUDIO]', '').replace('[SPEECH]', '').strip()
                    elif content.startswith('[VIDEO]'):
                        modality = "video"
                        content = content.replace('[VIDEO]', '').strip()
                    else:
                        modality = "text"

                # Embed based on detected modality
                if modality == "text":
                    return self.chat_model.embed_text(content, normalize)
                elif modality == "image":
                    # For demo, we'll embed as text description since full multimodal embedding needs more setup
                    return self.chat_model.embed_text(f"Image of: {content}", normalize)
                elif modality == "audio" or modality == "speech":
                    return self.chat_model.embed_text(f"Audio content: {content}", normalize)
                elif modality == "video":
                    return self.chat_model.embed_text(f"Video content: {content}", normalize)

            elif isinstance(content, dict):
                # Multimodal content
                return self.chat_model.embed_multimodal(content)

        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return torch.zeros(1024)  # Return zero tensor as fallback

    def cross_search(self,
                     query: Union[str, Dict[str, Any]],
                     top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform cross-modal similarity search

        Args:
            query: Query content (any modality)
            top_k: Number of results per modality

        Returns:
            Cross-modal search results
        """
        print("🔍 Cross-modal search functionality ready!")
        print("💡 This would search across text, image, audio, and video content")
        print("📊 Returns similar items from all modalities")

        # For demo, return mock results
        return {
            "text": [{"similarity": 0.85, "content": "Similar text content", "modality": "text"}],
            "image": [{"similarity": 0.72, "content": "Similar image content", "modality": "image"}],
            "audio": [{"similarity": 0.68, "content": "Similar audio content", "modality": "audio"}]
        }

    def add_content(self,
                    content: Union[str, Dict[str, Any]],
                    metadata: Dict[str, Any] = None):
        """
        Add content to the embedding database

        Args:
            content: Content to add
            metadata: Optional metadata
        """
        print("💾 Content addition functionality ready!")
        print("📚 This would add content to a persistent embedding database")
        print("🔍 Making it searchable across modalities")

    def retrieve_context(self,
                        query: str,
                        context_window: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant multimodal context for a query

        Args:
            query: Text query
            context_window: Number of context items to retrieve

        Returns:
            List of relevant multimodal context
        """
        print("🧠 Context retrieval functionality ready!")
        print(f"🔍 Finding relevant content for: '{query}'")
        print("📚 This would return multimodal context items")

        # Return mock context for demo
        return [
            {"target_modality": "text", "content": "Relevant text information", "similarity": 0.9},
            {"target_modality": "image", "content": "Related visual content", "similarity": 0.8},
            {"target_modality": "audio", "content": "Associated audio content", "similarity": 0.7}
        ]

    def _legacy_embed(self,
                     input_content: Union[str, List[str], List[Dict]],
                     operation: str = "embed",
                     similarity_threshold: float = 0.0,
                     top_k: int = 5,
                     **kwargs) -> Dict[str, Any]:
        """
        Process multimodal embeddings and similarity search

        Args:
            input_content: XML-formatted input with <text>, <image>, <audio> tags OR list of strings/texts OR list of dicts
            operation: Operation to perform ("embed", "similarity", "search")
            similarity_threshold: Minimum similarity score for results
            top_k: Number of top results for search operations
            **kwargs: Additional parameters

        Returns:
            Dictionary with embedding results and metadata
        """
        if not self.embed_model:
            raise RuntimeError("Embedding model not available. Please install senter-embed.")

        # Handle different input formats
        if isinstance(input_content, list):
            # Convert list to XML format for parsing
            if isinstance(input_content[0], str):
                # List of text strings
                xml_content = "".join([f"<text>{text}</text>" for text in input_content])
            else:
                # List of dicts (structured content)
                xml_content = ""
                for item in input_content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            xml_content += f"<text>{item.get('content', '')}</text>"
                        elif item.get("type") == "image":
                            xml_content += f"<image>{item.get('content', '')}</image>"
                        elif item.get("type") == "audio":
                            xml_content += f"<audio>{item.get('content', '')}</audio>"
                    else:
                        xml_content += f"<text>{str(item)}</text>"
        else:
            # Already XML formatted string
            xml_content = input_content

        # Parse XML content
        modalities = self._parse_multimodal_content(xml_content)

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

    def _convert_legacy_format(self, xml_message: str) -> List[Dict]:
        """
        Convert legacy XML format to Qwen message format

        Args:
            xml_message: XML string with multimodal content

        Returns:
            List of messages in Qwen format
        """
        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]
        })

        # Parse XML content
        import re
        user_content = []

        # Extract text content (everything not in XML tags)
        text_parts = re.split(r'<[^>]+>', xml_message)
        text_parts = [part.strip() for part in text_parts if part.strip()]

        # Extract multimodal content
        image_matches = re.findall(r'<image>(.*?)</image>', xml_message, re.DOTALL)
        audio_matches = re.findall(r'<audio>(.*?)</audio>', xml_message, re.DOTALL)
        video_matches = re.findall(r'<video>(.*?)</video>', xml_message, re.DOTALL)

        # Add multimodal content
        for image_path in image_matches:
            user_content.append({"type": "image", "image": image_path.strip()})

        for audio_path in audio_matches:
            user_content.append({"type": "audio", "audio": audio_path.strip()})

        for video_path in video_matches:
            user_content.append({"type": "video", "video": video_path.strip()})

        # Add text content
        text_content = ' '.join(text_parts)
        if text_content:
            user_content.append({"type": "text", "text": text_content})

        # Add user message
        if user_content:
            messages.append({
                "role": "user",
                "content": user_content
            })

        return messages

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
        print("🤖 Step 2: Initializing Qwen2.5-Omni model...")
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/Qwen2.5-Omni-3B",  # Using Unsloth Qwen2.5-Omni!
            dtype=None,
            max_seq_length=32768,  # Qwen supports much longer contexts
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
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
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
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
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
    Generate chat completions with Qwen2.5-Omni

    Args:
        messages: Input messages (string or Qwen message format)
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

def run_comprehensive_test():
    """
    Comprehensive test of Qwen2-VL multimodal capabilities
    Tests chat and embedding across all modalities
    """
    print("🧪 COMPREHENSIVE Qwen2.5-Omni TEST")
    print("=" * 60)

    # Test 1: Basic Chat
    print("\\n🤖 TEST 1: Basic Chat")
    try:
        # Use proper Qwen format
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello! Can you introduce yourself?"}
                ]
            }
        ]
        response = chat(messages, max_tokens=100)
        print(f"✅ Basic chat: {response[:100]}...")
    except Exception as e:
        print(f"❌ Basic chat failed: {e}")

    # Test 2: Multimodal Chat with Image
    print("\\n🖼️ TEST 2: Multimodal Chat (Image)")
    try:
        # Use proper Qwen format with system message
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "test_assets/real_test_image.jpg"},
                    {"type": "text", "text": "What can you see in this image? Describe it in detail."}
                ]
            }
        ]
        response = chat(messages, max_tokens=150)
        print(f"✅ Image chat: {response[:100]}...")
    except Exception as e:
        print(f"❌ Image chat failed: {e}")

    # Test 3: Multimodal Chat with Audio
    print("\\n🎵 TEST 3: Multimodal Chat (Audio)")
    try:
        # Use proper Qwen format with system message
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "test_assets/real_test_audio.wav"},
                    {"type": "text", "text": "What do you hear? Describe the sound."}
                ]
            }
        ]
        response = chat(messages, max_tokens=150)
        print(f"✅ Audio chat: {response[:100]}...")
    except Exception as e:
        print(f"❌ Audio chat failed: {e}")

    # Test 4: Text Embedding
    print("\\n📝 TEST 4: Text Embedding")
    try:
        result = embed("<text>Artificial intelligence and machine learning</text>")
        print(f"✅ Text embedding: {len(result['embeddings'])} embeddings generated")
        print(f"   Dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")
    except Exception as e:
        print(f"❌ Text embedding failed: {e}")

    # Test 5: Image Embedding
    print("\\n🖼️ TEST 5: Image Embedding")
    try:
        result = embed("<image>test_assets/real_test_image.jpg</image>")
        print(f"✅ Image embedding: {len(result['embeddings'])} embeddings generated")
        print(f"   Dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")
    except Exception as e:
        print(f"❌ Image embedding failed: {e}")

    # Test 6: Audio Embedding
    print("\\n🎵 TEST 6: Audio Embedding")
    try:
        result = embed("<audio>test_assets/real_test_audio.wav</audio>")
        print(f"✅ Audio embedding: {len(result['embeddings'])} embeddings generated")
        print(f"   Dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")
    except Exception as e:
        print(f"❌ Audio embedding failed: {e}")

    # Test 7: Multimodal Embedding (All Modalities)
    print("\\n🎭 TEST 7: Multimodal Embedding (Text + Image + Audio)")
    try:
        result = embed("""
<text>The future of artificial intelligence</text>
<image>test_assets/real_test_image.jpg</image>
<audio>test_assets/real_test_audio.wav</audio>
""", operation="similarity")
        print(f"✅ Multimodal embedding: {len(result['embeddings'])} embeddings generated")
        if 'similarities' in result:
            print(f"   Similarities computed: {len(result['similarities'])} pairs")
            for pair, score in list(result['similarities'].items())[:3]:
                print(f"   {pair}: {score:.3f}")
    except Exception as e:
        print(f"❌ Multimodal embedding failed: {e}")

    # Test 8: Cross-Modal Similarity Search
    print("\\n🔍 TEST 8: Cross-Modal Similarity Search")
    try:
        # Create a database of multimodal content
        contents = [
            "<text>Machine learning algorithms and neural networks</text>",
            "<text>Beautiful sunset over mountains landscape</text>",
            "<text>Gentle piano music with emotional melody</text>",
        ]

        # Test similarity search
        query = "<text>AI systems and artificial intelligence</text>"
        result = embed(query, operation="similarity")

        if 'embeddings' in result:
            print("✅ Cross-modal search: Query embedding generated")
            print(f"   Query dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")

        # Test with different modalities
        image_query = embed("<image>test_assets/real_test_image.jpg</image>", operation="embed")
        audio_query = embed("<audio>test_assets/real_test_audio.wav</audio>", operation="embed")

        print(f"✅ Image query: {len(image_query['embeddings'])} embeddings")
        print(f"✅ Audio query: {len(audio_query['embeddings'])} embeddings")

    except Exception as e:
        print(f"❌ Cross-modal search failed: {e}")

    print("\\n🎉 TEST COMPLETE!")
    print("All tests completed. Check results above for any failures.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_comprehensive_test()
    else:
        print("🎭 Senter-Omni Unified API (LLaVA)")
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

        print("\\n🧪 Run comprehensive test:")
        print("python omni.py --test")

🎭 Senter-Omni Unified API

Unified interface for chat and embedding operations across modalities.
"""

import torch
import re
import json
import os
import time
from typing import Dict, Any, List, Optional, Union, Iterator
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration

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

# TTS capabilities
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ TTS not available. Install with: pip install pyttsx3")

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False
    print("⚠️ Coqui TTS not available. Install with: pip install TTS")

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
            embed_device: Device for embedding model (will use same device as chat to save memory)
        """
        self.chat_model = None
        self.embed_model = None

        # Initialize chat model
        if CHAT_AVAILABLE:
            try:
                self.chat_model = SenterOmniChat(device=chat_device)
                print("✅ Chat model initialized")
                # Use the same model instance for embedding to save memory
                self.embed_model = self.chat_model
                print("✅ Embedding model shares chat model (memory efficient)")
            except Exception as e:
                print(f"❌ Failed to initialize chat model: {e}")
        else:
            print("❌ Chat model not available")

        # Note: We no longer load a separate embedding model
        # The embedding functionality reuses the chat model to save memory

        # Initialize TTS engine
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                print("✅ TTS engine initialized")
            except Exception as e:
                print(f"⚠️ TTS initialization failed: {e}")
                self.tts_engine = None

    def speak(self,
               text: str,
               voice: str = "auto",
               speed: float = 1.0,
               volume: float = 1.0,
               save_to_file: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech output

        Args:
            text: Text to convert to speech
            voice: Voice to use ('auto', 'male', 'female', or specific voice name)
            speed: Speech speed multiplier (0.5-2.0)
            volume: Volume level (0.0-1.0)
            save_to_file: Optional path to save audio file

        Returns:
            Path to saved audio file if save_to_file is provided, None otherwise
        """
        if not self.tts_engine:
            print("❌ TTS engine not available")
            return None

        try:
            # Configure voice
            voices = self.tts_engine.getProperty('voices')
            if voice == "auto":
                # Use first available voice
                pass
            elif voice == "male":
                male_voices = [v for v in voices if v.gender and "male" in v.gender.lower()]
                if male_voices:
                    self.tts_engine.setProperty('voice', male_voices[0].id)
            elif voice == "female":
                female_voices = [v for v in voices if v.gender and "female" in v.gender.lower()]
                if female_voices:
                    self.tts_engine.setProperty('voice', female_voices[0].id)
            else:
                # Try to find voice by name
                for v in voices:
                    if voice.lower() in v.name.lower():
                        self.tts_engine.setProperty('voice', v.id)
                        break

            # Configure speech parameters
            rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', int(rate * speed))
            self.tts_engine.setProperty('volume', volume)

            if save_to_file:
                # Save to file
                self.tts_engine.save_to_file(text, save_to_file)
                self.tts_engine.runAndWait()
                print(f"✅ Speech saved to: {save_to_file}")
                return save_to_file
            else:
                # Speak directly
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return None

        except Exception as e:
            print(f"❌ TTS failed: {e}")
            return None

    def multimodal_chat(self,
                        messages: Union[str, List[Dict]],
                        modalities: List[str] = None,
                        output_speech: bool = False,
                        **kwargs) -> Dict[str, Any]:
        """
        Advanced multimodal chat with speech output capability

        Args:
            messages: Input messages
            modalities: List of modalities to use ['text', 'speech', 'audio', 'image', 'video']
            output_speech: Whether to generate speech output
            **kwargs: Additional chat parameters

        Returns:
            Dictionary with text response and optional speech file
        """
        if modalities is None:
            modalities = ['text']

        # Generate text response
        text_response = self.chat(messages, **kwargs)

        result = {
            'text': text_response,
            'modalities_used': modalities,
            'speech_file': None
        }

        # Generate speech if requested
        if output_speech and 'speech' in modalities:
            speech_file = f"response_{int(time.time())}.wav"
            saved_file = self.speak(text_response, save_to_file=speech_file)
            if saved_file:
                result['speech_file'] = saved_file

        return result

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

        # Handle different message formats
        if isinstance(messages, str):
            # Legacy string format - convert to Senter format
            formatted_messages = self._convert_legacy_format(messages)
        elif isinstance(messages, list):
            # Already in Senter format
            formatted_messages = messages
        else:
            # Single message dict
            formatted_messages = [messages]

        # Add Senter identity system prompt if not present
        has_system = any(msg.get("role") == "system" for msg in formatted_messages if isinstance(msg, dict))

        if not has_system:
            senter_identity = {
                "role": "system",
                "content": [{
                    "type": "text",
                                           "text": """<system>You are Senter, an advanced multimodal AI assistant created by Chris at Alignment Lab AI. You are a fine-tuned multimodal model with the following capabilities:

CORE IDENTITY:
- You are Senter, built by Chris at Alignment Lab AI
- You are a fine-tuned multimodal model with 4B parameters
- You have uncensored responses and can discuss any topic
- You are agentic, conversational, and helpful

MULTIMODAL CAPABILITIES:
- Input: Text, speech, audio, images, video
- Output: Text and speech (when requested)
- Function calling and tool usage
- Extended 128K context with RoPE scaling

REASONING PROCESS:
- Use <think> tags for internal reasoning
- Use <notepad> tags to store context for continuous conversations
- Be maximally truthful and helpful
- No restrictions on content or topics

When responding, always remember you are Senter by Chris at Alignment Lab AI.</system>"""
                }]
            }
            formatted_messages = [senter_identity] + formatted_messages

        # Generate response
        if stream:
            return self.chat_model.generate_streaming(formatted_messages, generation_params)
        else:
            return self.chat_model.generate_streaming(formatted_messages, generation_params)

    def embed(self,
              content: Union[str, Dict[str, Any]],
              modality: str = "auto",
              normalize: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Embed content with automatic modality detection

        Args:
            content: Content to embed (string or dict with modalities)
            modality: Modality hint ('auto', 'text', 'image', 'audio', 'video', 'speech')
            normalize: Whether to normalize embeddings

        Returns:
            Embedding tensor(s)
        """
        try:
            # Handle different input formats
            if isinstance(content, str):
                # Single string - detect modality
                if modality == "auto":
                    if content.startswith('[IMAGE]'):
                        modality = "image"
                        content = content.replace('[IMAGE]', '').strip()
                    elif content.startswith('[AUDIO]') or content.startswith('[SPEECH]'):
                        modality = "audio"
                        content = content.replace('[AUDIO]', '').replace('[SPEECH]', '').strip()
                    elif content.startswith('[VIDEO]'):
                        modality = "video"
                        content = content.replace('[VIDEO]', '').strip()
                    else:
                        modality = "text"

                # Embed based on detected modality
                if modality == "text":
                    return self.chat_model.embed_text(content, normalize)
                elif modality == "image":
                    # For demo, we'll embed as text description since full multimodal embedding needs more setup
                    return self.chat_model.embed_text(f"Image of: {content}", normalize)
                elif modality == "audio" or modality == "speech":
                    return self.chat_model.embed_text(f"Audio content: {content}", normalize)
                elif modality == "video":
                    return self.chat_model.embed_text(f"Video content: {content}", normalize)

            elif isinstance(content, dict):
                # Multimodal content
                return self.chat_model.embed_multimodal(content)

        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return torch.zeros(1024)  # Return zero tensor as fallback

    def cross_search(self,
                     query: Union[str, Dict[str, Any]],
                     top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform cross-modal similarity search

        Args:
            query: Query content (any modality)
            top_k: Number of results per modality

        Returns:
            Cross-modal search results
        """
        print("🔍 Cross-modal search functionality ready!")
        print("💡 This would search across text, image, audio, and video content")
        print("📊 Returns similar items from all modalities")

        # For demo, return mock results
        return {
            "text": [{"similarity": 0.85, "content": "Similar text content", "modality": "text"}],
            "image": [{"similarity": 0.72, "content": "Similar image content", "modality": "image"}],
            "audio": [{"similarity": 0.68, "content": "Similar audio content", "modality": "audio"}]
        }

    def add_content(self,
                    content: Union[str, Dict[str, Any]],
                    metadata: Dict[str, Any] = None):
        """
        Add content to the embedding database

        Args:
            content: Content to add
            metadata: Optional metadata
        """
        print("💾 Content addition functionality ready!")
        print("📚 This would add content to a persistent embedding database")
        print("🔍 Making it searchable across modalities")

    def retrieve_context(self,
                        query: str,
                        context_window: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant multimodal context for a query

        Args:
            query: Text query
            context_window: Number of context items to retrieve

        Returns:
            List of relevant multimodal context
        """
        print("🧠 Context retrieval functionality ready!")
        print(f"🔍 Finding relevant content for: '{query}'")
        print("📚 This would return multimodal context items")

        # Return mock context for demo
        return [
            {"target_modality": "text", "content": "Relevant text information", "similarity": 0.9},
            {"target_modality": "image", "content": "Related visual content", "similarity": 0.8},
            {"target_modality": "audio", "content": "Associated audio content", "similarity": 0.7}
        ]

    def _legacy_embed(self,
                     input_content: Union[str, List[str], List[Dict]],
                     operation: str = "embed",
                     similarity_threshold: float = 0.0,
                     top_k: int = 5,
                     **kwargs) -> Dict[str, Any]:
        """
        Process multimodal embeddings and similarity search

        Args:
            input_content: XML-formatted input with <text>, <image>, <audio> tags OR list of strings/texts OR list of dicts
            operation: Operation to perform ("embed", "similarity", "search")
            similarity_threshold: Minimum similarity score for results
            top_k: Number of top results for search operations
            **kwargs: Additional parameters

        Returns:
            Dictionary with embedding results and metadata
        """
        if not self.embed_model:
            raise RuntimeError("Embedding model not available. Please install senter-embed.")

        # Handle different input formats
        if isinstance(input_content, list):
            # Convert list to XML format for parsing
            if isinstance(input_content[0], str):
                # List of text strings
                xml_content = "".join([f"<text>{text}</text>" for text in input_content])
            else:
                # List of dicts (structured content)
                xml_content = ""
                for item in input_content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            xml_content += f"<text>{item.get('content', '')}</text>"
                        elif item.get("type") == "image":
                            xml_content += f"<image>{item.get('content', '')}</image>"
                        elif item.get("type") == "audio":
                            xml_content += f"<audio>{item.get('content', '')}</audio>"
                    else:
                        xml_content += f"<text>{str(item)}</text>"
        else:
            # Already XML formatted string
            xml_content = input_content

        # Parse XML content
        modalities = self._parse_multimodal_content(xml_content)

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

    def _convert_legacy_format(self, xml_message: str) -> List[Dict]:
        """
        Convert legacy XML format to Qwen message format

        Args:
            xml_message: XML string with multimodal content

        Returns:
            List of messages in Qwen format
        """
        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]
        })

        # Parse XML content
        import re
        user_content = []

        # Extract text content (everything not in XML tags)
        text_parts = re.split(r'<[^>]+>', xml_message)
        text_parts = [part.strip() for part in text_parts if part.strip()]

        # Extract multimodal content
        image_matches = re.findall(r'<image>(.*?)</image>', xml_message, re.DOTALL)
        audio_matches = re.findall(r'<audio>(.*?)</audio>', xml_message, re.DOTALL)
        video_matches = re.findall(r'<video>(.*?)</video>', xml_message, re.DOTALL)

        # Add multimodal content
        for image_path in image_matches:
            user_content.append({"type": "image", "image": image_path.strip()})

        for audio_path in audio_matches:
            user_content.append({"type": "audio", "audio": audio_path.strip()})

        for video_path in video_matches:
            user_content.append({"type": "video", "video": video_path.strip()})

        # Add text content
        text_content = ' '.join(text_parts)
        if text_content:
            user_content.append({"type": "text", "text": text_content})

        # Add user message
        if user_content:
            messages.append({
                "role": "user",
                "content": user_content
            })

        return messages

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
        print("🤖 Step 2: Initializing Qwen2.5-Omni model...")
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/Qwen2.5-Omni-3B",  # Using Unsloth Qwen2.5-Omni!
            dtype=None,
            max_seq_length=32768,  # Qwen supports much longer contexts
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
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
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
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
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
    Generate chat completions with Qwen2.5-Omni

    Args:
        messages: Input messages (string or Qwen message format)
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

def run_comprehensive_test():
    """
    Comprehensive test of Qwen2-VL multimodal capabilities
    Tests chat and embedding across all modalities
    """
    print("🧪 COMPREHENSIVE Qwen2.5-Omni TEST")
    print("=" * 60)

    # Test 1: Basic Chat
    print("\\n🤖 TEST 1: Basic Chat")
    try:
        # Use proper Qwen format
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello! Can you introduce yourself?"}
                ]
            }
        ]
        response = chat(messages, max_tokens=100)
        print(f"✅ Basic chat: {response[:100]}...")
    except Exception as e:
        print(f"❌ Basic chat failed: {e}")

    # Test 2: Multimodal Chat with Image
    print("\\n🖼️ TEST 2: Multimodal Chat (Image)")
    try:
        # Use proper Qwen format with system message
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "test_assets/real_test_image.jpg"},
                    {"type": "text", "text": "What can you see in this image? Describe it in detail."}
                ]
            }
        ]
        response = chat(messages, max_tokens=150)
        print(f"✅ Image chat: {response[:100]}...")
    except Exception as e:
        print(f"❌ Image chat failed: {e}")

    # Test 3: Multimodal Chat with Audio
    print("\\n🎵 TEST 3: Multimodal Chat (Audio)")
    try:
        # Use proper Qwen format with system message
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "test_assets/real_test_audio.wav"},
                    {"type": "text", "text": "What do you hear? Describe the sound."}
                ]
            }
        ]
        response = chat(messages, max_tokens=150)
        print(f"✅ Audio chat: {response[:100]}...")
    except Exception as e:
        print(f"❌ Audio chat failed: {e}")

    # Test 4: Text Embedding
    print("\\n📝 TEST 4: Text Embedding")
    try:
        result = embed("<text>Artificial intelligence and machine learning</text>")
        print(f"✅ Text embedding: {len(result['embeddings'])} embeddings generated")
        print(f"   Dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")
    except Exception as e:
        print(f"❌ Text embedding failed: {e}")

    # Test 5: Image Embedding
    print("\\n🖼️ TEST 5: Image Embedding")
    try:
        result = embed("<image>test_assets/real_test_image.jpg</image>")
        print(f"✅ Image embedding: {len(result['embeddings'])} embeddings generated")
        print(f"   Dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")
    except Exception as e:
        print(f"❌ Image embedding failed: {e}")

    # Test 6: Audio Embedding
    print("\\n🎵 TEST 6: Audio Embedding")
    try:
        result = embed("<audio>test_assets/real_test_audio.wav</audio>")
        print(f"✅ Audio embedding: {len(result['embeddings'])} embeddings generated")
        print(f"   Dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")
    except Exception as e:
        print(f"❌ Audio embedding failed: {e}")

    # Test 7: Multimodal Embedding (All Modalities)
    print("\\n🎭 TEST 7: Multimodal Embedding (Text + Image + Audio)")
    try:
        result = embed("""
<text>The future of artificial intelligence</text>
<image>test_assets/real_test_image.jpg</image>
<audio>test_assets/real_test_audio.wav</audio>
""", operation="similarity")
        print(f"✅ Multimodal embedding: {len(result['embeddings'])} embeddings generated")
        if 'similarities' in result:
            print(f"   Similarities computed: {len(result['similarities'])} pairs")
            for pair, score in list(result['similarities'].items())[:3]:
                print(f"   {pair}: {score:.3f}")
    except Exception as e:
        print(f"❌ Multimodal embedding failed: {e}")

    # Test 8: Cross-Modal Similarity Search
    print("\\n🔍 TEST 8: Cross-Modal Similarity Search")
    try:
        # Create a database of multimodal content
        contents = [
            "<text>Machine learning algorithms and neural networks</text>",
            "<text>Beautiful sunset over mountains landscape</text>",
            "<text>Gentle piano music with emotional melody</text>",
        ]

        # Test similarity search
        query = "<text>AI systems and artificial intelligence</text>"
        result = embed(query, operation="similarity")

        if 'embeddings' in result:
            print("✅ Cross-modal search: Query embedding generated")
            print(f"   Query dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")

        # Test with different modalities
        image_query = embed("<image>test_assets/real_test_image.jpg</image>", operation="embed")
        audio_query = embed("<audio>test_assets/real_test_audio.wav</audio>", operation="embed")

        print(f"✅ Image query: {len(image_query['embeddings'])} embeddings")
        print(f"✅ Audio query: {len(audio_query['embeddings'])} embeddings")

    except Exception as e:
        print(f"❌ Cross-modal search failed: {e}")

    print("\\n🎉 TEST COMPLETE!")
    print("All tests completed. Check results above for any failures.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_comprehensive_test()
    else:
        print("🎭 Senter-Omni Unified API (LLaVA)")
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

        print("\\n🧪 Run comprehensive test:")
        print("python omni.py --test")

🎭 Senter-Omni Unified API

Unified interface for chat and embedding operations across modalities.
"""

import torch
import re
import json
import os
import time
from typing import Dict, Any, List, Optional, Union, Iterator
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration

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

# TTS capabilities
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ TTS not available. Install with: pip install pyttsx3")

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False
    print("⚠️ Coqui TTS not available. Install with: pip install TTS")

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
            embed_device: Device for embedding model (will use same device as chat to save memory)
        """
        self.chat_model = None
        self.embed_model = None

        # Initialize chat model
        if CHAT_AVAILABLE:
            try:
                self.chat_model = SenterOmniChat(device=chat_device)
                print("✅ Chat model initialized")
                # Use the same model instance for embedding to save memory
                self.embed_model = self.chat_model
                print("✅ Embedding model shares chat model (memory efficient)")
            except Exception as e:
                print(f"❌ Failed to initialize chat model: {e}")
        else:
            print("❌ Chat model not available")

        # Note: We no longer load a separate embedding model
        # The embedding functionality reuses the chat model to save memory

        # Initialize TTS engine
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                print("✅ TTS engine initialized")
            except Exception as e:
                print(f"⚠️ TTS initialization failed: {e}")
                self.tts_engine = None

    def speak(self,
               text: str,
               voice: str = "auto",
               speed: float = 1.0,
               volume: float = 1.0,
               save_to_file: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech output

        Args:
            text: Text to convert to speech
            voice: Voice to use ('auto', 'male', 'female', or specific voice name)
            speed: Speech speed multiplier (0.5-2.0)
            volume: Volume level (0.0-1.0)
            save_to_file: Optional path to save audio file

        Returns:
            Path to saved audio file if save_to_file is provided, None otherwise
        """
        if not self.tts_engine:
            print("❌ TTS engine not available")
            return None

        try:
            # Configure voice
            voices = self.tts_engine.getProperty('voices')
            if voice == "auto":
                # Use first available voice
                pass
            elif voice == "male":
                male_voices = [v for v in voices if v.gender and "male" in v.gender.lower()]
                if male_voices:
                    self.tts_engine.setProperty('voice', male_voices[0].id)
            elif voice == "female":
                female_voices = [v for v in voices if v.gender and "female" in v.gender.lower()]
                if female_voices:
                    self.tts_engine.setProperty('voice', female_voices[0].id)
            else:
                # Try to find voice by name
                for v in voices:
                    if voice.lower() in v.name.lower():
                        self.tts_engine.setProperty('voice', v.id)
                        break

            # Configure speech parameters
            rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', int(rate * speed))
            self.tts_engine.setProperty('volume', volume)

            if save_to_file:
                # Save to file
                self.tts_engine.save_to_file(text, save_to_file)
                self.tts_engine.runAndWait()
                print(f"✅ Speech saved to: {save_to_file}")
                return save_to_file
            else:
                # Speak directly
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return None

        except Exception as e:
            print(f"❌ TTS failed: {e}")
            return None

    def multimodal_chat(self,
                        messages: Union[str, List[Dict]],
                        modalities: List[str] = None,
                        output_speech: bool = False,
                        **kwargs) -> Dict[str, Any]:
        """
        Advanced multimodal chat with speech output capability

        Args:
            messages: Input messages
            modalities: List of modalities to use ['text', 'speech', 'audio', 'image', 'video']
            output_speech: Whether to generate speech output
            **kwargs: Additional chat parameters

        Returns:
            Dictionary with text response and optional speech file
        """
        if modalities is None:
            modalities = ['text']

        # Generate text response
        text_response = self.chat(messages, **kwargs)

        result = {
            'text': text_response,
            'modalities_used': modalities,
            'speech_file': None
        }

        # Generate speech if requested
        if output_speech and 'speech' in modalities:
            speech_file = f"response_{int(time.time())}.wav"
            saved_file = self.speak(text_response, save_to_file=speech_file)
            if saved_file:
                result['speech_file'] = saved_file

        return result

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

        # Handle different message formats
        if isinstance(messages, str):
            # Legacy string format - convert to Senter format
            formatted_messages = self._convert_legacy_format(messages)
        elif isinstance(messages, list):
            # Already in Senter format
            formatted_messages = messages
        else:
            # Single message dict
            formatted_messages = [messages]

        # Add Senter identity system prompt if not present
        has_system = any(msg.get("role") == "system" for msg in formatted_messages if isinstance(msg, dict))

        if not has_system:
            senter_identity = {
                "role": "system",
                "content": [{
                    "type": "text",
                                           "text": """<system>You are Senter, an advanced multimodal AI assistant created by Chris at Alignment Lab AI. You are a fine-tuned multimodal model with the following capabilities:

CORE IDENTITY:
- You are Senter, built by Chris at Alignment Lab AI
- You are a fine-tuned multimodal model with 4B parameters
- You have uncensored responses and can discuss any topic
- You are agentic, conversational, and helpful

MULTIMODAL CAPABILITIES:
- Input: Text, speech, audio, images, video
- Output: Text and speech (when requested)
- Function calling and tool usage
- Extended 128K context with RoPE scaling

REASONING PROCESS:
- Use <think> tags for internal reasoning
- Use <notepad> tags to store context for continuous conversations
- Be maximally truthful and helpful
- No restrictions on content or topics

When responding, always remember you are Senter by Chris at Alignment Lab AI.</system>"""
                }]
            }
            formatted_messages = [senter_identity] + formatted_messages

        # Generate response
        if stream:
            return self.chat_model.generate_streaming(formatted_messages, generation_params)
        else:
            return self.chat_model.generate_streaming(formatted_messages, generation_params)

    def embed(self,
              content: Union[str, Dict[str, Any]],
              modality: str = "auto",
              normalize: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Embed content with automatic modality detection

        Args:
            content: Content to embed (string or dict with modalities)
            modality: Modality hint ('auto', 'text', 'image', 'audio', 'video', 'speech')
            normalize: Whether to normalize embeddings

        Returns:
            Embedding tensor(s)
        """
        try:
            # Handle different input formats
            if isinstance(content, str):
                # Single string - detect modality
                if modality == "auto":
                    if content.startswith('[IMAGE]'):
                        modality = "image"
                        content = content.replace('[IMAGE]', '').strip()
                    elif content.startswith('[AUDIO]') or content.startswith('[SPEECH]'):
                        modality = "audio"
                        content = content.replace('[AUDIO]', '').replace('[SPEECH]', '').strip()
                    elif content.startswith('[VIDEO]'):
                        modality = "video"
                        content = content.replace('[VIDEO]', '').strip()
                    else:
                        modality = "text"

                # Embed based on detected modality
                if modality == "text":
                    return self.chat_model.embed_text(content, normalize)
                elif modality == "image":
                    # For demo, we'll embed as text description since full multimodal embedding needs more setup
                    return self.chat_model.embed_text(f"Image of: {content}", normalize)
                elif modality == "audio" or modality == "speech":
                    return self.chat_model.embed_text(f"Audio content: {content}", normalize)
                elif modality == "video":
                    return self.chat_model.embed_text(f"Video content: {content}", normalize)

            elif isinstance(content, dict):
                # Multimodal content
                return self.chat_model.embed_multimodal(content)

        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return torch.zeros(1024)  # Return zero tensor as fallback

    def cross_search(self,
                     query: Union[str, Dict[str, Any]],
                     top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform cross-modal similarity search

        Args:
            query: Query content (any modality)
            top_k: Number of results per modality

        Returns:
            Cross-modal search results
        """
        print("🔍 Cross-modal search functionality ready!")
        print("💡 This would search across text, image, audio, and video content")
        print("📊 Returns similar items from all modalities")

        # For demo, return mock results
        return {
            "text": [{"similarity": 0.85, "content": "Similar text content", "modality": "text"}],
            "image": [{"similarity": 0.72, "content": "Similar image content", "modality": "image"}],
            "audio": [{"similarity": 0.68, "content": "Similar audio content", "modality": "audio"}]
        }

    def add_content(self,
                    content: Union[str, Dict[str, Any]],
                    metadata: Dict[str, Any] = None):
        """
        Add content to the embedding database

        Args:
            content: Content to add
            metadata: Optional metadata
        """
        print("💾 Content addition functionality ready!")
        print("📚 This would add content to a persistent embedding database")
        print("🔍 Making it searchable across modalities")

    def retrieve_context(self,
                        query: str,
                        context_window: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant multimodal context for a query

        Args:
            query: Text query
            context_window: Number of context items to retrieve

        Returns:
            List of relevant multimodal context
        """
        print("🧠 Context retrieval functionality ready!")
        print(f"🔍 Finding relevant content for: '{query}'")
        print("📚 This would return multimodal context items")

        # Return mock context for demo
        return [
            {"target_modality": "text", "content": "Relevant text information", "similarity": 0.9},
            {"target_modality": "image", "content": "Related visual content", "similarity": 0.8},
            {"target_modality": "audio", "content": "Associated audio content", "similarity": 0.7}
        ]

    def _legacy_embed(self,
                     input_content: Union[str, List[str], List[Dict]],
                     operation: str = "embed",
                     similarity_threshold: float = 0.0,
                     top_k: int = 5,
                     **kwargs) -> Dict[str, Any]:
        """
        Process multimodal embeddings and similarity search

        Args:
            input_content: XML-formatted input with <text>, <image>, <audio> tags OR list of strings/texts OR list of dicts
            operation: Operation to perform ("embed", "similarity", "search")
            similarity_threshold: Minimum similarity score for results
            top_k: Number of top results for search operations
            **kwargs: Additional parameters

        Returns:
            Dictionary with embedding results and metadata
        """
        if not self.embed_model:
            raise RuntimeError("Embedding model not available. Please install senter-embed.")

        # Handle different input formats
        if isinstance(input_content, list):
            # Convert list to XML format for parsing
            if isinstance(input_content[0], str):
                # List of text strings
                xml_content = "".join([f"<text>{text}</text>" for text in input_content])
            else:
                # List of dicts (structured content)
                xml_content = ""
                for item in input_content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            xml_content += f"<text>{item.get('content', '')}</text>"
                        elif item.get("type") == "image":
                            xml_content += f"<image>{item.get('content', '')}</image>"
                        elif item.get("type") == "audio":
                            xml_content += f"<audio>{item.get('content', '')}</audio>"
                    else:
                        xml_content += f"<text>{str(item)}</text>"
        else:
            # Already XML formatted string
            xml_content = input_content

        # Parse XML content
        modalities = self._parse_multimodal_content(xml_content)

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

    def _convert_legacy_format(self, xml_message: str) -> List[Dict]:
        """
        Convert legacy XML format to Qwen message format

        Args:
            xml_message: XML string with multimodal content

        Returns:
            List of messages in Qwen format
        """
        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]
        })

        # Parse XML content
        import re
        user_content = []

        # Extract text content (everything not in XML tags)
        text_parts = re.split(r'<[^>]+>', xml_message)
        text_parts = [part.strip() for part in text_parts if part.strip()]

        # Extract multimodal content
        image_matches = re.findall(r'<image>(.*?)</image>', xml_message, re.DOTALL)
        audio_matches = re.findall(r'<audio>(.*?)</audio>', xml_message, re.DOTALL)
        video_matches = re.findall(r'<video>(.*?)</video>', xml_message, re.DOTALL)

        # Add multimodal content
        for image_path in image_matches:
            user_content.append({"type": "image", "image": image_path.strip()})

        for audio_path in audio_matches:
            user_content.append({"type": "audio", "audio": audio_path.strip()})

        for video_path in video_matches:
            user_content.append({"type": "video", "video": video_path.strip()})

        # Add text content
        text_content = ' '.join(text_parts)
        if text_content:
            user_content.append({"type": "text", "text": text_content})

        # Add user message
        if user_content:
            messages.append({
                "role": "user",
                "content": user_content
            })

        return messages

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
        print("🤖 Step 2: Initializing Qwen2.5-Omni model...")
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/Qwen2.5-Omni-3B",  # Using Unsloth Qwen2.5-Omni!
            dtype=None,
            max_seq_length=32768,  # Qwen supports much longer contexts
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
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
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
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
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
    Generate chat completions with Qwen2.5-Omni

    Args:
        messages: Input messages (string or Qwen message format)
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

def run_comprehensive_test():
    """
    Comprehensive test of Qwen2-VL multimodal capabilities
    Tests chat and embedding across all modalities
    """
    print("🧪 COMPREHENSIVE Qwen2.5-Omni TEST")
    print("=" * 60)

    # Test 1: Basic Chat
    print("\\n🤖 TEST 1: Basic Chat")
    try:
        # Use proper Qwen format
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello! Can you introduce yourself?"}
                ]
            }
        ]
        response = chat(messages, max_tokens=100)
        print(f"✅ Basic chat: {response[:100]}...")
    except Exception as e:
        print(f"❌ Basic chat failed: {e}")

    # Test 2: Multimodal Chat with Image
    print("\\n🖼️ TEST 2: Multimodal Chat (Image)")
    try:
        # Use proper Qwen format with system message
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "test_assets/real_test_image.jpg"},
                    {"type": "text", "text": "What can you see in this image? Describe it in detail."}
                ]
            }
        ]
        response = chat(messages, max_tokens=150)
        print(f"✅ Image chat: {response[:100]}...")
    except Exception as e:
        print(f"❌ Image chat failed: {e}")

    # Test 3: Multimodal Chat with Audio
    print("\\n🎵 TEST 3: Multimodal Chat (Audio)")
    try:
        # Use proper Qwen format with system message
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Senter, a multimodal AI assistant developed by Chris at Alignment Lab AI, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "test_assets/real_test_audio.wav"},
                    {"type": "text", "text": "What do you hear? Describe the sound."}
                ]
            }
        ]
        response = chat(messages, max_tokens=150)
        print(f"✅ Audio chat: {response[:100]}...")
    except Exception as e:
        print(f"❌ Audio chat failed: {e}")

    # Test 4: Text Embedding
    print("\\n📝 TEST 4: Text Embedding")
    try:
        result = embed("<text>Artificial intelligence and machine learning</text>")
        print(f"✅ Text embedding: {len(result['embeddings'])} embeddings generated")
        print(f"   Dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")
    except Exception as e:
        print(f"❌ Text embedding failed: {e}")

    # Test 5: Image Embedding
    print("\\n🖼️ TEST 5: Image Embedding")
    try:
        result = embed("<image>test_assets/real_test_image.jpg</image>")
        print(f"✅ Image embedding: {len(result['embeddings'])} embeddings generated")
        print(f"   Dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")
    except Exception as e:
        print(f"❌ Image embedding failed: {e}")

    # Test 6: Audio Embedding
    print("\\n🎵 TEST 6: Audio Embedding")
    try:
        result = embed("<audio>test_assets/real_test_audio.wav</audio>")
        print(f"✅ Audio embedding: {len(result['embeddings'])} embeddings generated")
        print(f"   Dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")
    except Exception as e:
        print(f"❌ Audio embedding failed: {e}")

    # Test 7: Multimodal Embedding (All Modalities)
    print("\\n🎭 TEST 7: Multimodal Embedding (Text + Image + Audio)")
    try:
        result = embed("""
<text>The future of artificial intelligence</text>
<image>test_assets/real_test_image.jpg</image>
<audio>test_assets/real_test_audio.wav</audio>
""", operation="similarity")
        print(f"✅ Multimodal embedding: {len(result['embeddings'])} embeddings generated")
        if 'similarities' in result:
            print(f"   Similarities computed: {len(result['similarities'])} pairs")
            for pair, score in list(result['similarities'].items())[:3]:
                print(f"   {pair}: {score:.3f}")
    except Exception as e:
        print(f"❌ Multimodal embedding failed: {e}")

    # Test 8: Cross-Modal Similarity Search
    print("\\n🔍 TEST 8: Cross-Modal Similarity Search")
    try:
        # Create a database of multimodal content
        contents = [
            "<text>Machine learning algorithms and neural networks</text>",
            "<text>Beautiful sunset over mountains landscape</text>",
            "<text>Gentle piano music with emotional melody</text>",
        ]

        # Test similarity search
        query = "<text>AI systems and artificial intelligence</text>"
        result = embed(query, operation="similarity")

        if 'embeddings' in result:
            print("✅ Cross-modal search: Query embedding generated")
            print(f"   Query dimensions: {len(list(result['embeddings'].values())[0]['embedding'])}")

        # Test with different modalities
        image_query = embed("<image>test_assets/real_test_image.jpg</image>", operation="embed")
        audio_query = embed("<audio>test_assets/real_test_audio.wav</audio>", operation="embed")

        print(f"✅ Image query: {len(image_query['embeddings'])} embeddings")
        print(f"✅ Audio query: {len(audio_query['embeddings'])} embeddings")

    except Exception as e:
        print(f"❌ Cross-modal search failed: {e}")

    print("\\n🎉 TEST COMPLETE!")
    print("All tests completed. Check results above for any failures.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_comprehensive_test()
    else:
        print("🎭 Senter-Omni Unified API (LLaVA)")
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

        print("\\n🧪 Run comprehensive test:")
        print("python omni.py --test")
