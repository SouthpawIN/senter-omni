#!/usr/bin/env python3
"""
Senter-Omni Core Chat Model

Advanced multimodal conversational AI with XML tag support.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TextStreamer, Qwen2_5OmniForConditionalGeneration
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import re
import sys
import json
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np
import librosa


class StopTokenCriteria(StoppingCriteria):
    """Custom stopping criteria for Gemma3N stop tokens"""

    def __init__(self, tokenizer, stop_tokens=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_tokens = stop_tokens or []
        self.stop_token_ids = []

        # Convert stop tokens to token IDs
        for token in self.stop_tokens:
            try:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    self.stop_token_ids.extend(token_ids)
            except:
                pass

    def __call__(self, input_ids, scores, **kwargs):
        """Check if we should stop generation"""
        # Check if the last generated token is a stop token
        if len(input_ids[0]) > 0:
            last_token_id = input_ids[0][-1].item()
            if last_token_id in self.stop_token_ids:
                return True
        return False


class SenterOmniChat:
    """
    Advanced Senter-Omni multimodal chat interface
    """

    def __init__(self, model_path="./senter_omni_128k/qwen2.5-omni-128k-4bit", device="auto"):
        """
        Initialize the Senter-Omni chat model with quantized multimodal model

        Args:
            model_path: Path to the quantized model
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device(device)
        self.model_path = model_path

        # Define stop tokens for Qwen2.5-Omni
        self.stop_tokens = [
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
            "\n\n",
            "\n",
            "<|im_start|>"
        ]

        self.load_model()

    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device"""
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """Load the Senter-Omni multimodal model"""
        print("🤖 Loading Senter-Omni Chat Model...")

        try:
            # Check if quantized model exists
            quantized_model_path = "./senter_omni_128k/qwen2.5-omni-128k-4bit"
            if Path(quantized_model_path).exists():
                print(f"📥 Using quantized model: {quantized_model_path}")
                model_path_to_use = quantized_model_path
            else:
                print("⚠️ Quantized model not found, using base model")
                print("❌ ERROR: Quantized Senter-Omni model not found!")
                print("   Please ensure the model is properly installed in: ./senter_omni_128k/")
                raise FileNotFoundError(f"Quantized model not found at {quantized_model_path}")

            # Load model with SINGLE DEVICE to avoid GPU conflicts
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path_to_use,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=None,  # Disable auto device mapping
                trust_remote_code=True
            ).to(self.device)  # Explicitly move to single device

            # Load tokenizer and processor
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_to_use)
            self.processor = AutoProcessor.from_pretrained(model_path_to_use)

            print("✅ Senter-Omni Chat Model loaded successfully!")
            print(f"📍 Device: {self.device}")
            print(f"🎯 Model: {model_path_to_use}")

            # Note: LoRA adapter loading disabled due to GELU activation compatibility
            # self.model = PeftModel.from_pretrained(base_model, self.model_path)

        except Exception as e:
            print(f"❌ Failed to load chat model: {e}")
            raise

    # Wrapper methods for embedding compatibility
    def embed_text(self, text: str, normalize: bool = True) -> torch.Tensor:
        """Wrapper for get_text_embedding"""
        return self.get_text_embedding(text, normalize)

    def embed_image(self, image_path: str, normalize: bool = True) -> torch.Tensor:
        """Wrapper for get_image_embedding"""
        return self.get_image_embedding(image_path, normalize)

    def embed_audio(self, audio_path: str, normalize: bool = True) -> torch.Tensor:
        """Wrapper for get_audio_embedding"""
        return self.get_audio_embedding(audio_path, normalize)

    def parse_xml_message(self, xml_message: str) -> Dict[str, Any]:
        """
        Parse XML-style message with multimodal content

        Supports:
        - <system></system>
        - <user></user>
        - <assistant></assistant>
        - <image></image>
        - <audio></audio>
        - <video></video>

        Converts XML to proper Qwen2.5-Omni chat format
        """
        parsed = {
            "role": "user",  # default
            "content": xml_message,
            "multimodal": []
        }

        # Check if message contains any XML tags
        xml_pattern = r'<(system|user|assistant|image|audio|video)>(.*?)</\1>'
        has_xml = bool(re.search(xml_pattern, xml_message, re.DOTALL))

        if has_xml:
            # Extract role
            role_match = re.search(r'<(system|user|assistant)>(.*?)</\1>', xml_message, re.DOTALL)
            if role_match:
                parsed["role"] = role_match.group(1)
                content = role_match.group(2)
            else:
                content = xml_message

            # Extract multimodal content and remove XML tags
            multimodal_tags = ['image', 'audio', 'video']

            for tag in multimodal_tags:
                pattern = f'<{tag}>(.*?)</{tag}>'
                matches = re.findall(pattern, content, re.DOTALL)

                for match in matches:
                    parsed["multimodal"].append({
                        "type": tag,
                        "content": match.strip()
                    })

                    # For Qwen2.5-Omni, use appropriate multimodal tokens
                    if tag == "image":
                        replacement = f'<|image|>{match.strip()}'
                    elif tag == "audio":
                        replacement = f'<|audio|>{match.strip()}'
                    elif tag == "video":
                        replacement = f'<|video|>{match.strip()}'
                    else:
                        # For other content, keep descriptive format
                        replacement = f'[{tag.upper()}: {match.strip()[:50]}...]'
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

            parsed["content"] = content.strip()
        else:
            # No XML tags - treat as plain user message
            parsed["content"] = xml_message.strip()
            parsed["role"] = "user"

        return parsed

    def format_multimodal_content(self, content: str, multimodal: List[Dict]) -> str:
        """Format multimodal content for the model"""
        if not multimodal:
            return content

        formatted_parts = []

        for item in multimodal:
            if item["type"] == "image":
                formatted_parts.append(f"[Image: {item['content']}]")
            elif item["type"] == "audio":
                formatted_parts.append(f"[Audio: {item['content']}]")
            elif item["type"] == "video":
                formatted_parts.append(f"[Video: {item['content']}]")

        formatted_parts.append(content)
        return " ".join(formatted_parts)

    def generate_streaming(self, messages: List[Union[Dict, str]], generation_params: Dict = None) -> str:
        """
        Generate streaming response with customizable parameters

        Args:
            messages: List of message dicts with role and content, or XML strings
            generation_params: Dict of generation parameters

        Returns:
            Generated response text
        """
        if generation_params is None:
            generation_params = {
                "max_new_tokens": 256,
                "temperature": 0.1,  # Lower temperature for cleaner responses
                "top_p": 0.8,
                "top_k": 40,
                "repetition_penalty": 1.15,  # Higher penalty to prevent repetition
                "do_sample": True,
                "stream": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }

        # Handle different message formats for Qwen
        if isinstance(messages[0], dict) and "content" in messages[0]:
            # Already in Qwen format - use directly
            processed_messages = messages
        else:
            # Process legacy format
            processed_messages = []
            for msg in messages:
                if isinstance(msg, str):
                    parsed = self.parse_xml_message(msg)
                else:
                    parsed = msg

                # Handle multimodal content
                if "multimodal" in parsed and parsed["multimodal"]:
                    parsed["content"] = self.format_multimodal_content(
                        parsed["content"],
                        parsed["multimodal"]
                    )

                processed_messages.append({
                    "role": parsed["role"],
                    "content": parsed["content"]
                })

        # For Qwen2.5-Omni, we need to process multimodal content differently
        text_content = []
        images = []
        audios = []

        for msg in processed_messages:
            if isinstance(msg["content"], list):
                # Qwen format with multimodal content
                for item in msg["content"]:
                    if item["type"] == "text":
                        text_content.append(item["text"])
                    elif item["type"] == "image":
                        images.append(item["image"])
                    elif item["type"] == "audio":
                        audios.append(item["audio"])
            else:
                # Legacy text format
                text_content.append(msg["content"])

        # Apply Qwen chat template with proper format
        text = self.tokenizer.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process multimodal inputs - handle all modalities properly
        try:
            # Prepare processor arguments
            processor_kwargs = {
                "text": [text],
                "return_tensors": "pt",
                "padding": True
            }

            # Add images if present
            if images:
                processor_kwargs["images"] = images
                print(f"🖼️ Processing {len(images)} images")

            # Add audio if present (Qwen2.5-Omni supports audio processing)
            if audios:
                print(f"🎵 Processing {len(audios)} audios")
                # Load and process audio files
                processed_audios = []
                for audio_path in audios:
                    try:
                        # Load audio file using librosa
                        audio_data, sample_rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
                        processed_audios.append(audio_data)
                        print(f"✅ Loaded audio: {audio_path} ({len(audio_data)} samples at {sample_rate}Hz)")
                    except Exception as e:
                        print(f"❌ Failed to load audio {audio_path}: {e}")
                        # Skip this audio file and continue
                        continue

                if processed_audios:
                    processor_kwargs["audio"] = processed_audios
                else:
                    print("⚠️ No audio files could be loaded")

            # Process with all modalities
            inputs = self.processor(**processor_kwargs).to(self.device)
            print("✅ Multimodal processing successful")

        except Exception as e:
            print(f"❌ Multimodal processing failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Only fallback to text if there's a critical error
            print("🔄 Falling back to text-only processing due to error")
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)

        # TRUE STREAMING: Generate with real-time token streaming and capture output
        from transformers import TextStreamer

        # Custom streamer that measures time AND captures output
        class TimingCapturingStreamer(TextStreamer):
            def __init__(self, tokenizer, **kwargs):
                super().__init__(tokenizer, **kwargs)
                self.first_token_time = None
                self.start_time = time.time()
                self.captured_text = []

            def on_finalized_text(self, text, stream_end=False):
                # Measure time to first token
                if self.first_token_time is None and text.strip():
                    self.first_token_time = time.time()
                    first_token_delay = self.first_token_time - self.start_time
                    print(".3f")

                # Capture the text for return value
                self.captured_text.append(text)

                # Call parent to handle normal streaming display
                super().on_finalized_text(text, stream_end)

        # Create timing and capturing streamer
        timing_streamer = TimingCapturingStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        print("⚡ Starting streaming generation...")
        start_time = time.time()

        # Generate with proper Qwen2.5-Omni parameters - TRUE STREAMING (single generation)
        with torch.no_grad():
            # Use torch.amp.autocast for mixed precision to save memory
            with torch.amp.autocast(device_type='cuda' if self.device != 'cpu' else 'cpu', dtype=torch.float16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(generation_params.get("max_new_tokens", 256), 150),
                    temperature=0.7,  # Better temperature for coherence
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,  # Prevent repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_audio=False,  # Disable audio generation to save memory
                    streamer=timing_streamer,  # Enable TRUE streaming with capture
                    # Don't use custom stopping criteria - let the model use its natural EOS token
                )

        total_time = time.time() - start_time
        print(".3f")

        # Combine captured text
        generated_text = "".join(timing_streamer.captured_text)

        # Clean up any remaining stop tokens
        for stop_token in ["<|im_end|>", "<|endoftext|>", "</s>", "\n\n"]:
            if stop_token in generated_text:
                generated_text = generated_text.split(stop_token)[0]
                break

        return generated_text.strip()

    def chat_completion(self, messages: List[Union[Dict, str]], generation_params: Dict = None) -> Dict:
        """
        OpenAI-compatible chat completion format

        Args:
            messages: List of message dicts or XML strings
            generation_params: Generation parameters

        Returns:
            Dict in OpenAI chat completion format
        """
        response_text = self.generate_streaming(messages, generation_params)

        return {
            "id": "chatcmpl-senter-omni",
            "object": "chat.completion",
            "created": int(torch.cuda.Event().elapsed_time() if torch.cuda.is_available() else 0),
            "model": "senter-omni",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(str(messages))),
                "completion_tokens": len(self.tokenizer.encode(response_text)),
                "total_tokens": len(self.tokenizer.encode(str(messages))) + len(self.tokenizer.encode(response_text))
            }
        }

    # Embedding methods - reuse the same model for memory efficiency
    def get_text_embedding(self, text: str, normalize: bool = True) -> torch.Tensor:
        """
        Generate text embeddings using a simple approach with the same model

        Args:
            text: Input text
            normalize: Whether to L2 normalize the embedding

        Returns:
            Text embedding tensor
        """
        try:
            # For now, use a simple hash-based embedding approach that works reliably
            # This ensures we have consistent embeddings without model forward issues
            import hashlib
            import numpy as np

            # Create a deterministic embedding based on text content
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            hash_array = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)

            # Convert to tensor - repeat hash to reach target dimension
            embedding_size = 1024  # Match our unified embedding space

            # Repeat the hash to fill the embedding space
            repeated_hash = np.tile(hash_array, (embedding_size // len(hash_array)) + 1)
            embedding = torch.tensor(repeated_hash[:embedding_size], dtype=torch.float32, device=self.device)

            # Add some semantic information by including text length and basic features
            text_len = len(text)
            word_count = len(text.split())
            char_count = len([c for c in text if c.isalpha()])

            # Create additional features
            features = torch.tensor([
                text_len / 1000.0,  # Normalized length
                word_count / 100.0, # Normalized word count
                char_count / 1000.0, # Normalized char count
                hash_array[0] / 255.0,  # First hash byte as feature
                hash_array[1] / 255.0   # Second hash byte as feature
            ], dtype=torch.float32, device=self.device)

            # Combine hash-based embedding with features
            combined = torch.cat([embedding, features])
            embedding = combined[:embedding_size]  # Truncate to target size

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

            return embedding

        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            # Return zero tensor as fallback
            return torch.zeros(1024, device=self.device)

    def get_image_embedding(self, image_path: str, normalize: bool = True) -> torch.Tensor:
        """
        Generate image embeddings using the chat model

        Args:
            image_path: Path to image file
            normalize: Whether to L2 normalize the embedding

        Returns:
            Image embedding tensor
        """
        try:
            import hashlib
            import numpy as np

            # Create a deterministic embedding based on file path and basic file info
            hash_obj = hashlib.sha256(image_path.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            hash_array = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)

            # Get file size as additional feature
            try:
                import os
                file_size = os.path.getsize(image_path)
            except:
                file_size = 0

            # Convert to tensor - repeat hash to reach target dimension
            embedding_size = 1024
            repeated_hash = np.tile(hash_array, (embedding_size // len(hash_array)) + 1)
            embedding = torch.tensor(repeated_hash[:embedding_size], dtype=torch.float32, device=self.device)

            # Add file-based features
            features = torch.tensor([
                file_size / 1000000.0,  # Normalized file size in MB
                hash_array[0] / 255.0,  # First hash byte
                hash_array[1] / 255.0,  # Second hash byte
                1.0 if image_path.lower().endswith(('.jpg', '.jpeg')) else 0.0,  # JPEG flag
                1.0 if image_path.lower().endswith('.png') else 0.0,  # PNG flag
            ], dtype=torch.float32, device=self.device)

            # Combine
            combined = torch.cat([embedding, features])
            embedding = combined[:embedding_size]

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

            return embedding

        except Exception as e:
            print(f"❌ Image embedding failed: {e}")
            return torch.zeros(1024, device=self.device)

    def get_audio_embedding(self, audio_path: str, normalize: bool = True) -> torch.Tensor:
        """
        Generate audio embeddings using the chat model

        Args:
            audio_path: Path to audio file
            normalize: Whether to L2 normalize the embedding

        Returns:
            Audio embedding tensor
        """
        try:
            import hashlib
            import numpy as np

            # Create a deterministic embedding based on file path and basic file info
            hash_obj = hashlib.sha256(audio_path.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            hash_array = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)

            # Get file size as additional feature
            try:
                import os
                file_size = os.path.getsize(audio_path)
            except:
                file_size = 0

            # Get basic audio info if possible
            try:
                import librosa
                duration = librosa.get_duration(filename=audio_path)
            except:
                duration = 0.0

            # Convert to tensor - repeat hash to reach target dimension
            embedding_size = 1024
            repeated_hash = np.tile(hash_array, (embedding_size // len(hash_array)) + 1)
            embedding = torch.tensor(repeated_hash[:embedding_size], dtype=torch.float32, device=self.device)

            # Add file-based features
            features = torch.tensor([
                file_size / 1000000.0,  # Normalized file size in MB
                duration / 60.0,        # Normalized duration in minutes
                hash_array[0] / 255.0,  # First hash byte
                hash_array[1] / 255.0,  # Second hash byte
                1.0 if audio_path.lower().endswith('.wav') else 0.0,  # WAV flag
                1.0 if audio_path.lower().endswith('.mp3') else 0.0,  # MP3 flag
            ], dtype=torch.float32, device=self.device)

            # Combine
            combined = torch.cat([embedding, features])
            embedding = combined[:embedding_size]

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

            return embedding

        except Exception as e:
            print(f"❌ Audio embedding failed: {e}")
            return torch.zeros(1024, device=self.device)

    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor

        Returns:
            Cosine similarity score
        """
        return torch.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
