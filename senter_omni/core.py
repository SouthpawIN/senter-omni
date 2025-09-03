#!/usr/bin/env python3
"""
Senter-Omni Core Chat Model

Advanced multimodal conversational AI with XML tag support.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import re
import sys
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


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

    def __init__(self, model_path="models/huggingface/senter-omni-lora", device="auto"):
        """
        Initialize the Senter-Omni chat model

        Args:
            model_path: Path to the model (LoRA or merged)
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device(device)
        self.model_path = model_path

        # Define stop tokens for Gemma3N
        self.stop_tokens = [
            "<start_of_turn>",
            "<end_of_turn>",
            "<start_of_turn>user",
            "<start_of_turn>system",
            "</s>"
        ]

        self.load_model()

    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device"""
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """Load the Gemma3N model"""
        print("🤖 Loading Senter-Omni Chat Model...")

        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "unsloth/gemma-3n-E4B-it",
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map={"": self.device},
                trust_remote_code=True
            )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

            # Load tokenizer and processor
            self.tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3n-E4B-it")

            print("✅ Senter-Omni Chat Model loaded successfully!")
            print(f"📍 Device: {self.device}")

        except Exception as e:
            print(f"❌ Failed to load chat model: {e}")
            raise

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

        Converts XML to proper Gemma3N chat format
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

                    # For Gemma3N, images use <start_of_image> token
                    if tag == "image":
                        replacement = f'<start_of_image>{match.strip()}'
                    else:
                        # For audio/video, keep descriptive format
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
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "stream": True
            }

        # Process messages for multimodal content and XML parsing
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

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            processed_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True
        ).to(self.model.device)

        # Create stopping criteria for Gemma3N format
        stopping_criteria = StoppingCriteriaList([
            StopTokenCriteria(self.tokenizer, self.stop_tokens)
        ])

        # Create streamer for streaming output
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        # Generate with streaming and stop tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_params.get("max_new_tokens", 256),
                temperature=generation_params.get("temperature", 0.8),
                top_p=generation_params.get("top_p", 0.9),
                top_k=generation_params.get("top_k", 50),
                repetition_penalty=generation_params.get("repetition_penalty", 1.1),
                do_sample=generation_params.get("do_sample", True),
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )

        # Return the generated text (cleaned of stop tokens)
        generated_text = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

        # Clean up the generated text by removing any stop tokens that might have been included
        for stop_token in self.stop_tokens:
            if stop_token in generated_text:
                generated_text = generated_text.split(stop_token)[0]

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
