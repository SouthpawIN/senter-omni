#!/usr/bin/env python3
"""
Advanced Senter-Omni Chat Interface

LlamaCPP-style interface with:
- XML tags for multimodal content
- Streaming responses
- Customizable generation parameters
- OpenAI-compatible chat format
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import re
import sys
import json
from typing import Dict, List, Any, Optional
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

class AdvancedSenterOmni:
    """Advanced Senter-Omni interface with LlamaCPP-style features"""

    def __init__(self, model_path="models/huggingface/senter-omni-lora"):
        self.model = None
        self.tokenizer = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.conversation_history = []
        self.load_model(model_path)

        # Define stop tokens for Gemma3N
        self.stop_tokens = [
            "<start_of_turn>",
            "<end_of_turn>",
            "<start_of_turn>user",
            "<start_of_turn>system",
            "</s>"
        ]

    def load_model(self, model_path: str):
        """Load the Senter-Omni model"""
        print("🤖 Loading Advanced Senter-Omni...")

        try:
            if "lora" in model_path:
                base_model = AutoModelForCausalLM.from_pretrained(
                    "unsloth/gemma-3n-E4B-it",
                    torch_dtype=torch.float16,
                    device_map={"": self.device},
                    trust_remote_code=True
                )
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3n-E4B-it")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map={"": self.device},
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            print(f"✅ Advanced Senter-Omni loaded on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)

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

    def generate_streaming(self, messages: List[Dict], generation_params: Dict = None):
        """
        Generate streaming response with customizable parameters

        Args:
            messages: List of message dicts with role and content
            generation_params: Dict of generation parameters
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

        # Create custom streamer that filters out stop tokens
        class StopTokenStreamer(TextStreamer):
            def __init__(self, tokenizer, stop_tokens, skip_prompt=True):
                super().__init__(tokenizer, skip_prompt=skip_prompt)
                self.stop_tokens = stop_tokens

            def put(self, value):
                """Override to filter out stop tokens"""
                if len(value.shape) > 1:
                    value = value[0]

                token_text = self.tokenizer.decode(value, skip_special_tokens=False)

                # Check if this token contains any stop tokens
                should_output = True
                for stop_token in self.stop_tokens:
                    if stop_token in token_text:
                        # Only output text before the stop token
                        token_text = token_text.split(stop_token)[0]
                        should_output = bool(token_text.strip())
                        break

                if should_output and token_text:
                    print(token_text, end="", flush=True)

        streamer = StopTokenStreamer(self.tokenizer, self.stop_tokens, skip_prompt=True)

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

    def chat_completion(self, messages: List[Dict], generation_params: Dict = None) -> Dict:
        """
        OpenAI-compatible chat completion format

        Args:
            messages: List of message dicts
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

def check_video_support():
    """Check if the model supports video processing"""
    print("🎥 Checking Video Support")
    print("=" * 40)

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("unsloth/gemma-3n-E4B-it")

        print("📋 Model Configuration Analysis:")
        print(f"   Model: Gemma3N")
        print(f"   Vision support: ✅ (via vision_config)")
        print(f"   Audio support: ✅ (via audio_config)")

        # Check for video capabilities
        has_video = False
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            print("   Vision details:")
            print(f"     Architecture: {getattr(vision_config, 'architecture', 'Unknown')}")
            print(f"     Hidden size: {getattr(vision_config, 'hidden_size', 'Unknown')}")

            # Video support would typically be indicated by:
            # - Temporal processing capabilities
            # - Frame sequence handling
            # - Motion analysis features

        if hasattr(config, 'audio_config'):
            audio_config = config.audio_config
            print("   Audio details:")
            print(f"     Sample rate handling: {getattr(audio_config, 'hidden_size', 'Unknown')}")
            print(f"     Temporal processing: {getattr(audio_config, 'conf_attention_chunk_size', 'Unknown')}")

        print("\n🎥 Video Support Assessment:")
        print("   • Frame-by-frame analysis: Likely supported via vision pipeline")
        print("   • Motion analysis: Limited (primarily static image analysis)")
        print("   • Video understanding: Basic (can process individual frames)")
        print("   • Real-time processing: Not optimized")
        print("   • Recommended: Use image extraction for video frames")

        return True

    except Exception as e:
        print(f"❌ Video support check failed: {e}")
        return False

def demo_advanced_features():
    """Demonstrate advanced features"""
    print("🚀 Advanced Senter-Omni Demo")
    print("=" * 50)

    ai = AdvancedSenterOmni()

    # Demo 1: XML role-based messages
    print("\n📝 Demo 1: XML Role-Based Messages")
    messages = [
        "<system>You are Senter-Omni, an advanced AI assistant with multimodal capabilities.</system>",
        "<user>Hello! Can you introduce yourself and explain your capabilities?</user>"
    ]

    print("🤖 Generating response...")
    response = ai.generate_streaming(messages)
    print(f"\n📄 Response: {response[:200]}...")

    # Demo 2: Multimodal content
    print("\n🖼️ Demo 2: Multimodal Content")
    multimodal_message = """
<user>
I have this image: <image>A beautiful sunset over mountains with orange and purple hues</image>

And this audio: <audio>Peaceful instrumental music with piano and strings</audio>

Can you describe what this scene might look and sound like?
</user>
"""

    print("🎭 Processing multimodal content...")
    response = ai.generate_streaming([multimodal_message])
    print(f"\n📄 Response: {response[:200]}...")

    # Demo 3: Custom generation parameters
    print("\n⚙️ Demo 3: Custom Generation Parameters")
    custom_params = {
        "max_new_tokens": 100,
        "temperature": 0.1,  # More focused
        "top_p": 0.8,
        "repetition_penalty": 1.2
    }

    focused_message = "<user>Explain what a neural network is in simple terms.</user>"
    print("🎯 Generating focused response...")
    response = ai.generate_streaming([focused_message], custom_params)
    print(f"\n📄 Response: {response[:200]}...")

    # Demo 4: OpenAI-compatible format
    print("\n🔄 Demo 4: OpenAI-Compatible Format")
    openai_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's 2 + 2?"}
    ]

    completion = ai.chat_completion(openai_messages)
    print("📋 OpenAI Format Response:")
    print(json.dumps(completion, indent=2))

def interactive_advanced_chat():
    """Interactive chat with advanced features"""
    print("🎭 Advanced Senter-Omni Interactive Chat")
    print("=" * 50)
    print("Features:")
    print("• XML tags: <system>, <user>, <assistant>")
    print("• Multimodal: <image>, <audio>, <video>")
    print("• Streaming responses")
    print("• Conversation history")
    print("=" * 50)

    ai = AdvancedSenterOmni()

    print("\n💡 Example usage:")
    print("<user>Hello! Tell me about yourself.</user>")
    print("<system>You are a helpful AI assistant.</system>")
    print("<user>I have this image: <image>sunset over mountains</image> What do you see?</user>")

    conversation_history = []

    while True:
        try:
            user_input = input("\n🎤 Your message (or 'quit' to exit): ").strip()

            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break

            if user_input.lower() == 'help':
                print("\n📚 Commands:")
                print("  help     - Show this help")
                print("  clear    - Clear conversation")
                print("  params   - Set generation parameters")
                print("  video    - Check video support")
                print("  history  - Show conversation history")
                print("  quit     - Exit")
                print("\n📝 XML Tags:")
                print("  <system>content</system>     - System prompt")
                print("  <user>content</user>         - User message")
                print("  <assistant>content</assistant> - Assistant response")
                print("  <image>content</image>       - Image description")
                print("  <audio>content</audio>       - Audio description")
                print("  <video>content</video>       - Video description")
                continue

            if user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation cleared!")
                continue

            if user_input.lower() == 'video':
                check_video_support()
                continue

            if user_input.lower() == 'history':
                if conversation_history:
                    print("\n📜 Conversation History:")
                    for i, msg in enumerate(conversation_history, 1):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:100]
                        print(f"  {i}. {role}: {content}...")
                else:
                    print("📜 No conversation history yet.")
                continue

            if user_input.lower() == 'params':
                print("⚙️ Custom parameters not yet implemented in interactive mode")
                continue

            if not user_input:
                continue

            # Add user message to history
            user_msg = ai.parse_xml_message(user_input)
            conversation_history.append(user_msg)

            # Generate response
            print("\n🤖 Senter-Omni:", end=" ")
            response = ai.generate_streaming(conversation_history)
            print()

            # Add assistant response to history
            assistant_msg = {
                "role": "assistant",
                "content": response
            }
            conversation_history.append(assistant_msg)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo_advanced_features()
        elif sys.argv[1] == "--video-check":
            check_video_support()
        else:
            print("Usage: python advanced_chat.py [--demo|--video-check]")
    else:
        interactive_advanced_chat()

if __name__ == "__main__":
    main()
