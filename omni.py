#!/usr/bin/env python3
"""
🎭 Senter-Omni Unified API

Unified interface for chat and embedding operations across modalities.
"""

import torch
import re
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
    print("=" * 50)
    print("Available functions:")
    print("• omni.chat() - Chat completions with parameters")
    print("• omni.embed() - Multimodal embeddings with XML tags")
    print("• omni.create_chat_completion() - OpenAI-style API")
    print("• omni.generate() - llama.cpp-style generation")

    print("\\n📖 Usage Examples:")
    print("```python")
    print("# Chat")
    print("response = omni.chat('<user>Hello!</user>', max_tokens=100, temperature=0.7)")
    print("")
    print("# Embeddings")
    print("result = omni.embed('<text>Hello</text><image>photo.jpg</image>')")
    print("```")
