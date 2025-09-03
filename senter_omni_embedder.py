#!/usr/bin/env python3
"""
Senter-Omni Embedder - Multimodal Embedding Model for Similarity Search

This module provides embedding generation and similarity search capabilities
for text, images, audio, and video using the Gemma3N architecture.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import warnings
from PIL import Image
import io

# Optional imports for video/audio processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

class SenterOmniEmbedder:
    """
    Multimodal embedding model for similarity search using Gemma3N
    """

    def __init__(self, model_path="models/huggingface/senter-omni-lora", device="auto", use_memory_efficient=True):
        """
        Initialize the Senter-Omni embedder

        Args:
            model_path: Path to the model (LoRA or merged)
            device: Device to run on ('auto', 'cuda', 'cpu')
            use_memory_efficient: Use memory-efficient loading and inference
        """
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.use_memory_efficient = use_memory_efficient

        # Embedding dimensions for different modalities
        self.text_embed_dim = 4096  # Gemma3N hidden size
        self.vision_embed_dim = 2048  # Vision encoder output (from config)
        self.audio_embed_dim = 1536  # Audio encoder output (from config)

        # Unified embedding dimension (projected)
        self.unified_embed_dim = 1024

        self.load_model()

    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device"""
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """Load the Gemma3N model and processors"""
        print("🤖 Loading Senter-Omni Embedder...")

        try:
            # Load base model with memory optimization
            if self.use_memory_efficient and self.device.startswith("cuda"):
                # Use 4-bit quantization for memory efficiency
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    base_model = AutoModelForCausalLM.from_pretrained(
                        "unsloth/gemma-3n-E4B-it",
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except ImportError:
                    print("⚠️ bitsandbytes not available, using standard loading")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        "unsloth/gemma-3n-E4B-it",
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                        device_map={"": self.device},
                        trust_remote_code=True
                    )
            else:
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
            self.processor = AutoProcessor.from_pretrained("unsloth/gemma-3n-E4B-it")

            print("✅ Senter-Omni Embedder loaded successfully!")
            print(f"📍 Device: {self.device}")
            print(f"📝 Text embedding dim: {self.text_embed_dim}")
            print(f"🖼️ Vision embedding dim: {self.vision_embed_dim}")
            print(f"🎵 Audio embedding dim: {self.audio_embed_dim}")
            print(f"🔄 Unified embedding dim: {self.unified_embed_dim}")
            print("🎯 Multimodal embedding generation ready!")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def get_text_embedding(self, text: str, normalize: bool = True) -> torch.Tensor:
        """
        Generate text embedding from input text

        Args:
            text: Input text to embed
            normalize: Whether to L2 normalize the embedding

        Returns:
            Text embedding tensor
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs, output_hidden_states=True)

            # Use the last hidden state as embedding (mean pooling)
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

            # Mean pooling across sequence dimension
            embedding = hidden_states.mean(dim=1)  # [batch, hidden_size]

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0)  # Remove batch dimension

    def get_image_embedding(self, image: Union[str, Image.Image, np.ndarray], normalize: bool = True) -> torch.Tensor:
        """
        Generate image embedding using Gemma3N vision tower

        Args:
            image: Image path, PIL Image, or numpy array
            normalize: Whether to L2 normalize the embedding

        Returns:
            Image embedding tensor
        """
        # Load and preprocess image
        if isinstance(image, str):
            # Load from file path
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL
            pil_image = Image.fromarray(image).convert('RGB')
        else:
            pil_image = image

        # Process image with vision encoder
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Use the vision tower from the base model (bypass LoRA for vision)
            base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
            vision_outputs = base_model.vision_tower(**inputs)

            # Get image features using the model's method
            if hasattr(base_model, 'get_image_features'):
                embedding = base_model.get_image_features(vision_outputs)
            else:
                # Fallback to mean pooling
                embedding = vision_outputs.last_hidden_state.mean(dim=1)

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0)

    def get_audio_embedding(self, audio: Union[str, np.ndarray], sr: int = 16000, normalize: bool = True) -> torch.Tensor:
        """
        Generate audio embedding from speech/audio using Gemma3N audio tower

        Args:
            audio: Audio file path or numpy array
            sr: Sample rate (if numpy array provided)
            normalize: Whether to L2 normalize the embedding

        Returns:
            Audio embedding tensor
        """
        # Load audio
        if isinstance(audio, str):
            # Load from file
            if not LIBROSA_AVAILABLE:
                raise ImportError("librosa is required for audio file processing. Install with: pip install librosa")
            audio_array, sr = librosa.load(audio, sr=sr)
        else:
            audio_array = audio

        # Process audio with audio encoder
        inputs = self.processor(audio=audio_array, sampling_rate=sr, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Use the audio tower from the base model (bypass LoRA for audio)
            base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
            audio_outputs = base_model.audio_tower(**inputs)

            # Use mean pooling for audio embeddings
            embedding = audio_outputs.last_hidden_state.mean(dim=1)

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0)

    def get_video_embedding(self, video_path: str, frame_rate: int = 1, max_frames: int = 10, normalize: bool = True) -> torch.Tensor:
        """
        Generate video embedding by processing individual frames

        Args:
            video_path: Path to video file
            frame_rate: Sample every Nth frame
            max_frames: Maximum number of frames to process
            normalize: Whether to L2 normalize the embedding

        Returns:
            Video embedding tensor (averaged from frame embeddings)
        """
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python is required for video processing. Install with: pip install opencv-python")

        cap = cv2.VideoCapture(video_path)
        frame_embeddings = []

        frame_count = 0
        processed_frames = 0

        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Sample frames at specified rate
            if frame_count % frame_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get frame embedding
                frame_embedding = self.get_image_embedding(frame_rgb, normalize=False)
                frame_embeddings.append(frame_embedding)
                processed_frames += 1

        cap.release()

        if not frame_embeddings:
            raise ValueError("No frames could be extracted from video")

        # Average all frame embeddings
        video_embedding = torch.stack(frame_embeddings).mean(dim=0)

        if normalize:
            video_embedding = F.normalize(video_embedding, p=2, dim=-1)

        return video_embedding

    def project_to_unified_space(self, embedding: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Project modality-specific embedding to unified embedding space

        Args:
            embedding: Input embedding
            modality: Modality type ('text', 'image', 'audio', 'video')

        Returns:
            Projected embedding in unified space
        """
        # For now, we'll use simple projection. In a production system,
        # you'd train dedicated projection layers for each modality

        if modality == 'text':
            # Text embeddings are already in the right space (Gemma3N hidden size)
            # Project to unified dimension
            projection_matrix = torch.randn(self.text_embed_dim, self.unified_embed_dim, device=self.device)
            projection_matrix = F.normalize(projection_matrix, p=2, dim=0)
            return torch.matmul(embedding, projection_matrix)

        elif modality in ['image', 'video']:
            # Vision embeddings to unified space
            projection_matrix = torch.randn(self.vision_embed_dim, self.unified_embed_dim, device=self.device)
            projection_matrix = F.normalize(projection_matrix, p=2, dim=0)
            return torch.matmul(embedding, projection_matrix)

        elif modality == 'audio':
            # Audio embeddings to unified space
            projection_matrix = torch.randn(self.audio_embed_dim, self.unified_embed_dim, device=self.device)
            projection_matrix = F.normalize(projection_matrix, p=2, dim=0)
            return torch.matmul(embedding, projection_matrix)

        else:
            raise ValueError(f"Unknown modality: {modality}")

    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        return F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

    def embed_multimodal_content(self, content: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for multimodal content

        Args:
            content: Dictionary with content for different modalities
                     {'text': str, 'image': str/Image, 'audio': str/np.ndarray, 'video': str}

        Returns:
            Dictionary of embeddings for each modality
        """
        embeddings = {}

        if 'text' in content:
            embeddings['text'] = self.get_text_embedding(content['text'])

        if 'image' in content:
            embeddings['image'] = self.get_image_embedding(content['image'])

        if 'audio' in content:
            embeddings['audio'] = self.get_audio_embedding(content['audio'])

        if 'video' in content:
            embeddings['video'] = self.get_video_embedding(content['video'])

        return embeddings

    def find_similar(self, query_embedding: torch.Tensor, embedding_database: List[torch.Tensor],
                    top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings in database

        Args:
            query_embedding: Query embedding
            embedding_database: List of embeddings to search
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []

        for i, db_embedding in enumerate(embedding_database):
            similarity = self.compute_similarity(query_embedding, db_embedding)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class MultimodalEmbeddingDatabase:
    """
    Database for storing and retrieving multimodal embeddings
    """

    def __init__(self, embedder: SenterOmniEmbedder):
        self.embedder = embedder
        self.embeddings = []
        self.metadata = []
        self.modalities = []

    def add_content(self, content: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        Add multimodal content to database

        Args:
            content: Multimodal content dictionary
            metadata: Optional metadata for the content
        """
        embeddings = self.embedder.embed_multimodal_content(content)

        for modality, embedding in embeddings.items():
            self.embeddings.append(embedding)
            self.modalities.append(modality)
            self.metadata.append({
                'modality': modality,
                'content': content.get(modality),
                'metadata': metadata or {}
            })

    def search_similar(self, query_content: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar content in database

        Args:
            query_content: Query content dictionary
            top_k: Number of results to return

        Returns:
            List of similar content with scores
        """
        # Generate query embeddings
        query_embeddings = self.embedder.embed_multimodal_content(query_content)

        results = []

        # Search for each query modality
        for query_modality, query_embedding in query_embeddings.items():
            # Filter database to same modality
            same_modality_indices = [i for i, mod in enumerate(self.modalities) if mod == query_modality]

            if same_modality_indices:
                same_modality_embeddings = [self.embeddings[i] for i in same_modality_indices]

                # Find similar embeddings
                similar_results = self.embedder.find_similar(query_embedding, same_modality_embeddings, top_k)

                # Convert to result format
                for rank, (db_idx, score) in enumerate(similar_results):
                    original_idx = same_modality_indices[db_idx]
                    results.append({
                        'rank': rank + 1,
                        'similarity': score,
                        'modality': query_modality,
                        'content': self.metadata[original_idx]['content'],
                        'metadata': self.metadata[original_idx]['metadata']
                    })

        # Sort by similarity score
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]


def demo_multimodal_embeddings():
    """Demonstrate multimodal embedding capabilities"""
    print("🎭 Senter-Omni Multimodal Embedder Demo")
    print("=" * 60)

    try:
        # Initialize embedder
        embedder = SenterOmniEmbedder()

        # Create database
        db = MultimodalEmbeddingDatabase(embedder)

        print("\n📝 Adding sample content to database...")

        # Add text content
        db.add_content({'text': 'The quick brown fox jumps over the lazy dog'}, {'type': 'sample', 'id': 1})
        db.add_content({'text': 'Machine learning is transforming artificial intelligence'}, {'type': 'sample', 'id': 2})

        # Add image content (if available)
        try:
            if Path('test_assets/test_image.jpg').exists():
                db.add_content({'image': 'test_assets/test_image.jpg'}, {'type': 'sample', 'id': 3})
                print("✅ Added image content")
        except Exception as e:
            print(f"⚠️ Image processing skipped: {e}")

        # Add audio content (if available)
        try:
            if Path('test_assets/test_audio.wav').exists():
                db.add_content({'audio': 'test_assets/test_audio.wav'}, {'type': 'sample', 'id': 4})
                print("✅ Added audio content")
        except Exception as e:
            print(f"⚠️ Audio processing skipped: {e}")

        print("\n🔍 Searching for similar content...")

        # Search examples
        text_query = {'text': 'A fast fox leaps over a sleeping canine'}
        results = db.search_similar(text_query, top_k=3)

        print("📊 Text similarity results:")
        for result in results:
            print(f"  Rank {result['rank']}: Similarity {result['similarity']:.3f}")
            print(f"    Content: {result['content'][:50]}...")
            print()

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Note: Make sure the model is properly loaded and test files exist")


if __name__ == "__main__":
    demo_multimodal_embeddings()
