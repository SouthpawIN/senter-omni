#!/usr/bin/env python3
"""
Senter-Embed Core Embedding Model

Comprehensive multimodal embedding system for similarity search.
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


class SenterEmbedder:
    """
    Comprehensive multimodal embedding model for similarity search using Gemma3N
    """

    def __init__(self, model_path="models/huggingface/senter-omni-lora", device="auto", use_memory_efficient=True):
        """
        Initialize the Senter-Embed multimodal embedder

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
        print("🤖 Loading Senter-Embed Multimodal Model...")

        try:
            # Load base model with same memory approach as chat model
            base_model = AutoModelForCausalLM.from_pretrained(
                "unsloth/gemma-3n-E4B-it",
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map={"": self.device},  # Same as chat model - no auto device mapping
                trust_remote_code=True
            )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

            # Load tokenizer and processor
            self.tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3n-E4B-it")
            self.processor = AutoProcessor.from_pretrained("unsloth/gemma-3n-E4B-it")

            print("✅ Senter-Embed Multimodal Model loaded successfully!")
            print(f"📍 Device: {self.device}")
            print(f"📝 Text embedding dim: {self.text_embed_dim}")
            print(f"🖼️ Vision embedding dim: {self.vision_embed_dim}")
            print(f"🎵 Audio embedding dim: {self.audio_embed_dim}")
            print(f"🔄 Unified embedding dim: {self.unified_embed_dim}")
            print("🎯 Multimodal embedding generation ready!")

        except Exception as e:
            print(f"❌ Failed to load multimodal model: {e}")
            raise

    def get_text_embedding(self, text: str, normalize: bool = True) -> torch.Tensor:
        """
        Generate text embedding from input text using proper Gemma3N extraction

        Args:
            text: Input text to embed
            normalize: Whether to L2 normalize the embedding

        Returns:
            Text embedding tensor
        """
        # Use proper Gemma3N text embedding extraction
        with torch.no_grad():
            # Use autocast for precision stability
            with torch.amp.autocast('cuda', dtype=torch.float32):
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)

                # Use the last hidden state as embedding (mean pooling)
                hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

                # For text embeddings, we want a single vector, not per token
                # Mean pool across sequence dimension, then ensure we have the right shape
                embedding = hidden_states.mean(dim=1)  # [batch, hidden_size]

                # If we have multiple sequences (shouldn't happen with single text), take first
                if len(embedding.shape) > 1 and embedding.shape[0] > 1:
                    embedding = embedding[0]  # Take first sequence

                # Ensure we have the expected text embedding dimension
                if embedding.shape[-1] != self.text_embed_dim:
                    # Pad or truncate to match expected dimension
                    if embedding.shape[-1] < self.text_embed_dim:
                        # Create padding with same number of dimensions as embedding
                        padding_shape = list(embedding.shape)
                        padding_shape[-1] = self.text_embed_dim - embedding.shape[-1]
                        padding = torch.zeros(*padding_shape, device=self.device)
                        embedding = torch.cat([embedding, padding], dim=-1)
                    else:
                        embedding = embedding[..., :self.text_embed_dim]

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        # Ensure we return a 1D embedding vector
        if len(embedding.shape) > 1:
            if embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)  # Remove batch dimension
            else:
                # If we have multiple embeddings, take the first one
                embedding = embedding[0] if embedding.shape[0] > 1 else embedding.squeeze()

        return embedding

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
        inputs = self.processor(
            images=pil_image,
            text="",  # Empty text input for image-only processing
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # Use the vision tower from the base model (bypass LoRA for vision)
            base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model

            # Filter inputs to only pass image-related tensors to vision tower
            vision_inputs = {k: v for k, v in inputs.items() if k in ['pixel_values']}

            # Use autocast to prevent NaN issues with Conv2D layers (Gemma3n specific fix)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                # Get image features directly using the model's method
                if hasattr(base_model, 'get_image_features'):
                    # This method handles the vision tower internally
                    embedding = base_model.get_image_features(**vision_inputs)

                    # get_image_features returns [batch, patches, features]
                    # We need to pool across patches to get [batch, features]
                    if len(embedding.shape) == 3:
                        embedding = embedding.mean(dim=1)  # Average across patches
                else:
                    # Fallback: use vision tower and mean pool
                    vision_outputs = base_model.vision_tower(**vision_inputs)

                    # Handle different output formats
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        # 4D tensor [batch, channels, height, width] -> flatten and mean pool
                        hidden_states = vision_outputs.last_hidden_state
                        embedding = hidden_states.mean(dim=[2, 3])  # Average over spatial dimensions
                    else:
                        # Fallback for other formats
                        embedding = vision_outputs.mean(dim=[2, 3]) if len(vision_outputs.shape) == 4 else vision_outputs

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        # Ensure we return a 1D embedding vector
        if len(embedding.shape) > 1:
            if embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)  # Remove batch dimension
            else:
                # If we have multiple embeddings, take the first one
                embedding = embedding[0] if embedding.shape[0] > 1 else embedding.squeeze()

        return embedding

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

        # Ensure proper shape (add batch dimension if needed)
        if len(audio_array.shape) == 1:
            audio_array = audio_array[np.newaxis, :]  # Add batch dimension

        # Process audio with audio encoder
        # Gemma3N processor requires both text and audio inputs
        inputs = self.processor(
            audio=audio_array,
            sampling_rate=sr,
            text="",  # Empty text input for audio-only processing
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # Use the audio tower from the base model (bypass LoRA for audio)
            base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model

            # Map processor outputs to audio tower expected inputs
            audio_inputs = {
                'audio_mel': inputs['input_features'],
                'audio_mel_mask': inputs['input_features_mask']
            }

            # Use autocast to prevent precision issues (Gemma3n specific fix)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                audio_outputs = base_model.audio_tower(**audio_inputs)

                # Audio tower returns tuple (hidden_states, mask)
                hidden_states = audio_outputs[0]  # Extract hidden states from tuple

                # Use mean pooling for audio embeddings
                embedding = hidden_states.mean(dim=1)

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
            Video embedding tensor
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

    def project_to_unified_space(self, embedding: torch.Tensor, target_dim: int = None) -> torch.Tensor:
        """
        Project embedding to unified dimension space

        Args:
            embedding: Input embedding
            target_dim: Target dimension (default: self.unified_embed_dim)

        Returns:
            Projected embedding
        """
        if target_dim is None:
            target_dim = self.unified_embed_dim

        current_dim = embedding.shape[-1]

        if current_dim == target_dim:
            return embedding

        # Simple projection using linear layer (could be improved with proper training)
        # For now, we'll use a simple downsampling/upsampling approach
        if current_dim > target_dim:
            # Down-project by averaging chunks
            kernel_size = current_dim // target_dim
            if kernel_size > 1:
                # Reshape and average
                embedding_reshaped = embedding.view(-1, target_dim, kernel_size)
                projected = embedding_reshaped.mean(dim=-1)
            else:
                # Simple truncation
                projected = embedding[..., :target_dim]
        else:
            # Up-project by interpolation
            projected = embedding
            while projected.shape[-1] < target_dim:
                # Duplicate values
                projected = torch.cat([projected, projected], dim=-1)
            projected = projected[..., :target_dim]

        return projected

    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings (with unified projection)

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        # Project both embeddings to unified dimension space
        emb1_proj = self.project_to_unified_space(embedding1)
        emb2_proj = self.project_to_unified_space(embedding2)

        # Ensure tensors are 1D for cosine similarity
        if len(emb1_proj.shape) > 1:
            emb1_proj = emb1_proj.squeeze()
        if len(emb2_proj.shape) > 1:
            emb2_proj = emb2_proj.squeeze()

        return F.cosine_similarity(emb1_proj.unsqueeze(0), emb2_proj.unsqueeze(0)).item()

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
