#!/usr/bin/env python3
"""
Senter-Omni Multimodal Embedder

Lightweight embedding functionality for the chat model.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Optional
import warnings
from PIL import Image
import io


class MultimodalEmbedder:
    """
    Lightweight multimodal embedder for Senter-Omni chat
    """

    def __init__(self, chat_model):
        """
        Initialize embedder using the chat model's components

        Args:
            chat_model: SenterOmniChat instance
        """
        self.chat_model = chat_model
        self.device = chat_model.device

    def get_text_embedding(self, text: str, normalize: bool = True) -> torch.Tensor:
        """
        Generate text embedding using chat model

        Args:
            text: Input text
            normalize: Whether to L2 normalize

        Returns:
            Text embedding tensor
        """
        # Use chat model's tokenizer and base model for embeddings
        inputs = self.chat_model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            # Get model outputs (use base model to avoid LoRA for embeddings)
            base_model = self.chat_model.model.base_model if hasattr(self.chat_model.model, 'base_model') else self.chat_model.model
            outputs = base_model(**inputs, output_hidden_states=True)

            # Use the last hidden state as embedding (mean pooling)
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            embedding = hidden_states.mean(dim=1)  # [batch, hidden_size]

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0)

    def get_image_embedding(self, image: Union[str, Image.Image], normalize: bool = True) -> torch.Tensor:
        """
        Generate image embedding using chat model's vision components

        Args:
            image: Image path or PIL Image
            normalize: Whether to L2 normalize

        Returns:
            Image embedding tensor
        """
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            pil_image = image

        # Use chat model's processor
        inputs = self.chat_model.tokenizer.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Use base model vision components
            base_model = self.chat_model.model.base_model if hasattr(self.chat_model.model, 'base_model') else self.chat_model.model
            vision_outputs = base_model.vision_tower(**inputs)

            # Get image features
            if hasattr(base_model, 'get_image_features'):
                embedding = base_model.get_image_features(vision_outputs)
            else:
                embedding = vision_outputs.last_hidden_state.mean(dim=1)

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0)

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Compute cosine similarity between embeddings

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity score
        """
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
"""
Senter-Omni Multimodal Embedder

Lightweight embedding functionality for the chat model.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Optional
import warnings
from PIL import Image
import io


class MultimodalEmbedder:
    """
    Lightweight multimodal embedder for Senter-Omni chat
    """

    def __init__(self, chat_model):
        """
        Initialize embedder using the chat model's components

        Args:
            chat_model: SenterOmniChat instance
        """
        self.chat_model = chat_model
        self.device = chat_model.device

    def get_text_embedding(self, text: str, normalize: bool = True) -> torch.Tensor:
        """
        Generate text embedding using chat model

        Args:
            text: Input text
            normalize: Whether to L2 normalize

        Returns:
            Text embedding tensor
        """
        # Use chat model's tokenizer and base model for embeddings
        inputs = self.chat_model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            # Get model outputs (use base model to avoid LoRA for embeddings)
            base_model = self.chat_model.model.base_model if hasattr(self.chat_model.model, 'base_model') else self.chat_model.model
            outputs = base_model(**inputs, output_hidden_states=True)

            # Use the last hidden state as embedding (mean pooling)
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            embedding = hidden_states.mean(dim=1)  # [batch, hidden_size]

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0)

    def get_image_embedding(self, image: Union[str, Image.Image], normalize: bool = True) -> torch.Tensor:
        """
        Generate image embedding using chat model's vision components

        Args:
            image: Image path or PIL Image
            normalize: Whether to L2 normalize

        Returns:
            Image embedding tensor
        """
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            pil_image = image

        # Use chat model's processor
        inputs = self.chat_model.tokenizer.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Use base model vision components
            base_model = self.chat_model.model.base_model if hasattr(self.chat_model.model, 'base_model') else self.chat_model.model
            vision_outputs = base_model.vision_tower(**inputs)

            # Get image features
            if hasattr(base_model, 'get_image_features'):
                embedding = base_model.get_image_features(vision_outputs)
            else:
                embedding = vision_outputs.last_hidden_state.mean(dim=1)

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0)

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Compute cosine similarity between embeddings

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity score
        """
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
"""
Senter-Omni Multimodal Embedder

Lightweight embedding functionality for the chat model.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Optional
import warnings
from PIL import Image
import io


class MultimodalEmbedder:
    """
    Lightweight multimodal embedder for Senter-Omni chat
    """

    def __init__(self, chat_model):
        """
        Initialize embedder using the chat model's components

        Args:
            chat_model: SenterOmniChat instance
        """
        self.chat_model = chat_model
        self.device = chat_model.device

    def get_text_embedding(self, text: str, normalize: bool = True) -> torch.Tensor:
        """
        Generate text embedding using chat model

        Args:
            text: Input text
            normalize: Whether to L2 normalize

        Returns:
            Text embedding tensor
        """
        # Use chat model's tokenizer and base model for embeddings
        inputs = self.chat_model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            # Get model outputs (use base model to avoid LoRA for embeddings)
            base_model = self.chat_model.model.base_model if hasattr(self.chat_model.model, 'base_model') else self.chat_model.model
            outputs = base_model(**inputs, output_hidden_states=True)

            # Use the last hidden state as embedding (mean pooling)
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            embedding = hidden_states.mean(dim=1)  # [batch, hidden_size]

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0)

    def get_image_embedding(self, image: Union[str, Image.Image], normalize: bool = True) -> torch.Tensor:
        """
        Generate image embedding using chat model's vision components

        Args:
            image: Image path or PIL Image
            normalize: Whether to L2 normalize

        Returns:
            Image embedding tensor
        """
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            pil_image = image

        # Use chat model's processor
        inputs = self.chat_model.tokenizer.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Use base model vision components
            base_model = self.chat_model.model.base_model if hasattr(self.chat_model.model, 'base_model') else self.chat_model.model
            vision_outputs = base_model.vision_tower(**inputs)

            # Get image features
            if hasattr(base_model, 'get_image_features'):
                embedding = base_model.get_image_features(vision_outputs)
            else:
                embedding = vision_outputs.last_hidden_state.mean(dim=1)

            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0)

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Compute cosine similarity between embeddings

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity score
        """
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
