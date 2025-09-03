#!/usr/bin/env python3
"""
Senter-Embed Utility Functions

Utility functions for embedding operations and similarity computations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from pathlib import Path


def cosine_similarity(emb1: Union[torch.Tensor, np.ndarray],
                     emb2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute cosine similarity between two embeddings

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Cosine similarity score
    """
    # Convert to tensors if needed
    if isinstance(emb1, np.ndarray):
        emb1 = torch.from_numpy(emb1)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.from_numpy(emb2)

    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


def compute_similarity_matrix(embeddings: List[Union[torch.Tensor, np.ndarray]]) -> torch.Tensor:
    """
    Compute pairwise similarity matrix for a list of embeddings

    Args:
        embeddings: List of embeddings

    Returns:
        Similarity matrix tensor
    """
    # Convert to tensors
    tensor_embeddings = []
    for emb in embeddings:
        if isinstance(emb, np.ndarray):
            tensor_embeddings.append(torch.from_numpy(emb))
        else:
            tensor_embeddings.append(emb)

    # Stack into matrix
    embedding_matrix = torch.stack(tensor_embeddings)

    # Normalize embeddings
    embedding_matrix = F.normalize(embedding_matrix, p=2, dim=-1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(embedding_matrix, embedding_matrix.t())

    return similarity_matrix


def find_most_similar(query_embedding: Union[torch.Tensor, np.ndarray],
                     embedding_database: List[Union[torch.Tensor, np.ndarray]],
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
        similarity = cosine_similarity(query_embedding, db_embedding)
        similarities.append((i, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def normalize_embeddings(embeddings: List[Union[torch.Tensor, np.ndarray]]) -> List[torch.Tensor]:
    """
    Normalize a list of embeddings using L2 normalization

    Args:
        embeddings: List of embeddings to normalize

    Returns:
        List of normalized embeddings
    """
    normalized = []

    for emb in embeddings:
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)

        # L2 normalize
        normalized_emb = F.normalize(emb, p=2, dim=-1)
        normalized.append(normalized_emb)

    return normalized


def batch_embed_texts(embedder, texts: List[str], batch_size: int = 8) -> List[torch.Tensor]:
    """
    Generate embeddings for a batch of texts

    Args:
        embedder: Embedding model instance
        texts: List of texts to embed
        batch_size: Batch size for processing

    Returns:
        List of embeddings
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        for text in batch_texts:
            embedding = embedder.get_text_embedding(text)
            embeddings.append(embedding)

    return embeddings


def load_image_batch(image_paths: List[Union[str, Path]]) -> List[Any]:
    """
    Load a batch of images

    Args:
        image_paths: List of image paths

    Returns:
        List of loaded images
    """
    from PIL import Image

    images = []
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            images.append(image)
        except Exception as e:
            print(f"Warning: Failed to load image {path}: {e}")
            continue

    return images


def create_embedding_index(embeddings: List[torch.Tensor], metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create an index structure for efficient similarity search

    Args:
        embeddings: List of embeddings
        metadata: Optional metadata for each embedding

    Returns:
        Index dictionary
    """
    index = {
        'embeddings': embeddings,
        'metadata': metadata or [{} for _ in embeddings],
        'dimensions': embeddings[0].shape[-1] if embeddings else 0,
        'count': len(embeddings)
    }

    return index


def search_index(query_embedding: torch.Tensor, index: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search an embedding index for similar items

    Args:
        query_embedding: Query embedding
        index: Embedding index created by create_embedding_index
        top_k: Number of results to return

    Returns:
        List of similar items with metadata
    """
    similarities = find_most_similar(query_embedding, index['embeddings'], top_k)

    results = []
    for idx, score in similarities:
        result = {
            'index': idx,
            'similarity': score,
            'metadata': index['metadata'][idx]
        }
        results.append(result)

    return results


def calculate_embedding_stats(embeddings: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of embeddings

    Args:
        embeddings: List of embeddings

    Returns:
        Dictionary with embedding statistics
    """
    if not embeddings:
        return {'count': 0}

    # Stack embeddings for analysis
    embedding_matrix = torch.stack(embeddings)

    stats = {
        'count': len(embeddings),
        'dimensions': embedding_matrix.shape[-1],
        'mean_norm': torch.norm(embedding_matrix, dim=-1).mean().item(),
        'std_norm': torch.norm(embedding_matrix, dim=-1).std().item(),
        'mean_embedding': embedding_matrix.mean(dim=0).tolist(),
        'std_embedding': embedding_matrix.std(dim=0).tolist()
    }

    return stats
