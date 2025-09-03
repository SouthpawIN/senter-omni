#!/usr/bin/env python3
"""
Senter-Embed Multimodal Database

Database for storing and retrieving multimodal embeddings.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Union
import json
import pickle
from pathlib import Path
from .core import SenterEmbedder


class MultimodalEmbeddingDatabase:
    """
    Database for storing and retrieving multimodal embeddings
    """

    def __init__(self, embedder: SenterEmbedder):
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
        if not self.embeddings:
            return []

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

    def save_database(self, filepath: Union[str, Path]):
        """
        Save database to disk

        Args:
            filepath: Path to save the database
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to numpy for serialization
        embeddings_np = [emb.detach().cpu().numpy() for emb in self.embeddings]

        data = {
            'embeddings': embeddings_np,
            'metadata': self.metadata,
            'modalities': self.modalities
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"✅ Database saved to {filepath}")

    def load_database(self, filepath: Union[str, Path]):
        """
        Load database from disk

        Args:
            filepath: Path to load the database from
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Database file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Convert numpy arrays back to tensors
        self.embeddings = [torch.from_numpy(emb) for emb in data['embeddings']]
        self.metadata = data['metadata']
        self.modalities = data['modalities']

        print(f"✅ Database loaded from {filepath}")
        print(f"   Loaded {len(self.embeddings)} embeddings")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with database statistics
        """
        modality_counts = {}
        for modality in self.modalities:
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        return {
            'total_embeddings': len(self.embeddings),
            'modalities': modality_counts,
            'embedding_shape': self.embeddings[0].shape if self.embeddings else None
        }
