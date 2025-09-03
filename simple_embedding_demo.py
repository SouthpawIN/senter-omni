#!/usr/bin/env python3
"""
Simple CPU-based demo of Senter-Omni embedding concepts
This demonstrates the embedding functionality without requiring GPU resources
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import warnings

class SimpleEmbeddingDemo:
    """
    Simplified embedding demo that shows the concepts without full model loading
    """

    def __init__(self):
        # Use same dimension for all embeddings in demo
        self.embedding_dim = 256  # Common embedding dimension

        # Mock embeddings database
        self.embeddings_db = []
        self.metadata_db = []

    def mock_text_embedding(self, text: str) -> np.ndarray:
        """Generate mock text embedding based on text content"""
        # Simple hash-based embedding for demo
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to embedding-like values
        embedding = np.array([b / 255.0 for b in hash_bytes], dtype=np.float32)

        # Expand to desired dimension
        embedding = np.tile(embedding, self.embedding_dim // len(embedding) + 1)[:self.embedding_dim]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def mock_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate mock image embedding"""
        # Use file size and name as features
        import os
        if os.path.exists(image_path):
            file_size = os.path.getsize(image_path)
            file_name = os.path.basename(image_path)

            # Create embedding from file characteristics
            features = [file_size % 1000, len(file_name), hash(file_name) % 1000]
            embedding = np.array(features * (self.embedding_dim // len(features) + 1))[:self.embedding_dim]
        else:
            # Default embedding for missing files
            embedding = np.random.randn(self.embedding_dim)

        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)

    def mock_audio_embedding(self, audio_path: str) -> np.ndarray:
        """Generate mock audio embedding"""
        import os
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            features = [file_size % 500, file_size % 300, file_size % 700]
            embedding = np.array(features * (self.embedding_dim // len(features) + 1))[:self.embedding_dim]
        else:
            embedding = np.random.randn(self.embedding_dim)

        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def add_to_database(self, content: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Add content to the mock database"""
        embeddings = {}

        if 'text' in content:
            embeddings['text'] = self.mock_text_embedding(content['text'])

        if 'image' in content:
            embeddings['image'] = self.mock_image_embedding(content['image'])

        if 'audio' in content:
            embeddings['audio'] = self.mock_audio_embedding(content['audio'])

        # Store the primary embedding (use first available)
        if embeddings:
            primary_modality = list(embeddings.keys())[0]
            self.embeddings_db.append(embeddings[primary_modality])
            self.metadata_db.append({
                'modality': primary_modality,
                'content': content.get(primary_modality),
                'metadata': metadata or {},
                'all_embeddings': embeddings
            })

    def search_similar(self, query_content: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar content in the database"""
        if not self.embeddings_db:
            return []

        # Generate query embedding
        query_emb = None
        query_modality = None

        if 'text' in query_content:
            query_emb = self.mock_text_embedding(query_content['text'])
            query_modality = 'text'
        elif 'image' in query_content:
            query_emb = self.mock_image_embedding(query_content['image'])
            query_modality = 'image'
        elif 'audio' in query_content:
            query_emb = self.mock_audio_embedding(query_content['audio'])
            query_modality = 'audio'

        if query_emb is None:
            return []

        # Find similar items
        similarities = []
        for i, db_emb in enumerate(self.embeddings_db):
            similarity = self.cosine_similarity(query_emb, db_emb)
            similarities.append((i, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for rank, (idx, score) in enumerate(similarities[:top_k]):
            metadata = self.metadata_db[idx]
            results.append({
                'rank': rank + 1,
                'similarity': score,
                'modality': metadata['modality'],
                'content': metadata['content'],
                'metadata': metadata['metadata']
            })

        return results


def demo_multimodal_embeddings():
    """Demonstrate the multimodal embedding concepts"""
    print("🎭 Senter-Omni Multimodal Embedding Demo")
    print("=" * 60)

    demo = SimpleEmbeddingDemo()

    print("\n📝 Adding sample content to database...")

    # Add sample content
    demo.add_to_database({'text': 'The quick brown fox jumps over the lazy dog'}, {'type': 'sample', 'category': 'animals'})
    demo.add_to_database({'text': 'Machine learning is transforming artificial intelligence'}, {'type': 'sample', 'category': 'technology'})
    demo.add_to_database({'text': 'Python is a powerful programming language'}, {'type': 'sample', 'category': 'programming'})

    # Mock image and audio (would use real files in production)
    demo.add_to_database({'image': '/mock/image1.jpg'}, {'type': 'sample', 'category': 'nature'})
    demo.add_to_database({'audio': '/mock/audio1.wav'}, {'type': 'sample', 'category': 'music'})

    print("✅ Added content to database")

    print("\n🔍 Searching for similar content...")

    # Text search
    text_query = {'text': 'A fast fox leaps over a sleeping canine'}
    results = demo.search_similar(text_query, top_k=2)

    print("📊 Text similarity results:")
    for result in results:
        print(f"  Rank {result['rank']}: Similarity {result['similarity']:.3f}")
        print(f"    Content: {result['content'][:50]}...")
        print(f"    Category: {result['metadata'].get('category', 'unknown')}")
        print()

    # Show what real implementation would do
    print("🚀 Real Implementation Features:")
    print("  • Text embeddings from Gemma3N hidden states (4096D)")
    print("  • Image embeddings from MobileNetV5 vision encoder (2048D)")
    print("  • Audio embeddings from Gemma3N audio encoder (1536D)")
    print("  • Video embeddings via frame extraction and averaging")
    print("  • Unified embedding space projection (1024D)")
    print("  • Cosine similarity search across modalities")
    print("  • Memory-efficient loading with 4-bit quantization")
    print()
    print("💡 To use the real implementation:")
    print("   from senter_omni_embedder import SenterOmniEmbedder")
    print("   embedder = SenterOmniEmbedder()")
    print("   text_emb = embedder.get_text_embedding('Your text here')")


if __name__ == "__main__":
    demo_multimodal_embeddings()
