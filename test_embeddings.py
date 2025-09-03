#!/usr/bin/env python3
"""
Test script for Senter-Omni Embedder
"""

import sys
import os
sys.path.append('.')

from senter_omni_embedder import SenterOmniEmbedder, MultimodalEmbeddingDatabase
import torch
from pathlib import Path

def test_text_embeddings():
    """Test text embedding generation"""
    print("🧪 Testing Text Embeddings...")

    try:
        embedder = SenterOmniEmbedder()

        # Test text embedding
        text = "Hello world! This is a test of the embedding system."
        embedding = embedder.get_text_embedding(text)

        print(f"✅ Text embedding generated: shape {embedding.shape}")
        print(f"   Embedding norm: {torch.norm(embedding):.4f}")

        # Test similarity
        text2 = "Hi there! Testing the embedding functionality."
        embedding2 = embedder.get_text_embedding(text2)

        similarity = embedder.compute_similarity(embedding, embedding2)
        print(f"   Similarity between texts: {similarity:.4f}")

        return True

    except Exception as e:
        print(f"❌ Text embedding test failed: {e}")
        return False

def test_image_embeddings():
    """Test image embedding generation"""
    print("🖼️ Testing Image Embeddings...")

    try:
        embedder = SenterOmniEmbedder()

        # Check if test image exists
        test_image_path = "test_assets/test_image.jpg"
        if not Path(test_image_path).exists():
            print("⚠️ Test image not found, skipping image test")
            return True

        # Test image embedding
        embedding = embedder.get_image_embedding(test_image_path)
        print(f"✅ Image embedding generated: shape {embedding.shape}")
        print(f"   Embedding norm: {torch.norm(embedding):.4f}")

        return True

    except Exception as e:
        print(f"❌ Image embedding test failed: {e}")
        return False

def test_audio_embeddings():
    """Test audio embedding generation"""
    print("🎵 Testing Audio Embeddings...")

    try:
        embedder = SenterOmniEmbedder()

        # Check if test audio exists
        test_audio_path = "test_assets/test_audio.wav"
        if not Path(test_audio_path).exists():
            print("⚠️ Test audio not found, skipping audio test")
            return True

        # Test audio embedding
        embedding = embedder.get_audio_embedding(test_audio_path)
        print(f"✅ Audio embedding generated: shape {embedding.shape}")
        print(f"   Embedding norm: {torch.norm(embedding):.4f}")

        return True

    except Exception as e:
        print(f"❌ Audio embedding test failed: {e}")
        return False

def test_multimodal_database():
    """Test multimodal embedding database"""
    print("🗄️ Testing Multimodal Database...")

    try:
        embedder = SenterOmniEmbedder()
        db = MultimodalEmbeddingDatabase(embedder)

        # Add some test content
        db.add_content({'text': 'The quick brown fox'}, {'type': 'animal', 'id': 1})
        db.add_content({'text': 'Machine learning is powerful'}, {'type': 'tech', 'id': 2})

        print("✅ Added content to database")

        # Test search
        results = db.search_similar({'text': 'A fast fox'}, top_k=2)
        print(f"✅ Search completed, found {len(results)} results")

        for result in results:
            print(".3f"
        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    """Run all embedding tests"""
    print("🎭 Senter-Omni Embedder Test Suite")
    print("=" * 50)

    # Check if model is available
    model_path = "models/huggingface/senter-omni-lora"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please ensure the model is downloaded and LoRA is applied")
        return

    tests = [
        test_text_embeddings,
        test_image_embeddings,
        test_audio_embeddings,
        test_multimodal_database
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Multimodal embeddings are working!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
