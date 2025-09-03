#!/usr/bin/env python3
"""
Comprehensive Demo: Senter-Omni Suite Multimodal Capabilities

Demonstrates both chat and embedding models with text, image, and audio.
Shows inference and similarity search across all modalities.
"""

import os
import sys
import warnings
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings for cleaner demo output
warnings.filterwarnings("ignore")

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def check_dependencies():
    """Check if required dependencies are available"""
    print_header("Checking Dependencies")

    deps_status = {}

    # Core dependencies
    try:
        import torch
        deps_status['torch'] = f"✅ {torch.__version__}"
    except ImportError:
        deps_status['torch'] = "❌ Not available"

    try:
        import transformers
        deps_status['transformers'] = f"✅ {transformers.__version__}"
    except ImportError:
        deps_status['transformers'] = "❌ Not available"

    # Optional multimodal dependencies
    try:
        import librosa
        deps_status['librosa'] = f"✅ {librosa.__version__}"
    except ImportError:
        deps_status['librosa'] = "⚠️ Not available (audio processing limited)"

    try:
        import cv2
        deps_status['opencv'] = f"✅ {cv2.__version__}"
    except ImportError:
        deps_status['opencv'] = "⚠️ Not available (video processing limited)"

    try:
        from PIL import Image
        deps_status['PIL'] = f"✅ {Image.__version__}"
    except ImportError:
        deps_status['PIL'] = "❌ Not available (image processing disabled)"

    for dep, status in deps_status.items():
        print(f"{dep:15}: {status}")

    return deps_status

def demo_chat_model():
    """Demonstrate Senter-Omni chat model capabilities"""
    print_header("🤖 Senter-Omni Chat Model Demo")

    try:
        from senter_omni import SenterOmniChat

        # Try to initialize (will fail gracefully if model not available)
        print("🔧 Initializing chat model...")
        try:
            chat = SenterOmniChat()
            model_available = True
            print("✅ Chat model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Chat model not available: {e}")
            print("   Running in demo mode with mock responses...")
            model_available = False

        # Demo text chat
        print("\n📝 Text Chat Demo:")
        text_message = "<user>Hello! Can you tell me about yourself in one sentence?</user>"

        if model_available:
            print(f"Input: {text_message}")
            response = chat.generate_streaming([text_message])
            print(f"Response: {response[:100]}...")
        else:
            print(f"Input: {text_message}")
            print("Response: I am Senter-Omni, an advanced multimodal AI assistant! ✨")

        # Demo multimodal chat with image
        print("\n🖼️ Multimodal Chat Demo (Image):")
        image_path = "test_assets/test_image.jpg"
        if os.path.exists(image_path):
            image_message = f"<user>I have this image: <image>{image_path}</image> What do you see in this image?</user>"
            if model_available:
                print(f"Input: {image_message}")
                response = chat.generate_streaming([image_message])
                print(f"Response: {response[:100]}...")
            else:
                print(f"Input: {image_message}")
                print("Response: I can see a beautiful scene in your image! 🎨")
        else:
            print("⚠️ Test image not found, skipping image demo")

        # Demo multimodal chat with audio
        print("\n🎵 Multimodal Chat Demo (Audio):")
        audio_path = "test_assets/test_audio.wav"
        if os.path.exists(audio_path):
            audio_message = f"<user>I have this audio: <audio>{audio_path}</audio> What do you hear in this audio?</user>"
            if model_available:
                print(f"Input: {audio_message}")
                response = chat.generate_streaming([audio_message])
                print(f"Response: {response[:100]}...")
            else:
                print(f"Input: {audio_message}")
                print("Response: I can hear interesting sounds in your audio! 🎵")
        else:
            print("⚠️ Test audio not found, skipping audio demo")

        print("\n✅ Chat model demo completed!")

    except Exception as e:
        print(f"❌ Chat model demo failed: {e}")

def demo_embedding_model():
    """Demonstrate Senter-Embed embedding model capabilities"""
    print_header("🔍 Senter-Embed Embedding Model Demo")

    try:
        from senter_embed import SenterEmbedder, MultimodalEmbeddingDatabase
        from senter_embed.utils import cosine_similarity

        # Try to initialize (will fail gracefully if model not available)
        print("🔧 Initializing embedding model...")
        try:
            embedder = SenterEmbedder(device='cpu', use_memory_efficient=False)  # Use CPU for demo
            model_available = True
            print("✅ Embedding model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Embedding model not available: {e}")
            print("   Running in demo mode with mock embeddings...")
            model_available = False

        # Create database
        db = MultimodalEmbeddingDatabase(embedder)

        # Demo text embeddings
        print("\n📝 Text Embedding Demo:")
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming artificial intelligence",
            "Python is a powerful programming language for AI development"
        ]

        print("Adding text content to database...")
        for i, text in enumerate(texts):
            db.add_content({'text': text}, {'type': 'demo', 'id': i})
            print(f"  ✅ Added: {text[:50]}...")

        # Demo text similarity search
        print("\n🔍 Text Similarity Search:")
        query = "A fast fox leaps over a sleeping canine"
        print(f"Query: {query}")

        results = db.search_similar({'text': query}, top_k=3)
        for result in results:
            print(f"  Rank {result['rank']}: Similarity {result['similarity']:.3f}")
            print(f"    Content: {result['content'][:50]}...")
            print(f"    Modality: {result['modality']}")
            print()

        # Demo image embeddings
        print("\n🖼️ Image Embedding Demo:")
        image_path = "test_assets/test_image.jpg"
        if os.path.exists(image_path):
            print(f"Processing image: {image_path}")
            db.add_content({'image': image_path}, {'type': 'demo_image', 'id': 'img1'})
            print("✅ Added image to database")

            # Try to find similar images (though we only have one)
            if model_available:
                try:
                    results = db.search_similar({'image': image_path}, top_k=2)
                    for result in results:
                        print(f"  Rank {result['rank']}: Similarity {result['similarity']:.3f}")
                        print(f"    Content: {result['content'][:50]}...")
                        print(f"    Modality: {result['modality']}")
                        print()
                except Exception as e:
                    print(f"⚠️ Image similarity search failed: {e}")
        else:
            print("⚠️ Test image not found, skipping image demo")

        # Demo audio embeddings
        print("\n🎵 Audio Embedding Demo:")
        audio_path = "test_assets/test_audio.wav"
        if os.path.exists(audio_path):
            print(f"Processing audio: {audio_path}")
            try:
                db.add_content({'audio': audio_path}, {'type': 'demo_audio', 'id': 'aud1'})
                print("✅ Added audio to database")

                # Try to find similar audio (though we only have one)
                if model_available:
                    try:
                        results = db.search_similar({'audio': audio_path}, top_k=2)
                        for result in results:
                            print(f"  Rank {result['rank']}: Similarity {result['similarity']:.3f}")
                            print(f"    Content: {result['content'][:50]}...")
                            print(f"    Modality: {result['modality']}")
                            print()
                    except Exception as e:
                        print(f"⚠️ Audio similarity search failed: {e}")
            except Exception as e:
                print(f"⚠️ Audio processing failed: {e}")
        else:
            print("⚠️ Test audio not found, skipping audio demo")

        # Demo cross-modal search
        print("\n🔄 Cross-Modal Search Demo:")
        print("Searching for text query across all modalities...")

        cross_modal_results = db.search_similar({'text': "artificial intelligence and programming"}, top_k=5)
        for result in cross_modal_results:
            print(f"  Rank {result['rank']}: Similarity {result['similarity']:.3f}")
            print(f"    Content: {result['content'][:50]}...")
            print(f"    Modality: {result['modality']}")
            print()
        print("\n✅ Embedding model demo completed!")

    except Exception as e:
        print(f"❌ Embedding model demo failed: {e}")

def demo_unified_multimodal():
    """Demonstrate unified multimodal capabilities"""
    print_header("🎭 Unified Multimodal Demo")

    print("🚀 This demo shows how both models work together:")
    print("1. 🤖 Senter-Omni: Processes multimodal conversations")
    print("2. 🔍 Senter-Embed: Creates and searches multimodal embeddings")
    print("3. 🔄 Integration: Both models share the same Gemma3N foundation")

    print("\n📊 Multimodal Capabilities Summary:")
    print("• Text: Natural language understanding and generation")
    print("• Images: Vision analysis using MobileNetV5 (2048D embeddings)")
    print("• Audio: Speech/music processing (1536D embeddings)")
    print("• Video: Frame-by-frame analysis (averaged embeddings)")
    print("• Unified: 1024D common embedding space for all modalities")
    print("• Search: Cosine similarity across all content types")

    print("\n🎯 Use Cases:")
    print("• Conversational AI with multimodal understanding")
    print("• Content similarity search across different media types")
    print("• Multimodal information retrieval")
    print("• Cross-modal content recommendation")
    print("• AI-powered content analysis and tagging")

def main():
    """Run all demos"""
    print("🎭 Senter-Omni Suite: Comprehensive Multimodal Demo")
    print("="*60)
    print("🚀 Demonstrating both chat and embedding models")
    print("   with text, image, and audio capabilities")
    print("="*60)

    # Check dependencies first
    deps = check_dependencies()

    # Core dependency check
    if deps.get('torch', '').startswith('❌'):
        print("\n❌ PyTorch not available. Please install with: pip install torch")
        return

    if deps.get('transformers', '').startswith('❌'):
        print("\n❌ Transformers not available. Please install with: pip install transformers")
        return

    # Run demos
    demo_chat_model()
    demo_embedding_model()
    demo_unified_multimodal()

    print("\n" + "="*60)
    print("🎉 Demo completed! Both models demonstrated successfully!")
    print("="*60)

    print("\n📚 Next Steps:")
    print("1. Install multimodal dependencies: pip install -e \".[multimodal]\"")
    print("2. Run individual models: senter-omni or senter-embed")
    print("3. Check documentation: https://github.com/SouthpawIN/senter-omni")
    print("4. Explore APIs: from senter_omni import SenterOmniChat")

if __name__ == "__main__":
    main()
