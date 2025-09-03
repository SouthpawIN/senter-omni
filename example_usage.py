#!/usr/bin/env python3
"""
🎭 Senter-Omni Unified API Examples

Complete examples showing how to use the unified omni.chat() and omni.embed() functions.
"""

import omni

def demo_chat_api():
    """Demonstrate omni.chat() function with various parameters"""
    print("🤖 OMNI.CHAT() EXAMPLES")
    print("=" * 50)

    # Basic chat
    print("\n1️⃣ Basic Chat:")
    response = omni.chat("<user>Hello! Tell me about yourself.</user>")
    print(f"Response: {response[:100]}...")

    # Chat with parameters (llama.cpp style)
    print("\n2️⃣ Chat with Generation Parameters:")
    response = omni.chat(
        messages="<user>What is machine learning?</user>",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
        top_k=40
    )
    print(f"Response: {response[:100]}...")

    # Multimodal chat
    print("\n3️⃣ Multimodal Chat:")
    response = omni.chat(
        messages="<user>Describe this image: <image>test_assets/real_test_image.jpg</image></user>",
        max_tokens=100,
        temperature=0.8
    )
    print(f"Response: {response[:100]}...")

    # Streaming chat
    print("\n4️⃣ Streaming Chat:")
    print("Streaming response: ", end="")
    for chunk in omni.chat(
        messages="<user>Tell me a short joke.</user>",
        max_tokens=30,
        temperature=0.9,
        stream=True
    ):
        print(chunk, end="")
    print()

    # OpenAI-style API
    print("\n5️⃣ OpenAI-style API:")
    completion = omni.create_chat_completion(
        messages="<user>What is AI?</user>",
        max_tokens=40,
        temperature=0.5
    )
    print(f"Completion: {completion[:80]}...")

def demo_embed_api():
    """Demonstrate omni.embed() function with XML tags"""
    print("\\n🔍 OMNI.EMBED() EXAMPLES")
    print("=" * 50)

    # Single modality
    print("\\n1️⃣ Single Text Modality:")
    result = omni.embed("<text>Python programming language</text>")
    print(f"Modalities: {result['modalities']}")
    print(f"Embeddings: {len(result['embeddings'])}")

    # Multiple modalities
    print("\\n2️⃣ Multiple Modalities:")
    result = omni.embed("""
    <text>Machine learning algorithms</text>
    <image>test_assets/real_test_image.jpg</image>
    <audio>test_assets/pure_tone_440hz.wav</audio>
    """)
    print(f"Modalities: {result['modalities']}")
    print(f"Embeddings: {len(result['embeddings'])}")

    # Similarity search
    print("\\n3️⃣ Cross-Modal Similarity:")
    result = omni.embed("""
    <text>Geometric shapes and patterns</text>
    <image>test_assets/real_test_image.jpg</image>
    """, operation="similarity")

    if 'similarities' in result:
        print("Similarities:")
        for pair, score in result['similarities'].items():
            print(".3f")

    # Filtered similarity
    print("\\n4️⃣ Filtered Similarity (threshold > 0.01):")
    result = omni.embed("""
    <text>Artificial intelligence</text>
    <image>test_assets/real_test_image.jpg</image>
    <audio>test_assets/pure_tone_440hz.wav</audio>
    """, operation="similarity", similarity_threshold=0.01)

    if 'similarities_filtered' in result:
        print("Filtered similarities:")
        for pair, score in result['similarities_filtered'].items():
            print(".3f")

def demo_advanced_usage():
    """Advanced usage examples"""
    print("\\n🚀 ADVANCED USAGE EXAMPLES")
    print("=" * 50)

    # Custom stop sequences
    print("\\n1️⃣ Custom Stop Sequences:")
    response = omni.chat(
        messages="<user>Write a haiku about AI.</user>",
        max_tokens=50,
        temperature=0.8,
        stop_sequences=["\\n\\n", "END"]
    )
    print(f"Haiku: {response}")

    # Batch processing (conceptual)
    print("\\n2️⃣ Processing Multiple Content Types:")
    contents = [
        "<text>Programming</text><image>test_assets/real_test_image.jpg</image>",
        "<text>Music theory</text><audio>test_assets/pure_tone_440hz.wav</audio>",
        "<text>Visual art</text><image>test_assets/real_test_image.jpg</image>"
    ]

    for i, content in enumerate(contents, 1):
        print(f"\\nContent {i}:")
        result = omni.embed(content, operation="embed")
        print(f"  Modalities: {result['modalities']}")
        print(f"  Embeddings: {len(result['embeddings'])}")

def main():
    """Run all demonstrations"""
    print("🎭 Senter-Omni Unified API Complete Demo")
    print("=" * 60)

    try:
        demo_chat_api()
        demo_embed_api()
        demo_advanced_usage()

        print("\\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📖 API REFERENCE:")
        print("• omni.chat(messages, max_tokens=256, temperature=0.8, ...) - Chat completions")
        print("• omni.embed(content_xml, operation='embed', ...) - Multimodal embeddings")
        print("• omni.create_chat_completion(...) - OpenAI-style API")
        print("• omni.generate(...) - llama.cpp-style generation")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure models are properly installed and GPU memory is available")

if __name__ == "__main__":
    main()
