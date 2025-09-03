# 🤖 Senter-Omni Suite

**Unified Multimodal AI API: Chat & Embedding with XML Tag Support**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**🎭 Simple, Powerful Interface for Multimodal AI**

Senter-Omni provides a unified API with **three core functions**:
- `omni.train()` - Train new multimodal chat models from datasets
- `omni.chat()` - llama.cpp-style chat completions with full parameter control
- `omni.embed()` - XML-based multimodal embeddings across text, images, and audio

## ✨ Core Features

### 🎭 **Unified API**
- **`omni.train()`** - Train new multimodal chat models from datasets
- **`omni.chat()`** - llama.cpp-style chat with full parameter control
- **`omni.embed()`** - XML-based multimodal embeddings
- **Smart Device Management** - Automatic GPU/CPU allocation
- **Memory Optimization** - Efficient resource usage

### 🤖 **Chat Capabilities**
- **Full Parameter Control** - max_tokens, temperature, top_p, top_k
- **Streaming Support** - Real-time token generation
- **Multimodal Input** - Text, images, and audio with XML tags
- **Custom Stop Sequences** - Flexible response termination
- **OpenAI Compatibility** - `create_chat_completion()` style API

### 🏋️ **Training Capabilities**
- **Dataset Flexibility** - HuggingFace datasets or local JSON/JSONL files
- **Multimodal Fine-tuning** - Vision, audio, and text layer optimization
- **LoRA Efficiency** - Parameter-efficient training with configurable rank
- **Memory Optimized** - 4-bit quantization and gradient checkpointing
- **Progress Monitoring** - Real-time training statistics and memory usage
- **Model Export** - Both LoRA adapters and merged full models

### 🔍 **Embedding & Similarity**
- **XML Tag Interface** - `<text>`, `<image>`, `<audio>` tags
- **Cross-Modal Search** - Similarity across different content types
- **Unified Embeddings** - 1024D common embedding space
- **Real-time Processing** - Fast embedding generation
- **Similarity Thresholding** - Filter results by relevance

### 🧬 **Technical Excellence**
- **Qwen2.5-Omni Foundation** - Apache 2.0 licensed multimodal architecture
- **Precision Optimization** - Fixed NaN issues with autocast
- **Memory Efficient** - GPU memory fragmentation resolved
- **Production Ready** - Robust error handling and fallbacks

## 🚀 Quick Start

### Install Package
```bash
# Install basic functionality
pip install -e .

# Install with multimodal support (recommended)
pip install -e ".[multimodal]"

# Install everything including dev tools
pip install -e ".[all]"
```

### 🎯 **Two Functions, Endless Possibilities**

```python
import omni

# 🎭 That's it! Just two functions for everything
response = omni.chat("Hello!", max_tokens=100, temperature=0.7)
result = omni.embed("<text>AI</text><image>robot.jpg</image>", operation="similarity")
```

## 📚 **Complete Examples**

### 🤖 **Chat Examples**

#### **Basic Chat**
```python
import omni

# Simple chat
response = omni.chat("<user>Hello! Tell me about yourself.</user>")
print(response)
```

#### **Advanced Parameters (llama.cpp Style)**
```python
# Full parameter control like llama.cpp
response = omni.chat(
    messages="<user>What is machine learning?</user>",
    max_tokens=150,        # Maximum response length
    temperature=0.7,       # Creativity (0.0-2.0)
    top_p=0.9,            # Nucleus sampling (0.0-1.0)
    top_k=40,             # Top-k sampling
    stream=False          # Set to True for streaming
)
```

#### **Streaming Chat**
```python
# Real-time streaming
for chunk in omni.chat(
    "<user>Tell me a story about AI.</user>",
    max_tokens=200,
    temperature=0.8,
    stream=True
):
    print(chunk, end="")
```

#### **Multimodal Chat**
```python
# Chat with images and audio
response = omni.chat("""
<user>Describe this scene: <image>beach_sunset.jpg</image>
What do you hear in this audio? <audio>ocean_waves.wav</audio></user>
""", max_tokens=100)
```

#### **Custom Stop Sequences**
```python
# Control response termination
response = omni.chat(
    "<user>Write a haiku about technology.</user>",
    max_tokens=50,
    temperature=0.9,
    stop_sequences=["\\n\\n", "END", "STOP"]
)
```

#### **OpenAI Compatibility**
```python
# Use familiar OpenAI-style API
completion = omni.create_chat_completion(
    messages="<user>What is the future of AI?</user>",
    max_tokens=100,
    temperature=0.8
)
```

### 🔍 **Embedding Examples**

#### **Single Modality**
```python
# Text only
result = omni.embed("<text>Python programming language</text>")
print(f"Embedding shape: {result['embeddings']['text_0']['embedding'].shape}")

# Image only
result = omni.embed("<image>photo.jpg</image>")

# Audio only
result = omni.embed("<audio>music.wav</audio>")
```

#### **Multiple Modalities**
```python
# Any combination of modalities
result = omni.embed("""
<text>Artificial intelligence systems</text>
<image>robot.jpg</image>
<audio>speech.wav</audio>
""")

print(f"Modalities processed: {result['modalities']}")
print(f"Embeddings generated: {len(result['embeddings'])}")
```

#### **Cross-Modal Similarity**
```python
# Find similarities between different content types
result = omni.embed("""
<text>Machine learning algorithms</text>
<image>neural_network_diagram.jpg</image>
""", operation="similarity")

# View all similarity pairs
for pair, score in result['similarities'].items():
    print(f"{pair}: {score:.3f}")
```

#### **Filtered Similarity Search**
```python
# Only show highly similar results
result = omni.embed("""
<text>Geometric patterns and shapes</text>
<image>abstract_art.jpg</image>
<audio>electronic_music.wav</audio>
""",
operation="similarity",
similarity_threshold=0.1  # Only show scores > 0.1
)

print("High similarity pairs:")
for pair, score in result.get('similarities_filtered', {}).items():
    print(f"  {pair}: {score:.3f}")
```

### 🎨 **Real-World Use Cases**

#### **Content Analysis**
```python
# Analyze blog post with featured image
result = omni.embed("""
<text>This article discusses the latest developments in quantum computing and its potential applications in cryptography.</text>
<image>quantum_circuit_diagram.png</image>
""", operation="similarity")
```

#### **Multimedia Search**
```python
# Search for related content across modalities
query_result = omni.embed("""
<text>Find content about space exploration</text>
<image>rocket_launch.jpg</image>
""", operation="similarity")
```

#### **Creative Content Generation**
```python
# Generate story inspired by image and music
story_prompt = omni.chat("""
<user>Write a short story inspired by this image: <image>mystical_forest.jpg</image>
Set the mood with this background music: <audio>ambient_forest_sounds.wav</audio></user>
""", max_tokens=300, temperature=0.8)
```

### 🔧 **Advanced Configuration**

#### **Device Management**
```python
# Automatic device selection (recommended)
import omni  # Uses GPU 0 for chat, GPU 1 for embeddings

# Manual device control
from omni import get_omni_client
client = get_omni_client(chat_device="cuda:0", embed_device="cuda:1")
```

#### **Memory Optimization**
```python
# The system automatically handles memory optimization
# Large models are loaded efficiently with memory management
# GPU memory fragmentation is automatically resolved
```

#### **Batch Processing**
```python
# Process multiple content pieces
contents = [
    "<text>AI ethics</text><image>debate.jpg</image>",
    "<text>Climate change</text><audio>podcast.wav</audio>",
    "<text>Space travel</text><image>rocket.jpg</image><audio>launch.wav</audio>"
]

for content in contents:
    result = omni.embed(content, operation="embed")
    print(f"Processed {len(result['embeddings'])} modalities")
```

## 📖 **API Reference**

### **`omni.train(dataset_name_or_path, **kwargs)`**

Train a new Senter-Omni multimodal chat model from datasets.

**Parameters:**
- `dataset_name_or_path` (str or list): HuggingFace dataset name(s) or local file path(s)
- `output_dir` (str, default="models/senter-omni-trained"): Directory to save trained model
- `max_samples` (int, default=10000): Maximum training samples to use
- `num_epochs` (int, default=3): Number of training epochs
- `learning_rate` (float, default=2e-4): Learning rate for training
- `batch_size` (int, default=2): Batch size per device
- `gradient_accumulation_steps` (int, default=4): Gradient accumulation steps
- `lora_rank` (int, default=16): LoRA rank for fine-tuning
- `save_steps` (int, default=500): Save model every N steps

**Returns:** Dictionary with training statistics, model paths, and metadata

**Examples:**
```python
# Train with HuggingFace datasets
result = omni.train([
    "NousResearch/Hermes-3-Dataset",
    "NousResearch/hermes-function-calling-v1"
])

# Train with local dataset
result = omni.train(
    "my_custom_dataset.jsonl",
    output_dir="models/my-model",
    max_samples=5000,
    num_epochs=5,
    lora_rank=32
)

# Custom training parameters
result = omni.train(
    ["dataset1", "dataset2"],
    learning_rate=1e-4,
    batch_size=4,
    gradient_accumulation_steps=2
)
```

**Training Output:**
```python
{
    "status": "completed",
    "training_stats": {
        "train_runtime": 1250.34,
        "train_samples_per_second": 24.0,
        "total_flos": 1.23e+16
    },
    "model_paths": {
        "lora": "models/my-model/lora",
        "merged": "models/my-model/merged"
    },
    "memory_stats": {
        "start_memory": 2.1,
        "peak_memory": 11.8,
        "max_memory": 12.0,
        "memory_percentage": 98.3
    },
    "dataset_info": {
        "samples_used": 5000,
        "datasets": ["NousResearch/Hermes-3-Dataset"]
    }
}
```

### **`omni.chat(messages, **kwargs)`**

Generate chat completions with full parameter control.

**Parameters:**
- `messages` (str or list): Input messages or XML-formatted content
- `max_tokens` (int, default=256): Maximum tokens to generate
- `temperature` (float, default=0.8): Sampling temperature (0.0-2.0)
- `top_p` (float, default=0.9): Nucleus sampling (0.0-1.0)
- `top_k` (int, default=50): Top-k sampling
- `stream` (bool, default=False): Enable streaming responses
- `stop_sequences` (list, optional): Custom stop sequences

**Returns:** Generated response string (or iterator if streaming)

**Examples:**
```python
# Basic usage
response = omni.chat("<user>Hello!</user>")

# Advanced parameters
response = omni.chat(
    "<user>Explain quantum physics.</user>",
    max_tokens=200,
    temperature=0.7,
    top_p=0.95,
    stream=True
)

# Multimodal
response = omni.chat("""
<user>Describe this image: <image>beach_sunset.jpg</image></user>
""", max_tokens=100)

# Custom stop sequences
response = omni.chat(
    "<user>Write a haiku about AI.</user>",
    stop_sequences=["\n\n", "<end>"]
)
```

### **`omni.embed(input_content, operation="embed", **kwargs)`**

Process multimodal embeddings with XML tags.

**Parameters:**
- `input_content` (str): XML with `<text>`, `<image>`, `<audio>` tags
- `operation` (str): "embed", "similarity", or "search"
- `similarity_threshold` (float, default=0.0): Minimum similarity score

**Returns:** Dictionary with embeddings and similarity results

**XML Tags:**
- `<text>content</text>` - Text to embed
- `<image>path.jpg</image>` - Image file path
- `<audio>path.wav</audio>` - Audio file path

**Examples:**
```python
# Single modality
result = omni.embed("<text>Python programming</text>")

# Multiple modalities
result = omni.embed("""
<text>AI systems</text>
<image>robot.jpg</image>
<audio>speech.wav</audio>
""")

# Similarity search
result = omni.embed("""
<text>Machine learning</text>
<image>neural_net.jpg</image>
""", operation="similarity")

# Filtered similarity
result = omni.embed("""
<text>Deep learning</text>
<image>cnn_diagram.jpg</image>
""", operation="similarity", similarity_threshold=0.7)
```

### **Response Format**

**Chat Response:**
```python
# Simple string response
"Hello! I'm Senter-Omni, your multimodal AI assistant..."
```

**Embedding Response:**
```python
{
    "operation": "similarity",
    "modalities": ["text", "image", "audio"],
    "embeddings": {
        "text_0": {
            "content": "Machine learning",
            "embedding": torch.Tensor([...])  # 4096D vector
        },
        "image_0": {
            "content": "neural_net.jpg",
            "embedding": torch.Tensor([...])  # 2048D vector
        }
    },
    "similarities": {
        "text_0_vs_image_0": 0.035,
        "text_0_vs_audio_0": 0.000
    }
}
```

## 🎯 **Interactive Mode**

```bash
# Start interactive chat
senter-omni

# Try these XML examples:
<system>You are a helpful AI assistant.</system>
<user>Hello!</user>
<user>I have this image: <image>test_assets/real_test_image.jpg</image> What do you see?</user>
```

### **Interactive Commands**
- `help` - Learn about XML tags and commands
- `history` - View your conversation history
- `clear` - Start a fresh conversation
- `quit` - Exit the chat

### **XML Tag Quick Reference**

```bash
# Chat with system prompts
<system>You are a helpful AI assistant.</system>
<user>Hello! Tell me about yourself.</user>

# Multimodal chat
<user>Describe this image: <image>photo.jpg</image></user>
<user>Analyze this audio: <audio>sound.wav</audio></user>

# Embedding with XML tags
<text>Artificial intelligence content</text>
<image>ai_diagram.jpg</image>
<audio>ai_podcast.wav</audio>
```

## 📁 Project Structure

```
senter-omni-suite/
├── omni.py                  # 🎭 UNIFIED API (Main Interface)
├── __init__.py              # Package initialization
├── senter_omni/             # Chat model package
│   ├── core.py              # Qwen2.5-Omni chat functionality
│   └── cli.py               # Chat CLI interface
├── senter_embed/            # Embedding model package
│   ├── core.py              # Multimodal embeddings
│   ├── database.py          # Similarity database
│   └── utils.py             # Utility functions
├── gpu_memory_optimizer.py  # Memory optimization tools
├── example_usage.py         # Comprehensive examples
├── models/                  # Model files (Qwen2.5-Omni + LoRA)
├── test_assets/             # Test files (images, audio)
├── requirements.txt         # Dependencies
└── README.md                # This documentation
```

## ⚙️ **System Requirements**

### **Hardware**
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090/4090 recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space for models

### **Software**
- **Python**: 3.8+
- **CUDA**: 12.0+ (for GPU acceleration)
- **PyTorch**: 2.0+
- **Transformers**: 4.40+

### **Installation**
```bash
# Clone repository
git clone https://github.com/SouthpawIN/senter-omni.git
cd senter-omni

# Install with all dependencies (recommended for training)
pip install -e ".[all]"

# Or install specific components
pip install -e .              # Basic chat functionality
pip install -e ".[embed]"     # Add embedding support
pip install -e ".[train]"     # Add training capabilities

# For training, also install:
pip install unsloth trl datasets
```

### **Training Setup**
```bash
# GPU Requirements for Training
# - NVIDIA GPU with 16GB+ VRAM (RTX 3090/4090 recommended)
# - CUDA 12.0+
# - 32GB+ system RAM

# Install training dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth transformers accelerate trl datasets
```

## 🚀 **Quick Start Examples**

### **1. Basic Chat**
```python
import omni

response = omni.chat("Hello! Tell me about AI.")
print(response)
```

### **2. Advanced Chat with Parameters**
```python
response = omni.chat(
    "<user>Explain quantum computing.</user>",
    max_tokens=200,
    temperature=0.7,
    stream=True
)
```

### **3. Multimodal Chat**
```python
response = omni.chat("""
<user>Describe this scene: <image>beach.jpg</image>
What emotions does this music convey? <audio>piano.wav</audio></user>
""")
```

### **4. Embedding & Similarity**
```python
result = omni.embed("""
<text>Machine learning algorithms</text>
<image>neural_network.jpg</image>
""", operation="similarity")

print(f"Similarity: {result['similarities']['text_0_vs_image_0']:.3f}")
```

### **5. Cross-Modal Search**
```python
# Find related content across modalities
result = omni.embed("""
<text>Find content about space exploration</text>
<image>rocket.jpg</image>
<audio>launch_sounds.wav</audio>
""", operation="similarity", similarity_threshold=0.1)
```

### **6. Complete Workflow: Train → Chat → Embed**
```python
import omni

# Step 1: Train a custom model (optional)
print("🚀 Training custom model...")
training_result = omni.train(
    ["NousResearch/Hermes-3-Dataset"],  # Your dataset
    output_dir="models/my-space-model",
    max_samples=2000,
    num_epochs=2
)
print(f"✅ Model trained: {training_result['model_paths']['merged']}")

# Step 2: Chat with the model
print("\\n🤖 Chatting with trained model...")
response = omni.chat("""
<user>Explain the concept of space-time curvature.</user>
""", max_tokens=150, temperature=0.8)
print(f"Response: {response[:100]}...")

# Step 3: Create embeddings for content analysis
print("\\n🔍 Creating multimodal embeddings...")
embed_result = omni.embed("""
<text>The theory of general relativity revolutionized physics</text>
<image>black_hole.jpg</image>
<audio>cosmic_background.wav</audio>
""", operation="similarity")

print(f"📊 Processed {len(embed_result['modalities'])} modalities")
print(f"🔗 Found {len(embed_result['similarities'])} similarity pairs")
for pair, score in embed_result['similarities'].items():
    print(f"  {pair}: {score:.3f}")

# Step 4: Use embeddings for content retrieval
print("\\n📈 Content analysis complete!")
print("🎉 Full multimodal AI pipeline executed successfully!")
```

## 🎯 **Model Capabilities**

### 🤖 **Advanced AI Features**
- **Superior Reasoning** - Fine-tuned on Hermes dataset for complex problem-solving
- **Code Generation** - Multi-language programming support
- **Mathematical Solving** - Advanced calculation and analysis
- **Creative Writing** - Stories, poems, and content generation
- **Technical Analysis** - Deep explanations of complex topics

### 🔧 **Function Calling & Tools**
- **API Integration** - Seamless tool and service connections
- **Parameter Validation** - Robust input handling and error recovery
- **Dynamic Responses** - Context-aware output generation

### 🎨 **Multimodal Understanding**
- **🖼️ Vision Analysis** - Detailed image descriptions and analysis
- **🎵 Audio Processing** - Speech recognition and sound analysis
- **🔄 Cross-Modal Reasoning** - Integrated understanding across modalities
- **📝 XML Interface** - Clean, structured multimodal input

## 📈 **Performance & Quality**

### **Technical Achievements**
- ✅ **Qwen2.5-Omni Foundation** - Apache 2.0 licensed multimodal architecture
- ✅ **Precision Optimization** - Fixed NaN issues with autocast
- ✅ **Memory Efficiency** - GPU memory fragmentation resolved
- ✅ **Cross-Modal Search** - Unified similarity across 1024D space
- ✅ **Production Ready** - Robust error handling and fallbacks

### **Quality Metrics**
- **Chat Accuracy**: Advanced reasoning with XML tag support
- **Embedding Quality**: 4096D text, 2048D image, 1536D audio vectors
- **Similarity Precision**: Cross-modal search with configurable thresholds
- **Memory Usage**: Optimized for RTX 3090/4090 GPUs

## 🏆 **Why Senter-Omni?**

### **🎭 Unified Experience**
- **Two Functions**: `omni.chat()` and `omni.embed()` for everything
- **XML Interface**: Consistent tagging across all modalities
- **Parameter Control**: Full llama.cpp-style generation parameters
- **Smart Management**: Automatic GPU/CPU allocation

### **🚀 Production Ready**
- **Stable Performance**: Fixed all precision and memory issues
- **Comprehensive Testing**: Examples across text, image, and audio
- **Error Handling**: Graceful fallbacks and recovery
- **Documentation**: Complete API reference and examples

### **🔬 Research Grade**
- **Qwen2.5-Omni Integration**: Apache 2.0 licensed multimodal architecture
- **Cross-Modal Understanding**: True multimodal reasoning
- **Extensible Design**: Easy to add new modalities
- **Open Source**: Fully transparent and modifiable

## 📞 **Support & Community**

### **Getting Help**
- **📖 Documentation**: Comprehensive examples and API reference
- **🐛 Issues**: GitHub issues for bug reports and feature requests
- **💬 Discussions**: Community forum for questions and ideas

### **Contributing**
- **🔧 Pull Requests**: Welcome contributions and improvements
- **📝 Documentation**: Help improve examples and guides
- **🧪 Testing**: Additional test cases and validation

## 🙏 **Acknowledgments**

- **Alibaba Cloud** - Qwen2.5-Omni architecture and models
- **Unsloth Team** - Optimization insights and fixes
- **Hugging Face** - Transformers library and model hosting
- **PyTorch Team** - Deep learning framework
- **Open Source Community** - Libraries and tools that made this possible

---

**🎉 Senter-Omni: Where Simple Meets Powerful**

**Two functions. Endless multimodal possibilities.** 🚀✨
