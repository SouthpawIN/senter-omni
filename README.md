# 🤖 Senter-Omni Suite

**Dual Multimodal AI Models: Chat & Embedding with XML Tag Support**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Two Powerful Models

### 🤖 Senter-Omni (Chat Model)
Advanced conversational AI with XML tag support for multimodal interactions.

### 🔍 Senter-Embed (Embedding Model)
Comprehensive multimodal embedding system for similarity search across text, images, audio, and video.

## ✨ Features

### Chat Model (Senter-Omni)
- **🧠 Advanced Reasoning** - Fine-tuned on Hermes-3-Dataset for superior problem-solving
- **🔧 Function Calling** - Enhanced tool use and API integration capabilities
- **🆓 Uncensored** - Direct, unrestricted responses without content filters
- **👁️ Multimodal Ready** - Vision and audio understanding architecture
- **⚡ Efficient** - LoRA fine-tuning for optimal performance
- **🎯 XML Tag Support** - Clean role-based conversation formatting
- **🛑 Stop Token Handling** - Automatic conversation termination
- **📝 Gemma3N Integration** - Native support for Gemma3N chat format

### Embedding Model (Senter-Embed)
- **🔍 Similarity Search** - Find similar content across modalities
- **📊 Multimodal Embeddings** - Text, image, audio, and video embeddings
- **🗄️ Database Support** - Persistent storage and retrieval
- **🔄 Unified Space** - Common embedding dimension for all modalities
- **⚡ Memory Efficient** - 4-bit quantization support
- **🎵 Audio Processing** - Speech and music embedding
- **🎥 Video Processing** - Frame-by-frame analysis
- **📈 Cosine Similarity** - Efficient similarity computation

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

### 1. Chat Model - Interactive Mode
```bash
# Start interactive chat
senter-omni

# Or use Python API
python -c "from senter_omni import SenterOmniChat; chat = SenterOmniChat()"
```

### 2. Embedding Model - Similarity Search
```bash
# Start embedding CLI
senter-embed demo  # Run demo
senter-embed embed --text "Hello world!"  # Generate embeddings
senter-embed db add --text "Sample text" --db my_database.pkl  # Add to database
senter-embed db search --text "Similar text" --db my_database.pkl  # Search database

# Or use Python API
python -c "
from senter_embed import SenterEmbedder, MultimodalEmbeddingDatabase
embedder = SenterEmbedder()
db = MultimodalEmbeddingDatabase(embedder)
db.add_content({'text': 'Sample content'})
results = db.search_similar({'text': 'Similar query'})
"
```

### 3. XML Tag Examples
```bash
# System prompts
<system>You are a helpful AI assistant.</system>

# User messages
<user>Hello! Tell me about yourself.</user>

# Assistant responses
<assistant>I'm Senter-Omni, your AI companion!</assistant>

# Multimodal content
<user>I have this image: <image>A beautiful sunset</image> What do you see?</user>
```

## 📁 Project Structure

```
senter-omni-suite/
├── senter_omni/              # Chat model package
│   ├── __init__.py          # Package initialization
│   ├── core.py              # Main chat functionality
│   ├── embedder.py         # Lightweight embedding for chat
│   └── cli.py               # Command-line interface
├── senter_embed/            # Embedding model package
│   ├── __init__.py          # Package initialization
│   ├── core.py              # Main embedding functionality
│   ├── database.py          # Database operations
│   ├── utils.py             # Utility functions
│   └── cli.py               # Command-line interface
├── senter_omni_cli.py       # CLI entry point for chat model
├── senter_embed_cli.py      # CLI entry point for embedding model
├── simple_embedding_demo.py # CPU-based demo
├── advanced_chat.py         # Legacy chat interface
├── senter_omni_embedder.py  # Legacy embedding interface
├── models/                  # Model files
│   └── huggingface/
│       └── senter-omni-lora/ # LoRA adapter with Gemma3N
├── scripts/                 # Utility scripts
├── notebooks/               # Training notebooks
├── test_assets/             # Test files
├── requirements.txt         # Dependencies
├── setup_package.py         # Package setup
└── README.md               # This file
```

## 🎯 Model Capabilities

### Text Understanding
- Mathematical problem solving
- Code generation and explanation
- Logical reasoning and analysis
- Creative writing and storytelling
- Technical explanations

### Function Calling
- Tool integration patterns
- API interaction examples
- Parameter passing and validation
- Error handling and recovery

### Multimodal Architecture
- **Vision**: Image analysis and description
- **Audio**: Speech and sound understanding
- **Combined**: Multi-input reasoning

## 💻 Usage Examples

### Senter-Omni (Chat Model)
```bash
# Interactive chat with XML support
senter-omni

# API mode with JSON input
echo '{"messages": [{"role": "user", "content": "Hello!"}]}' | senter-omni --api

# Python API
from senter_omni import SenterOmniChat
chat = SenterOmniChat()
response = chat.generate_streaming(["<user>Hello!</user>"])
```

**Chat Commands:**
- `help` - Show help information and XML examples
- `clear` - Clear conversation history
- `history` - View conversation history
- `video` - Check video support capabilities
- `quit` - Exit chat

### Senter-Embed (Embedding Model)
```bash
# Run demo
senter-embed demo

# Generate embeddings
senter-embed embed --text "Hello world!" --output embeddings.json

# Database operations
senter-embed db add --text "Sample text" --db my_db.pkl
senter-embed db search --text "Similar query" --db my_db.pkl --top-k 5
senter-embed db info --db my_db.pkl

# Python API
from senter_embed import SenterEmbedder, MultimodalEmbeddingDatabase
embedder = SenterEmbedder()
db = MultimodalEmbeddingDatabase(embedder)
```

### XML Tag Usage

**Role-based Conversations:**
```xml
<system>You are Senter-Omni, a helpful AI assistant with multimodal capabilities.</system>
<user>Hello! Can you introduce yourself?</user>
<assistant>Hi! I'm Senter-Omni, your advanced AI companion...</assistant>
```

**Plain Text (Auto-converted):**
```
Hello! Tell me about yourself.
```
→ Automatically treated as `<user>Hello! Tell me about yourself.</user>`

**Multimodal Content:**
```xml
<user>
I have this image: <image>A beautiful sunset over mountains</image>
What do you see in this scene?
</user>
```

### Example Conversations

**Math & Reasoning:**
```xml
<user>Solve 15 × 23 + 7</user>
<!-- Senter-Omni automatically handles stop tokens and formatting -->
```

**Code Generation:**
```xml
<user>Write a Python function to check if a number is prime</user>
<assistant>Here's an efficient implementation...</assistant>
```

**Multimodal Analysis:**
```xml
<user>
I have this audio: <audio>Gentle piano music</audio>
Describe the mood this creates.
</user>
```

## 🔧 Technical Details

### Base Model
- **Architecture**: Gemma3N 4B (Google DeepMind)
- **Training**: Instruction-tuned with multimodal capabilities
- **Multimodal**: Vision + Audio + Text support
- **Chat Format**: Native Gemma3N conversation format

### Fine-tuning
- **Method**: LoRA (Low-Rank Adaptation)
- **Rank**: 16
- **Training Data**: Hermes-3-Dataset conversations
- **Sources**: Hermes-3-Dataset + Function Calling v1
- **Special Tokens**: Custom stop token handling

### XML Tag Processing
- **Role Parsing**: Automatic `<user>`, `<system>`, `<assistant>` detection
- **Multimodal**: `<image>`, `<audio>`, `<video>` tag support
- **Format Conversion**: XML → Gemma3N chat template
- **Stop Tokens**: Automatic `<end_of_turn>` and `<start_of_turn>` handling

### Training Results
- **Final Loss**: 1.96
- **Convergence**: Excellent (2.99 → 1.96)
- **Training Time**: ~5 minutes
- **Memory Usage**: ~12GB peak

## 📊 Performance

- **Inference Speed**: Fast on modern GPUs
- **Memory Efficient**: LoRA optimization
- **Response Quality**: High (Hermes-trained)
- **Multimodal Ready**: Architecture supports vision/audio

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 16GB+ VRAM (for GPU acceleration)

## 📦 Installation

1. **Clone and setup:**
```bash
cd /home/sovthpaw/Desktop/senter-omni
source venv/bin/activate
```

2. **Dependencies are pre-installed in the virtual environment**

## 🎮 Interactive Features

The advanced chat interface provides:
- **XML Tag Support**: Clean role-based conversation formatting
- **Real-time Streaming**: Token-by-token response generation
- **Stop Token Handling**: Automatic conversation termination
- **Conversation History**: Persistent chat context
- **Command System**: Built-in utilities and help
- **Error Handling**: Robust recovery from generation errors
- **Multimodal Ready**: Image, audio, and video tag support

## 🔍 Embedding & Similarity Search

### Multimodal Embeddings
```python
from senter_omni_embedder import SenterOmniEmbedder, MultimodalEmbeddingDatabase

# Initialize embedder
embedder = SenterOmniEmbedder()

# Generate embeddings for different modalities
text_embedding = embedder.get_text_embedding("Hello world!")
image_embedding = embedder.get_image_embedding("path/to/image.jpg")
audio_embedding = embedder.get_audio_embedding("path/to/audio.wav")
video_embedding = embedder.get_video_embedding("path/to/video.mp4")

# Create multimodal database
db = MultimodalEmbeddingDatabase(embedder)
db.add_content({'text': 'Sample text'}, {'metadata': 'example'})
db.add_content({'image': 'image.jpg'}, {'type': 'photo'})

# Search for similar content
results = db.search_similar({'text': 'Similar text query'}, top_k=5)
```

### Test Embeddings
```bash
# Full implementation test (requires GPU/CPU with model)
python test_embeddings.py

# Simple CPU demo (no model required)
python simple_embedding_demo.py
```

**Features:**
- **Text Embeddings**: From Gemma3N hidden states with mean pooling (4096D)
- **Image Embeddings**: Using MobileNetV5 vision encoder (2048D)
- **Audio Embeddings**: Using Gemma3N audio encoder (1536D)
- **Video Embeddings**: Frame-by-frame processing and averaging
- **Unified Space**: Optional projection to common embedding dimension (1024D)
- **Similarity Search**: Cosine similarity with efficient retrieval
- **Multimodal Database**: Store and search across different content types
- **Memory Efficient**: 4-bit quantization support for reduced VRAM usage
- **CPU Demo**: Simple demonstration without model loading requirements

## 🔌 API Usage

### Programmatic Interface
```python
from advanced_chat import AdvancedSenterOmni

# Initialize the model
ai = AdvancedSenterOmni()

# Simple text generation
messages = [{"role": "user", "content": "Hello!"}]
response = ai.generate_streaming(messages)

# XML tag support
xml_messages = ["<user>Hello! Tell me about yourself.</user>"]
response = ai.generate_streaming(xml_messages)

# OpenAI-compatible format
completion = ai.chat_completion(messages)
```

### Streaming Responses
```python
# Real-time token streaming
response = ai.generate_streaming([user_input])
# Output appears incrementally without stop tokens
```

## 🔮 Future Enhancements

- [x] XML tag parsing and role-based conversations
- [x] Stop token handling for clean output
- [x] Gemma3N chat template integration
- [ ] GGUF model conversion for CPU deployment
- [ ] Full multimodal fine-tuning with images/audio
- [ ] REST API server deployment
- [ ] Web interface with real-time chat
- [ ] Custom dataset integration
- [ ] Function calling expansion
- [ ] Multi-turn conversation memory

## 📝 License

This project combines:
- **Gemma3N**: Google's model license
- **Hermes Datasets**: Apache 2.0
- **Training Code**: MIT License

## 🙏 Acknowledgments

- **Google** for Gemma3N architecture
- **Nous Research** for Hermes datasets
- **Unsloth** for efficient training framework
- **HuggingFace** for model hosting

---

## 🚀 Getting Started

**Ready to chat with Senter-Omni? Here's how to begin:**

```bash
# Activate the virtual environment
source venv/bin/activate

# Start the advanced chat interface
python advanced_chat.py

# Try these XML examples:
<system>You are a helpful AI assistant.</system>
<user>Hello! Tell me about yourself.</user>
<user>I have this image: <image>A beautiful landscape</image> What do you see?</user>
```

### Quick Commands
- `help` - Learn about XML tags and commands
- `history` - View your conversation history
- `clear` - Start a fresh conversation
- `quit` - Exit the chat

**🎯 Senter-Omni combines the power of Gemma3N with clean XML-based conversations and automatic stop token handling for the ultimate AI chat experience!**