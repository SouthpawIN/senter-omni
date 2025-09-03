# 🤖 Senter-Omni

**Advanced Multimodal AI Assistant with XML Tag Support**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ Features

- **🧠 Advanced Reasoning** - Fine-tuned on Hermes-3-Dataset for superior problem-solving
- **🔧 Function Calling** - Enhanced tool use and API integration capabilities
- **🆓 Uncensored** - Direct, unrestricted responses without content filters
- **👁️ Multimodal Ready** - Vision and audio understanding architecture
- **⚡ Efficient** - LoRA fine-tuning for optimal performance
- **🎯 XML Tag Support** - Clean role-based conversation formatting
- **🛑 Stop Token Handling** - Automatic conversation termination
- **📝 Gemma3N Integration** - Native support for Gemma3N chat format

## 🚀 Quick Start

### 1. Interactive Chat
```bash
cd /home/sovthpaw/Desktop/senter-omni
source venv/bin/activate  # Activate virtual environment
python advanced_chat.py  # Start advanced interactive chat
```

### 2. XML Tag Examples
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
senter-omni/
├── advanced_chat.py          # Main advanced chat interface with XML support
├── senter_omni_utils.py      # Utility functions and classes
├── models/
│   ├── huggingface/          # HuggingFace models
│   │   ├── senter-omni-lora/     # LoRA adapter (recommended)
│   │   │   └── chat_template.jinja  # Gemma3N chat format
│   │   └── senter-omni-merged/   # Full merged model
│   └── gguf/                 # GGUF models (future)
├── scripts/                  # Utility scripts
│   ├── convert_to_gguf.py    # Model conversion utilities
│   └── prepare_hermes_data.py # Data preparation
├── notebooks/                # Training and development notebooks
├── test_assets/              # Test files (images, audio)
├── data/                     # Training datasets
├── requirements.txt          # Python dependencies
└── venv/                     # Virtual environment (auto-created)
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

### Interactive Chat with XML Support
```bash
python advanced_chat.py
```

**Available Commands:**
- `help` - Show help information and XML examples
- `clear` - Clear conversation history
- `history` - View conversation history
- `video` - Check video support capabilities
- `quit` - Exit chat

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