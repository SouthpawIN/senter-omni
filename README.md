# 🤖 Senter-Omni

**Uncensored Hermes-Trained AI Assistant with Multimodal Capabilities**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ Features

- **🧠 Advanced Reasoning** - Trained on Hermes-3-Dataset for superior problem-solving
- **🔧 Function Calling** - Enhanced tool use and API integration capabilities
- **🆓 Uncensored** - Direct, unrestricted responses without content filters
- **👁️ Multimodal Ready** - Vision and audio understanding architecture
- **⚡ Efficient** - LoRA fine-tuning for optimal performance
- **🎯 Hermes-Trained** - Based on high-quality conversational data

## 🚀 Quick Start

### 1. Interactive Chat
```bash
cd /home/sovthpaw/Desktop/senter-omni
source venv/bin/activate  # Activate virtual environment
python chat.py           # Start interactive chat
```

### 2. Quick Test
```bash
python chat.py --test    # Run automated test
```

## 📁 Project Structure

```
senter-omni/
├── chat.py                    # Interactive chat interface
├── run_senter_omni.py         # Alternative chat script
├── models/
│   ├── huggingface/          # HuggingFace models
│   │   ├── senter-omni-lora/     # LoRA adapter (recommended)
│   │   └── senter-omni-merged/   # Full merged model
│   └── gguf/                 # GGUF models (conversion pending)
├── data/                     # Training data
├── scripts/                  # Utility scripts
├── notebooks/                # Training notebooks
├── test_assets/              # Test images/audio
└── requirements.txt          # Python dependencies
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

### Interactive Chat
```bash
python chat.py
```

**Available Commands:**
- `help` - Show help information
- `clear` - Clear screen
- `info` - Show model information
- `quit` - Exit chat

### Example Conversations

**Math & Reasoning:**
```
You: Solve 15 × 23 + 7
Senter-Omni: Let me calculate that step by step...
```

**Code Generation:**
```
You: Write a Python function to check if a number is prime
Senter-Omni: Here's an efficient prime checking function...
```

**Uncensored Discussion:**
```
You: Discuss the ethics of AI development
Senter-Omni: AI ethics involves several key considerations...
```

## 🔧 Technical Details

### Base Model
- **Architecture**: Gemma3N 4B
- **Training**: Instruction-tuned
- **Multimodal**: Vision + Audio + Text

### Fine-tuning
- **Method**: LoRA (Low-Rank Adaptation)
- **Rank**: 16
- **Training Data**: 5,003 conversations
- **Sources**: Hermes-3-Dataset + Function Calling v1

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

The chat interface provides:
- **Real-time streaming** responses
- **Command system** for utilities
- **Error handling** and recovery
- **Clean interface** with emojis and formatting
- **Help system** with examples

## 🔮 Future Enhancements

- [ ] GGUF model conversion (CMake build issues resolved)
- [ ] Full multimodal fine-tuning with images/audio
- [ ] API server deployment
- [ ] Web interface
- [ ] Custom dataset integration

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

**🚀 Ready to chat with Senter-Omni? Run `python chat.py` and start exploring!**