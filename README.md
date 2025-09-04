<div align="center">

<video width="100%" controls playsinline preload="metadata">
  <source src="https://media.githubusercontent.com/media/SouthpawIN/senter-omni/main/senter-animated-banner.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</div>

## 🌟 **Why Senter-Omni?**

**🎯 ONE MODEL, ALL MODALITIES** - Unlike pipeline approaches, Senter-Omni is a single 4B parameter model that understands and reasons across text, images, audio, and video simultaneously.

**🔓 OPEN & UNCENSORED** - Apache 2.0 licensed with unrestricted responses for maximum utility.

**128K CONTEXT** - Extended RoPE scaling for handling massive documents and conversations.

**MEMORY EFFICIENT** - 4-bit quantized model that fits on consumer GPUs while maintaining full multimodal capabilities.

---

## 🚀 **Quick Start**

### **🤗 Hugging Face Repositories**
- **Model**: https://huggingface.co/SouthpawIN/senter-omni-model
- **Dataset**: https://huggingface.co/datasets/SouthpawIN/senter-omni-data

### **Installation**
```bash
git clone https://github.com/SouthpawIN/senter-omni.git
cd senter-omni
pip install -r requirements.txt

# Download the quantized model (instructions below)
# Then run the demo:
python senter_omni_demo.py
```

### **Basic Usage**
```python
from omni import OmniClient

# Initialize Senter-Omni
client = OmniClient()

# Streaming chat
response = client.chat([
    {"role": "user", "content": "Hello Senter!"}
], stream=True)

# Multimodal chat with image
response = client.chat([
    {"role": "user", "content": [
        {"type": "image", "image": "photo.jpg"},
        {"type": "text", "text": "What do you see?"}
    ]}
])

# Cross-modal embeddings
embedding = client.embed("any content", modality="auto")
```

---

## 🎭 **Multimodal Capabilities**

### **Text Understanding & Generation**
- **Mathematical Reasoning**: Step-by-step problem solving
- **Code Generation**: Python, JavaScript, and more
- **Creative Writing**: Stories, scripts, poetry
- **Technical Analysis**: Complex explanations and documentation

### **Visual Understanding**
- **Image Analysis**: Detailed descriptions of visual content
- **Geometric Recognition**: Shapes, colors, spatial relationships
- **Creative Interpretation**: Stories inspired by images
- **Technical Diagrams**: Understanding charts, graphs, schematics

### **Audio Processing**
- **Sound Analysis**: Identifying audio content and patterns
- **Speech Understanding**: Transcribing and interpreting spoken content
- **Music Analysis**: Recognizing musical elements and genres
- **Environmental Audio**: Identifying sounds from various sources

### **Cross-Modal Reasoning**
- **Unified Understanding**: Connecting information across modalities
- **Contextual Analysis**: Using multiple inputs for better reasoning
- **Creative Synthesis**: Combining visual, audio, and text for rich responses

### **Model Specifications**
- **Parameters**: 4B (quantized to 4-bit)
- **Context Length**: 128K tokens (RoPE scaled)
- **Memory Usage**: ~8GB VRAM
- **Inference Speed**: Real-time streaming
- **Modalities**: Text, Image, Audio, Video

### **Embedding Capabilities**
- **Unified Space**: 1024D embeddings for all modalities
- **Cross-Modal Search**: Find similar content across text, images, audio
- **Similarity Matching**: Cosine similarity in unified space
- **Memory Efficient**: Same model for chat and embeddings

---

## 🎯 **Real Examples**

### **Image Analysis**
```python
# Analyze geometric shapes
response = client.chat([
    {"role": "user", "content": [
        {"type": "image", "image": "test_assets/real_test_image.jpg"},
        {"type": "text", "text": "What geometric shapes do you see?"}
    ]}
])

# Output: "I see a red square, blue square, and green oval arranged vertically"
```

### **Audio Understanding**
```python
# Process audio content
response = client.chat([
    {"role": "user", "content": [
        {"type": "audio", "audio": "test_assets/real_test_audio.wav"},
        {"type": "text", "text": "What do you hear?"}
    ]}
])

# Output: "I hear an electric hum from a device like a radio or TV"
```

### **Creative Multimodal Storytelling**
```python
# Create stories from images
response = client.chat([
    {"role": "user", "content": [
        {"type": "image", "image": "shapes.jpg"},
        {"type": "text", "text": "Create a story inspired by this image"}
    ]}
])

# Output: Rich, creative stories combining visual elements with narrative
```

### **Cross-Modal Embeddings**
```python
# Embed different modalities
text_emb = client.embed("beautiful mountain landscape")
image_emb = client.embed("mountain_photo.jpg", modality="image")
audio_emb = client.embed("nature_sounds.wav", modality="audio")

# All embeddings are in the same 1024D space for comparison
```

---

## 🔧 **Technical Architecture**

### **Model Details**
- **Base**: Qwen2.5-Omni-3B (Apache 2.0 licensed)
- **Quantization**: 4-bit NF4 for memory efficiency
- **Context Extension**: Yarn RoPE scaling to 128K
- **Streaming**: Custom TimingStreamer for real-time output
- **Embeddings**: Hash-based unified 1024D space

### **Training Data**
- **131,893 samples** from multiple high-quality datasets:
  - 50,000 ShareGPT conversations (chat)
  - 30,000 AgentCode samples (function calling)
  - 20,000 Stack Overflow (coding)
  - 30,000 Hermes-3 (instruction tuning)
  - 1,893 Hermes function calling

### **Key Features**
- **XML Tag Support**: `<think>`, `<notepad>`, `<system>`, `<user>`, `<assistant>`
- **Uncensored Responses**: No content restrictions
- **Function Calling**: Tool integration capabilities
- **Memory Efficient**: Single model for chat and embeddings

---

## 📦 **Installation & Setup**

### **1. Clone Repository**
```bash
git clone https://github.com/SouthpawIN/senter-omni.git
cd senter-omni
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Download Model**
The quantized model (3.5GB) is hosted on Hugging Face due to GitHub's 100MB file limit:

**🤗 Model Hosting:**
- **Model**: https://huggingface.co/SouthpawIN/senter-omni-model
- **Dataset**: https://huggingface.co/datasets/SouthpawIN/senter-omni-data

```bash
# Option 1: Download from Hugging Face (Recommended)
git lfs install
git clone https://huggingface.co/SouthpawIN/senter-omni-model
cp -r senter-omni-model/* ./senter_omni_128k/

<<<<<<< HEAD
# Option 2: Use base model (will download automatically)
# The system will fall back to unsloth/Qwen2.5-Omni-3B

# Option 3: Manual download
# Model: https://huggingface.co/SouthpawIN/senter-omni-model
# Dataset: https://huggingface.co/datasets/SouthpawIN/senter-omni-data
=======
# Option 2: Manual download
# Download from: https://huggingface.co/SouthpawIN/senter-omni-model
>>>>>>> ef9c90b68b12894eb877dacd7f741ec1228cffb6
```

## 🎮 **Interactive Demo**

The comprehensive demo showcases all capabilities:

```bash
python senter_omni_demo.py
```

**Demo Sections:**
1. **🎓 Training Capabilities** - Dataset overview and training features
2. **💬 Multimodal Chat** - Text, image, audio, and combined processing
3. **🔍 Cross-Modal Embeddings** - Unified embedding space demonstration
4. **🚀 Building Guide** - API usage and integration examples

---

## 🛠️ **API Reference**

### **Core Methods**

#### **`client.chat(messages, **kwargs)`**
```python
# Basic chat
response = client.chat([
    {"role": "user", "content": "Hello!"}
])

# With parameters
response = client.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
    temperature=0.7,
    stream=True
)

# Multimodal
response = client.chat([
    {"role": "user", "content": [
        {"type": "image", "image": "photo.jpg"},
        {"type": "text", "text": "Describe this image"}
    ]}
])
```

#### **`client.embed(content, modality="auto")`**
```python
# Text embedding
emb = client.embed("sample text")

# Image embedding
emb = client.embed("image.jpg", modality="image")

# Audio embedding
emb = client.embed("audio.wav", modality="audio")

# Auto-detect modality
emb = client.embed("[IMAGE] photo.jpg")  # Detects as image
```

#### **`client.cross_search(query, top_k=5)`**
```python
# Search across modalities
results = client.cross_search("mountain landscape")
# Returns: {"text": [...], "image": [...], "audio": [...]}
```

#### **`client.retrieve_context(query, context_window=5)`**
```python
# Get relevant context
context = client.retrieve_context("nature scenes")
# Returns multimodal context items
```

---

### **Memory Usage**
- **Model Loading**: ~8GB VRAM
- **Inference**: ~10GB VRAM peak
- **Embeddings**: Shared model (no additional memory)
- **Context (128K)**: ~2GB additional for full context

### **Development Setup**
```bash
git clone https://github.com/SouthpawIN/senter-omni.git
cd senter-omni
pip install -r requirements.txt
python senter_omni_demo.py  # Test installation
```

---

## 📄 **License**

**Apache 2.0 License** - See [LICENSE](LICENSE) for details.

This project uses:
- **Qwen2.5-Omni**: Apache 2.0 (Alibaba Cloud)
- **Training Datasets**: Various open licenses
- **Code**: Apache 2.0

---

## 🙏 **Acknowledgments**

- **Alibaba Cloud** for Qwen2.5-Omni architecture
- **Nous Research** for Hermes dataset and inspiration
- **Alignment Lab AI** for development and training
- **Unsloth** for efficient training framework
- **HuggingFace** for model hosting and tools
- **Open Source Community** for datasets and tools

---

<div align="center">

**🎭 EXPERIENCE THE FUTURE OF MULTIMODAL AI WITH SENTER-OMNI**

*Built with ❤️ by Chris at Alignment Lab AI*
Donations:
https://www.paypal.me/Sellgames1l
</div>
