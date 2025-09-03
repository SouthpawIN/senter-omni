"""
Senter-Omni: Advanced Multimodal Chat Model

A multimodal conversational AI built on Gemma3N with XML tag support,
real-time streaming, and advanced chat capabilities.
"""

from .core import SenterOmniChat
from .embedder import MultimodalEmbedder

__version__ = "1.0.0"
__author__ = "Sovthpaw"
__description__ = "Advanced multimodal chat AI with XML tag support"

__all__ = [
    "SenterOmniChat",
    "MultimodalEmbedder",
    "__version__",
    "__author__",
    "__description__"
]
