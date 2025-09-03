"""
Senter-Embed: Multimodal Embedding Model for Similarity Search

A comprehensive multimodal embedding system for text, images, audio, and video
with advanced similarity search capabilities.
"""

from .core import SenterEmbedder
from .database import MultimodalEmbeddingDatabase
from .utils import cosine_similarity, compute_similarity_matrix

__version__ = "1.0.0"
__author__ = "Sovthpaw"
__description__ = "Multimodal embedding system for similarity search"

__all__ = [
    "SenterEmbedder",
    "MultimodalEmbeddingDatabase",
    "cosine_similarity",
    "compute_similarity_matrix",
    "__version__",
    "__author__",
    "__description__"
]
