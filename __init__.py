#!/usr/bin/env python3
"""
🎭 Senter-Omni Unified API

Simple, powerful interface for multimodal AI operations.
"""

from .omni import (
    chat,
    embed,
    create_chat_completion,
    generate,
    get_omni_client,
    example_chat,
    example_embed
)

# Version info
__version__ = "1.0.0"
__author__ = "Senter-Omni Team"

# Module-level convenience functions
def ChatCompletion(**kwargs):
    """OpenAI-style chat completion"""
    return create_chat_completion(**kwargs)

def Embed(**kwargs):
    """Multimodal embedding with XML tags"""
    return embed(**kwargs)

# Auto-initialize on import
_omni_client = None

def _get_client():
    """Lazy initialization of client"""
    global _omni_client
    if _omni_client is None:
        from .omni import get_omni_client
        _omni_client = get_omni_client()
    return _omni_client

# Expose main functions at module level
def chat(*args, **kwargs):
    """Generate chat completions"""
    return _get_client().chat(*args, **kwargs)

def embed(*args, **kwargs):
    """Process multimodal embeddings"""
    return _get_client().embed(*args, **kwargs)

def create_chat_completion(*args, **kwargs):
    """OpenAI-style chat completion"""
    return _get_client().chat(*args, **kwargs)

def generate(*args, **kwargs):
    """llama.cpp-style generation"""
    return _get_client().chat(*args, **kwargs)
