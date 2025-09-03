#!/usr/bin/env python3
"""
Package setup for Senter-Omni Suite

Installs both senter-omni (chat) and senter-embed (embedding) packages.
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="senter-omni-suite",
    version="1.0.0",
    author="Sovthpaw",
    author_email="sovthpaw@github.com",
    description="Senter-Omni Suite: Multimodal Chat & Embedding Models with XML Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SouthpawIN/senter-omni",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "multimodal": [
            "librosa>=0.10.0",  # For audio processing
            "opencv-python>=4.5.0",  # For video processing
            "bitsandbytes>=0.40.0",  # For 4-bit quantization
        ],
        "all": [
            "librosa>=0.10.0",
            "opencv-python>=4.5.0",
            "bitsandbytes>=0.40.0",
            "scikit-learn>=1.0.0",  # For additional similarity metrics
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "senter-omni=senter_omni_cli:main",
            "senter-embed=senter_embed_cli:main",
            "senter-chat=senter_omni_cli:main",  # Alias for chat
            "senter-search=senter_embed_cli:main",  # Alias for embedding search
        ],
    },
    keywords=[
        "multimodal",
        "ai",
        "chat",
        "embedding",
        "similarity",
        "search",
        "gemma3n",
        "xml",
        "nlp",
        "computer-vision",
        "audio-processing",
        "video-processing"
    ],
    project_urls={
        "Bug Reports": "https://github.com/SouthpawIN/senter-omni/issues",
        "Source": "https://github.com/SouthpawIN/senter-omni",
        "Documentation": "https://github.com/SouthpawIN/senter-omni#readme",
    },
    include_package_data=True,
    zip_safe=False,
)
