#!/usr/bin/env python3
"""
Senter-Embed CLI Entry Point

Command-line interface for the Senter-Embed multimodal embedding system.
"""

import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from senter_embed.cli import main

if __name__ == "__main__":
    main()
