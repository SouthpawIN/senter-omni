#!/usr/bin/env python3
"""
Senter-Omni CLI Entry Point

Command-line interface for the Senter-Omni chat model.
"""

import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from senter_omni.cli import main

if __name__ == "__main__":
    main()
