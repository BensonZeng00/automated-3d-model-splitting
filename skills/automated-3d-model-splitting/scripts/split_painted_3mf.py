#!/usr/bin/env python3
"""Compatibility CLI for the object-oriented split3mf package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from split3mf.cli import main


if __name__ == "__main__":
    main()
