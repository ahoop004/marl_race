#!/usr/bin/env python3
"""Racing experiment CLI."""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.is_dir():
    if str(SRC_DIR) in sys.path:
        sys.path.remove(str(SRC_DIR))
    sys.path.insert(0, str(SRC_DIR))

from application.cli import main

if __name__ == "__main__":
    main()
