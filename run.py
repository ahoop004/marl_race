#!/usr/bin/env python3
"""Racing experiment CLI."""
import sys
import os
import math
from pathlib import Path

# TorchRL binds this timeout while importing its shared-weight transport.
# Keep its default when unset; reject overrides that disable bounded waiting.
if "WEIGHT_SYNC_TIMEOUT" in os.environ:
    try:
        weight_sync_timeout = float(os.environ["WEIGHT_SYNC_TIMEOUT"])
    except ValueError:
        raise SystemExit("WEIGHT_SYNC_TIMEOUT must be a positive finite number of seconds")
    if not math.isfinite(weight_sync_timeout) or weight_sync_timeout <= 0:
        raise SystemExit("WEIGHT_SYNC_TIMEOUT must be a positive finite number of seconds")

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.is_dir():
    if str(SRC_DIR) in sys.path:
        sys.path.remove(str(SRC_DIR))
    sys.path.insert(0, str(SRC_DIR))

from application.cli import main

if __name__ == "__main__":
    main()
