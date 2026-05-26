#!/usr/bin/env python3
"""HotpotQA workspace helpers."""

from pathlib import Path
import sys

SHARED_DIR = Path(__file__).resolve().parents[2] / "_shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from evolution_workspace import *  # noqa: F401,F403
