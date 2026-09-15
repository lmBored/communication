#!/usr/bin/env python3
"""
Source-checkout wrapper around the packaged trainer (``escape-room-train``).

Usage:
    python scripts/train.py --num-envs 4096 --num-updates 50 --ckpt-dir ./ckpts
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from escape_room.train import main  # noqa: E402


if __name__ == "__main__":
    main()
