#!/usr/bin/env python3
"""
Source-checkout wrapper around the packaged viewer (``escape-room-play``).

Usage:
    python scripts/play.py
    python scripts/play.py --ckpt ./ckpts/model_49.pt
    python scripts/play.py --headless --record demos/escape_demo.mp4 --steps 400
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from escape_room.play import main  # noqa: E402


if __name__ == "__main__":
    main()
