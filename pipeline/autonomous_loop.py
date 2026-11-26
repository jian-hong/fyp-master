"""
Thin wrapper — the canonical campaign controller lives at repo root autonomous_loop.py
so watchdog / Task Scheduler may invoke either path interchangeably.

See: autonomous_loop.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    script = ROOT / "autonomous_loop.py"
    cmd = [sys.executable, str(script)] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
