"""Ensure `src/` is importable for tests even when the editable install is not
picked up (e.g. uv on macOS sometimes sets UF_HIDDEN on .pth files, which
Python 3.12 silently skips)."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
