#!/usr/bin/env python3
"""Compatibility wrapper for the unified serving E2E tool.

The implementation now lives in ``tests/tools/serving_endpoints.py`` so that
all user-facing test and benchmark entry points are under ``tests/``.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(repo_root / "tests" / "tools" / "serving_endpoints.py"), run_name="__main__")
