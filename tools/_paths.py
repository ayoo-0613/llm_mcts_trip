from __future__ import annotations

import os
from typing import Optional


def repo_root() -> str:
    # tools/_paths.py -> tools/ -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def database_path(*parts: str) -> str:
    return os.path.join(repo_root(), "database", *parts)


def maybe_path(path: Optional[str], *default_parts: str) -> str:
    return path if path else database_path(*default_parts)

