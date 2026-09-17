"""Portable filesystem locations shared by the 3M reproduction pipeline.

The original research runs used an author-local Windows path.  Reproduction
must not inherit that machine-specific default: unless ``MLF_DATA_DIR`` is set,
all generated and downloaded data now lives in ``<repository>/data``.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root_path() -> Path:
    """Return the configured data root as an absolute path."""
    configured = os.environ.get("MLF_DATA_DIR")
    root = Path(configured).expanduser() if configured else REPO_ROOT / "data"
    return root.resolve()


def data_root() -> str:
    """String form retained for the existing ``os.path`` based modules."""
    return os.fspath(data_root_path())


def runs_root_path() -> Path:
    """Return the run root, defaulting to ``<repository>/runs``."""
    configured = os.environ.get("MLF_RUNS_DIR")
    root = Path(configured).expanduser() if configured else REPO_ROOT / "runs"
    return root.resolve()


def runs_root() -> str:
    return os.fspath(runs_root_path())
