from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root_path() -> Path:
    configured = os.environ.get("MLF_DATA_DIR")
    root = Path(configured).expanduser() if configured else REPO_ROOT / "data"
    return root.resolve()


def data_root() -> str:
    return os.fspath(data_root_path())


def runs_root_path() -> Path:
    configured = os.environ.get("MLF_RUNS_DIR")
    root = Path(configured).expanduser() if configured else REPO_ROOT / "runs"
    return root.resolve()


def runs_root() -> str:
    return os.fspath(runs_root_path())
