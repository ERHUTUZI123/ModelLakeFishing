import os
import pathlib

from .task import TaskType


def get_root_path_string() -> str:
    # resources/ lives in the original TransferGraph checkout, NOT in this
    # repo. Default assumes the checkout is a sibling of ModelLakeFishing
    # (../../codes/transfergraph); set TRANSFERGRAPH_ROOT to override.
    env = os.environ.get("TRANSFERGRAPH_ROOT")
    if env:
        return env
    return str(pathlib.Path(__file__).resolve().parents[4] / "transfergraph")


def get_directory_experiments(task_type: TaskType) -> str:
    return f"{get_root_path_string()}/resources/experiments/{task_type.value}"
