import os
import pathlib

from .task import TaskType

# all our data now lives in stage1BuildTransferGraph/dataset_embed/data/
# completely self-contained, no external dependencies
# __file__ = .../stage1BuildTransferGraph/dataset_embed/utils/config.py
# parents[1] = stage1BuildTransferGraph/dataset_embed
_DATA_DIR = os.path.join(
    str(pathlib.Path(__file__).resolve().parents[1]),
    'data'
)


def get_root_path_string() -> str:
    # return stage1BuildTransferGraph/ (used by code that expects
    # resources/experiments/<task_type>/ to live under this root)
    return str(pathlib.Path(__file__).resolve().parents[2].parent)


def get_directory_experiments(task_type: TaskType) -> str:
    # return our self-contained data directory (task_type parameter kept
    # for API compat with the original codebase)
    return _DATA_DIR
