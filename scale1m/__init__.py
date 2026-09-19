"""Current 3M-scale ModelLakeFishing pipeline.

Supported flow:
    crawl/canonicalize -> build_ladder_rf -> embed_lake_rf -> build_graph_rf
    -> train_rung -> export_rf -> eval_x6/eval_y2/eval_y4

The evaluated serving configuration is dense HNSW top-1,000 followed by the
split-safe deterministic task-evidence reranker in :mod:`eval_y2`. Superseded
rung builders and experimental evaluators are archived under ``legacy/``.
"""

# Historical modules use the original repository name as their package prefix.
# Keep those imports working when a reviewer clones into a differently named
# directory; no installation, author checkout, or parent PYTHONPATH is needed.
import sys as _sys
import types as _types
from importlib.machinery import ModuleSpec as _ModuleSpec
from pathlib import Path as _Path

_repo = _Path(__file__).resolve().parents[1]
# Shared feature helpers historically import ``dataset_embed`` as a top-level
# package. Resolve that location from this checkout, never its directory name.
_feature_helpers = str(_repo / "stage1BuildTransferGraph")
if _feature_helpers not in _sys.path:
    _sys.path.insert(0, _feature_helpers)

if "ModelLakeFishing" not in _sys.modules:
    _package = _types.ModuleType("ModelLakeFishing")
    _package.__path__ = [str(_repo)]
    _package.__package__ = "ModelLakeFishing"
    _package.__spec__ = _ModuleSpec("ModelLakeFishing", loader=None, is_package=True)
    _sys.modules["ModelLakeFishing"] = _package
if __name__ == "scale1m":
    _sys.modules.setdefault("ModelLakeFishing.scale1m", _sys.modules[__name__])
    # A sys.modules alias alone does not attach the child to its parent.
    # Attribute-based imports and tools must see the same package too.
    _sys.modules["ModelLakeFishing"].scale1m = _sys.modules["ModelLakeFishing.scale1m"]
