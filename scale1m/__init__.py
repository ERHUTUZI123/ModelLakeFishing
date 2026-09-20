import sys as _sys
import types as _types
from importlib.machinery import ModuleSpec as _ModuleSpec
from pathlib import Path as _Path

_repo = _Path(__file__).resolve().parents[1]
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
    _sys.modules["ModelLakeFishing"].scale1m = _sys.modules["ModelLakeFishing.scale1m"]
