from importlib import util
from pathlib import Path
import sys

_LEGACY_PATH = Path(__file__).resolve().parents[1] / "models.py"
_SPEC = util.spec_from_file_location("_ducor_legacy_models", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load legacy models module from {_LEGACY_PATH}")

_MODULE = util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

VQAmodel = _MODULE.VQAmodel

__all__ = ["VQAmodel"]
