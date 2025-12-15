import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
API_DIR = SRC_DIR / "api"

sys.path.insert(0, str(SRC_DIR))

def _import_if_exists(module: str, file_path: Path) -> None:
    if not file_path.exists():
        return
    importlib.import_module(module)

def test_import_api_modules():
    # Always required for your serving stack
    importlib.import_module("api.main")

    # Optional: only import if the file exists in this repo state
    _import_if_exists("api.train_model", API_DIR / "train_model.py")
    _import_if_exists("api.data_preprocessing", API_DIR / "data_preprocessing.py")
