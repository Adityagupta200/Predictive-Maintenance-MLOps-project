import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

def test_import_training_and_api():
    importlib.import_module("api.main")
    importlib.import_module("api.train_model")
    importlib.import_module("api.data_preprocessing")
