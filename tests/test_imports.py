# tests/test_imports.py
import sys
from pathlib import Path
import importlib

# Add <repo>/src to sys.path so 'api.*' can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_import_training_and_api():
    # Adjust these names if your filenames differ
    assert importlib.import_module("api.train_model.py")
    assert importlib.import_module("api.data_preprocessing.py")
