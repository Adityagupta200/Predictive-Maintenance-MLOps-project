# tests/test_imports.py
def test_import_training_and_api():
    # Import modules just to ensure they load in CI
    __import__("src.train_model")
    __import__("src.data_preprocessing")
    __import__("api.main")
