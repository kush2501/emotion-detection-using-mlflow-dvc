import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "artifacts" / "vectorizer.pkl"


def test_model_file_exists():
    """Check that trained model artifact exists."""
    assert MODEL_PATH.exists()


def test_vectorizer_file_exists():
    """Check that fitted vectorizer artifact exists."""
    assert VECTORIZER_PATH.exists()


def test_model_loading():
    """Check that trained model loads successfully."""
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    assert model is not None
    assert hasattr(model, "predict")


def test_vectorizer_loading():
    """Check that fitted vectorizer loads successfully."""
    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    assert vectorizer is not None
    assert hasattr(vectorizer, "transform")


def test_model_vectorizer_compatibility():
    """Model and vectorizer must use the same feature dimension."""
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    assert model.n_features_in_ == len(vectorizer.vocabulary_)