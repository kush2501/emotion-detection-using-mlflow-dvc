import pickle
import nltk
import logging
from pathlib import Path

from src.preprocessing.text_preprocessor import preprocess_text
from src.config.config_loader import (
    load_model_config,
    load_feature_config
)

logger = logging.getLogger(__name__)

# -------------------- Project Root -------------------- #

BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------- Load Model -------------------- #

MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

logger.info("Model loaded successfully")
logger.info(
    "Expected model features: %s",
    model.n_features_in_)

# -------------------- Load Vectorizer -------------------- #

VECTORIZER_PATH = BASE_DIR / "artifacts" / "vectorizer.pkl"

with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)

logger.info("Vectorizer loaded successfully")
logger.info(
    "Vectorizer vocabulary size: %s",
    len(vectorizer.vocabulary_)
)

# -------------------- Predict -------------------- #

def predict_sentiment(text: str):

    # -------------------- Preprocessing -------------------- #

    clean_text = preprocess_text(text)

    logger.debug(
    "Text preprocessing completed. Processed length=%d",
    len(clean_text)
)

    # -------------------- Vectorization -------------------- #

    text_vector = vectorizer.transform([clean_text])

    # -------------------- Prediction -------------------- #

    prediction = model.predict(text_vector)[0]

    probability = model.predict_proba(text_vector)[0]

    confidence = round(max(probability) * 100, 2)

    sentiment = "😊 Positive" if prediction == 1 else "😔 Negative"

    return {
        "prediction": sentiment,
        "confidence": confidence,
        "processed_text": clean_text
    }


def get_model_status():
    """
    Return health information for the inference pipeline.
    """

    from nltk.corpus import stopwords, wordnet

    # -------------------- NLTK Health -------------------- #

    nltk_status = {
        "stopwords": False,
        "wordnet": False
    }

    try:
        stopwords.words("english")
        nltk_status["stopwords"] = True
    except LookupError:
        pass

    try:
        wordnet.synsets("test")
        nltk_status["wordnet"] = True
    except LookupError:
        pass

    # -------------------- Component Health -------------------- #

    model_loaded = model is not None
    vectorizer_loaded = vectorizer is not None
    nltk_ready = all(nltk_status.values())

    healthy = (
        model_loaded
        and vectorizer_loaded
        and nltk_ready
    )

    return {
        "status": "healthy" if healthy else "unhealthy",
        "application": "Emotion Detection API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "vectorizer_loaded": vectorizer_loaded,
        "nltk_resources": nltk_status,
        "model_features": model.n_features_in_,
        "vocabulary_size": len(vectorizer.vocabulary_)
    }

def get_model_info():
    """
    Return information about the currently configured
    machine learning model and feature engineering pipeline.
    """

    model_config = load_model_config()
    feature_config = load_feature_config()

    algorithm = model_config["algorithm"]
    algorithm_params = model_config[algorithm]

    return {
        "status": "ready",
        "model": {
            "algorithm": algorithm,
            "parameters": algorithm_params
        },
        "feature_engineering": {
            "vectorizer": feature_config["vectorizer"],
            "max_features": feature_config["max_features"],
            "ngram_range": feature_config["ngram_range"]
        }
    }