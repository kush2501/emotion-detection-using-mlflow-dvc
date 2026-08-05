import logging
import pandas as pd

from sklearn.pipeline import Pipeline
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

N_SPLITS = 5
SCORING_METRIC = "f1"
RANDOM_STATE = 42

FEATURE_CONFIGS = [
    {
        "name": "tfidf_2000_unigram",
        "max_features": 2000,
        "ngram_range": (1, 1)
    },
    {
        "name": "tfidf_5000_unigram",
        "max_features": 5000,
        "ngram_range": (1, 1)
    },
    {
        "name": "tfidf_5000_unigram_bigram",
        "max_features": 5000,
        "ngram_range": (1, 2)
    },
    {
        "name": "tfidf_7500_unigram",
        "max_features": 7500,
        "ngram_range": (1, 1)
    },
    {
        "name": "tfidf_7500_unigram_bigram",
        "max_features": 7500,
        "ngram_range": (1, 2)
    },
    {
        "name": "tfidf_10000_unigram",
        "max_features": 10000,
        "ngram_range": (1, 1)
    },
    {
        "name": "tfidf_10000_unigram_bigram",
        "max_features": 10000,
        "ngram_range": (1, 2)
    }
]


def load_processed_training_data():
    """
    Load processed training text and target labels
    for feature tuning.
    """

    logger.info("Loading processed training data...")

    train_path = (
        BASE_DIR
        / "data"
        / "interim"
        / "train_processed.csv"
    )

    train_data = pd.read_csv(train_path)

    X_text = train_data["content"]
    y_train = train_data["sentiment"]

    logger.info(f"Training Samples : {len(X_text)}")
    logger.info(f"Target Shape     : {y_train.shape}")

    return X_text, y_train  


def evaluate_feature_config(
        config,
        X_text,
        y_train
    ):
        """
        Evaluate one TF-IDF feature configuration
        using Logistic Regression and cross-validation.
        """

        logger.info("=" * 60)
        logger.info(
            f"Evaluating Feature Config : {config['name']}"
        )


        pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=config["max_features"],
                ngram_range=config["ngram_range"]
            )
        ),
        (
            "model",
            LogisticRegression(
                C=1.0333143670701204,
                solver="liblinear",
                penalty="l2",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        scores = cross_val_score(
            pipeline,
            X_text,
            y_train,
            cv=cv,
            scoring=SCORING_METRIC,
            n_jobs=-1
        )

        mean_f1 = scores.mean()
        std_f1 = scores.std()

        logger.info(f"Mean CV F1 : {mean_f1:.4f}")
        logger.info(f"Std CV F1  : {std_f1:.4f}")

        return {
            "config_name": config["name"],
            "max_features": config["max_features"],
            "ngram_range": config["ngram_range"],
            "mean_f1": mean_f1,
            "std_f1": std_f1
        }

def run_feature_experiments(X_text, y_train):
    """
    Evaluate all feature configurations and
    identify the best configuration.
    """

    logger.info("=" * 60)
    logger.info("Starting Feature Configuration Experiments")
    logger.info(f"Total Configurations : {len(FEATURE_CONFIGS)}")

    results = []

    for config in FEATURE_CONFIGS:

        result = evaluate_feature_config(
            config,
            X_text,
            y_train
        )

        results.append(result)

    best_result = max(
        results,
        key=lambda result: result["mean_f1"]
    )

    logger.info("=" * 60)
    logger.info("Feature Experiments Completed")
    logger.info(
        f"Best Configuration : {best_result['config_name']}"
    )
    logger.info(
        f"Best Mean CV F1    : {best_result['mean_f1']:.4f}"
    )
    logger.info(
        f"Best Std CV F1     : {best_result['std_f1']:.4f}"
    )
    logger.info("=" * 60)

    return results, best_result


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    X_text, y_train = load_processed_training_data()

    results, best_result = run_feature_experiments(
        X_text,
        y_train
    )   