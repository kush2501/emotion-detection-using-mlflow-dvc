import logging
import optuna
import pandas as pd

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

N_SPLITS = 5
SCORING_METRIC = "f1"
RANDOM_STATE = 42


def load_training_data():
    """
    Load training features for hyperparameter tuning.
    """

    logger.info("Loading training data for hyperparameter tuning...")

    train_path = (
        BASE_DIR
        / "data"
        / "interim"
        / "train_processed.csv"
    )

    train_data = pd.read_csv(train_path)

    train_data = train_data.dropna(
        subset=["content", "sentiment"]
    )

    X_train = train_data["content"]
    y_train = train_data["sentiment"]

    logger.info(f"Training Samples : {len(X_train)}")
    logger.info(f"Target Shape     : {y_train.shape}")
    return X_train, y_train

def objective(trial, X_train, y_train):
    """
    Optuna objective function for Logistic Regression.

    The function searches for hyperparameters that maximize
    cross-validation F1 score.
    """

    # Hyperparameters suggested by Optuna
    C = trial.suggest_float(
        "C",
        0.001,
        100.0,
        log=True
    )

    solver = trial.suggest_categorical(
        "solver",
        ["liblinear", "lbfgs"]
    )


    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 1),
                stop_words="english"
            )
        ),
        (
            "model",
            LogisticRegression(
                C=C,
                solver=solver,
                penalty="l2",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])

    # Stratified Cross Validation
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Evaluate trial
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING_METRIC,
        n_jobs=-1
    )

    mean_f1 = scores.mean()
    std_f1 = scores.std()

    logger.info(
        f"Trial {trial.number} | "
        f"C={C:.5f} | "
        f"solver={solver} | "
        f"Mean F1={mean_f1:.4f} | "
        f"Std F1={std_f1:.4f}"  
    )

    return mean_f1

def run_tuning(X_train, y_train, n_trials=20):
    """
    Run Optuna hyperparameter optimization for Logistic Regression.
    """

    logger.info("=" * 60)
    logger.info("Starting Logistic Regression Hyperparameter Tuning")
    logger.info(f"Number of Trials : {n_trials}")

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train
        ),
        n_trials=n_trials
    )

    logger.info("=" * 60)
    logger.info("Hyperparameter Tuning Completed")
    logger.info(f"Best CV F1 Score : {study.best_value:.4f}")
    logger.info(f"Best Parameters   : {study.best_params}")
    logger.info("=" * 60)

    return study



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    X_train, y_train = load_training_data()

    study = run_tuning(
        X_train,
        y_train,
        n_trials=20
    )