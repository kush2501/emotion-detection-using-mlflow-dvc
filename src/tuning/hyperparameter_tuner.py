import logging
import optuna
import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_training_data():
    """
    Load training features for hyperparameter tuning.
    """

    logger.info("Loading training data for hyperparameter tuning...")

    train_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "train_bow.csv"
    )

    train_data = pd.read_csv(train_path)

    X_train = train_data.drop(columns=["label"])
    y_train = train_data["label"]

    logger.info(f"X_train Shape : {X_train.shape}")
    logger.info(f"y_train Shape : {y_train.shape}")

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

    # Create model
    model = LogisticRegression(
        C=C,
        solver=solver,
        penalty="l2",
        max_iter=1000,
        random_state=42
    )

    # Stratified Cross Validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Evaluate trial
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    mean_f1 = scores.mean()

    logger.info(
        f"Trial {trial.number} | "
        f"C={C:.5f} | "
        f"solver={solver} | "
        f"Mean F1={mean_f1:.4f}"
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