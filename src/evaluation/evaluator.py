import logging
import pandas as pd

from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Logging.
logger = logging.getLogger(__name__)


# -------------------- Evaluate Model -------------------- #

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """

    try:

        logger.info("Evaluating model...")

        y_pred = model.predict(X_test)

        # Signature.
        signature_input = pd.DataFrame(X_test).astype("float64")

        signature = infer_signature(
            signature_input,
            y_pred
        )

        metrics = {

            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred))

        }

        logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
        logger.info(f"Precision : {metrics['precision']:.4f}")
        logger.info(f"Recall    : {metrics['recall']:.4f}")
        logger.info(f"F1 Score  : {metrics['f1_score']:.4f}")

        return metrics, signature

    except Exception:

        logger.exception("Error during model evaluation.")
        raise
