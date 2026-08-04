import os
import json
import logging

import mlflow
import dagshub

from mlflow import MlflowClient
from pathlib import Path

from dotenv import load_dotenv
# -------------------- Logger -------------------- #

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "project.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# -------------------- Load Environment Variables -------------------- #

load_dotenv(BASE_DIR / ".env")
# -------------------- Load Run Information -------------------- #

def load_run_info():
    """
    Load Run ID and Model Name from artifacts/run_info.json
    """

    try:

        logger.info("Loading Run Information...")

        with open("./artifacts/run_info.json", "r") as file:
            run_info = json.load(file)

        logger.info("Run Information Loaded Successfully.")

        return run_info

    except Exception:

        logger.exception("Failed to load run_info.json")
        raise


# -------------------- Register Model -------------------- #

def register_model(run_info):
    """
    Register the evaluated MLflow model
    and return its registered version.
    """

    try:

        run_id = run_info["run_id"]

        registered_model_name = "emotion-detection-model"
        artifact_path = "model"

        model_uri = (
            f"runs:/{run_id}/{artifact_path}"
        )

        logger.info("=" * 60)
        logger.info("Registering Champion Candidate")

        logger.info(f"Run ID           : {run_id}")
        logger.info(f"Model URI        : {model_uri}")
        logger.info(
            f"Registered Model : {registered_model_name}"
        )

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )

        version = registered_model.version

        logger.info("Model Registered Successfully")
        logger.info(f"Version          : {version}")
        logger.info("=" * 60)

        return registered_model_name, version

    except Exception:

        logger.exception(
            "Model Registration Failed"
        )
        raise

def promote_best_model(model_name, new_version):
    """
    Compare the newly registered model with the current champion
    using F1 score and assign MLflow aliases.
    """

    try:

        logger.info("=" * 60)
        logger.info("Starting Champion Model Evaluation")
        logger.info("=" * 60)

        client = MlflowClient()

        # -------------------------------------------
        # Get new model F1 score
        # -------------------------------------------

        new_version_info = client.get_model_version(
            name=model_name,
            version=new_version
        )

        new_run = client.get_run(
            new_version_info.run_id
        )

        new_f1 = new_run.data.metrics.get("f1_score")

        if new_f1 is None:
            raise ValueError(
                "F1 score not found in the new model MLflow run."
            )

        new_f1 = float(new_f1)

        logger.info(
            f"New Model F1 Score : {new_f1:.4f}"
        )

        # -------------------------------------------
        # Find current champion
        # -------------------------------------------

        try:

            champion_version = (
                client.get_model_version_by_alias(
                    name=model_name,
                    alias="champion"
                )
            )

        except Exception:

            champion_version = None

        # -------------------------------------------
        # No champion yet
        # -------------------------------------------

        if champion_version is None:

            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=new_version
            )

            logger.info("No Existing Champion Found.")
            logger.info(
                f"Version {new_version} assigned as CHAMPION."
            )

            logger.info("=" * 60)

            return

        # -------------------------------------------
        # Existing champion F1
        # -------------------------------------------

        champion_run = client.get_run(
            champion_version.run_id
        )

        champion_f1 = (
            champion_run.data.metrics.get("f1_score")
        )

        if champion_f1 is None:
            raise ValueError(
                "F1 score not found for current champion."
            )

        champion_f1 = float(champion_f1)

        logger.info(
            f"Current Champion Version : "
            f"{champion_version.version}"
        )

        logger.info(
            f"Current Champion F1      : "
            f"{champion_f1:.4f}"
        )

        # -------------------------------------------
        # Compare models
        # -------------------------------------------

        if new_f1 > champion_f1:

            # Previous champion becomes challenger
            client.set_registered_model_alias(
                name=model_name,
                alias="challenger",
                version=champion_version.version
            )

            # New model becomes champion
            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=new_version
            )

            logger.info("Better Model Found.")
            logger.info(
                f"Version {new_version} promoted to CHAMPION."
            )

            logger.info(
                f"Version {champion_version.version} "
                f"moved to CHALLENGER."
            )

        else:

            # New model becomes challenger
            client.set_registered_model_alias(
                name=model_name,
                alias="challenger",
                version=new_version
            )

            logger.info(
                "Current Champion Performs Better."
            )

            logger.info(
                f"Version {new_version} assigned as CHALLENGER."
            )

        logger.info("=" * 60)
        logger.info("Champion Model Evaluation Completed")
        logger.info("=" * 60)

    except Exception:

        logger.exception(
            "Champion Model Evaluation Failed"
        )
        raise

# -------------------- Main -------------------- #

def main():

    try:

        logger.info("=" * 60)
        logger.info("Model Registry Pipeline Started")


        # -------------------- DagsHub Authentication -------------------- #

        os.environ["MLFLOW_TRACKING_USERNAME"] = "kush2501"

        token = os.getenv("DAGSHUB_TOKEN")

        if token is None:
            raise ValueError("DAGSHUB_TOKEN environment variable is not set.")

        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        # Register token with DagsHub client
        dagshub.auth.add_app_token(token)

        # -------------------- DagsHub + MLflow -------------------- #

        mlflow.set_tracking_uri(
            "https://dagshub.com/kush2501/emotion-detection-using-mlflow-dvc.mlflow"
        )

        dagshub.init(
            repo_owner="kush2501",
            repo_name="emotion-detection-using-mlflow-dvc",
            mlflow=True
        )

        # -------------------- Load Run Info -------------------- #

        run_info = load_run_info()

        # -------------------- Register Model -------------------- #
        model_name, version = register_model(
            run_info
        )

        # -------------------- Champion Promotion -------------------- #

        promote_best_model(
            model_name,
            version
        )

        logger.info("Model Registry Pipeline Completed Successfully.")
        logger.info("=" * 60)

    except Exception:

        logger.exception("Pipeline Failed")
        raise


if __name__ == "__main__":
    main()