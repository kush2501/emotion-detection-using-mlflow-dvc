from pathlib import Path
import yaml
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_params():
    """
    Load parameters from params.yaml.
    """

    try:

        logger.info("Loading params.yaml...")

        params_path = BASE_DIR / "params.yaml"

        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        return params

    except Exception:

        logger.exception("Failed to load params.yaml.")
        raise


def load_model_config():
    """
    Load model configuration.
    """

    try:

        params = load_params()

        model_config = params["model"]

        algorithm = model_config.get("algorithm")

        if algorithm is None:

            raise ValueError(
                "'algorithm' is missing in model configuration."
            )

        logger.info(f"Selected Algorithm : {algorithm}")

        logger.info("Model Parameters:")

        for key, value in model_config[algorithm].items():

            logger.info(f"    {key:<20}: {value}")

        return model_config

    except Exception:

        logger.exception("Failed to load model configuration.")
        raise


def load_feature_config():
    """
    Load feature engineering configuration.
    """

    try:

        params = load_params()

        feature_config = params["feature_engineering"]

        logger.info("Feature Configuration:")

        for key, value in feature_config.items():

            logger.info(f"    {key:<20}: {value}")

        return feature_config

    except Exception:

        logger.exception("Failed to load feature configuration.")
        raise


if __name__ == "__main__":

    print("Config Loader Started")

    load_model_config()

    print()

    load_feature_config()