import logging
from src.model.model_factory import MODEL_REGISTRY
from src.model.model_building import load_model_config


logger = logging.getLogger(__name__)

class BenchmarkEngine:
    """
    Run benchmark experiments across multiple machine learning models.
    """

    def __init__(self):

        self.model_registry = MODEL_REGISTRY

        self.model_config = load_model_config()

        self.results = []

        logger.info("Benchmark Engine Initialized.")
        logger.info(f"Available Models: {list(self.model_registry.keys())}")