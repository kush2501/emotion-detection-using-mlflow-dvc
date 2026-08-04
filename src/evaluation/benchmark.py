import logging
import time
import pandas as pd

from pathlib import Path
from src.model.model_factory import MODEL_REGISTRY
from src.config.config_loader import load_model_config
from src.evaluation.evaluator import evaluate_model


BASE_DIR = Path(__file__).resolve().parent.parent.parent

# logging.
logger = logging.getLogger(__name__)


class BenchmarkEngine:
    """
    Run benchmark experiments across multiple machine learning models.
    """

    def __init__(self):

        self.model_registry = MODEL_REGISTRY

        self.model_config = load_model_config()

        self.results = []

        print("Benchmark Engine Initialized.")
        print(f"Available Models: {list(self.model_registry.keys())}")


    def load_data(self):

        logger.info("Loading benchmark datasets...")

        train_path = BASE_DIR / "data" / "processed" / "train_bow.csv"
        test_path = BASE_DIR / "data" / "processed" / "test_bow.csv"

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        X_train = train_data.drop(columns=["label"])
        y_train = train_data["label"]

        X_test = test_data.drop(columns=["label"])
        y_test = test_data["label"]

        logger.info(f"X_train Shape : {X_train.shape}")
        logger.info(f"y_train Shape : {y_train.shape}")
        logger.info(f"X_test Shape  : {X_test.shape}")
        logger.info(f"y_test Shape  : {y_test.shape}")

        return X_train, y_train, X_test, y_test
            

    def run(self):

        logger.info("Benchmark Started.")

        X_train, y_train, X_test, y_test = self.load_data()

        for model_name, model_creator in self.model_registry.items():

            logger.info("=" * 60)
            logger.info(f"Creating Model : {model_name}")

            model = model_creator(self.model_config)

            logger.info(
                f"Model Created : {model.__class__.__name__}")

            logger.info(f"Training {model.__class__.__name__}...")

            start_time = time.perf_counter()

            model.fit(X_train, y_train)

            training_time = time.perf_counter() - start_time

            logger.info(
                f"{model.__class__.__name__} trained successfully."
            )

            logger.info(
                f"Training Time : {training_time:.2f} seconds"
            )


            logger.info(
                f"Evaluating {model.__class__.__name__}..."
            )

            metrics, signature = evaluate_model(
                model,
                X_test,
                y_test
            )

            self.results.append({
                "model_name": model_name,
                "model_type": model.__class__.__name__,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "training_time": round(training_time, 2),
                "status": "evaluated"
            })

        logger.info("=" * 60)
        logger.info("All benchmark models created successfully.")

        results_df = self.save_results()

        return results_df

    def save_results(self):
        """
        Save benchmark results as a CSV file
        and display the champion model.
        """

        logger.info("Saving benchmark results...")

        results_df = pd.DataFrame(self.results)

        results_df = results_df.sort_values(
            by="f1_score",
            ascending=False
        )

        reports_dir = BASE_DIR / "reports"
        reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            reports_dir
            / "benchmark_results.csv"
        )

        results_df.to_csv(
            output_path,
            index=False
        )

        logger.info(
            f"Benchmark results saved to: {output_path}"
        )

        # Select champion model
        best_model = results_df.iloc[0]

        logger.info("=" * 60)
        logger.info("CHAMPION MODEL SELECTED")

        logger.info(
            f"Model         : {best_model['model_name']}"
        )

        logger.info(
            f"F1 Score      : {best_model['f1_score']:.4f}"
        )

        logger.info(
            f"Accuracy      : {best_model['accuracy']:.4f}"
        )

        logger.info(
            f"Training Time : "
            f"{best_model['training_time']:.2f} seconds"
        )

        logger.info("=" * 60)

        return results_df


if __name__ == "__main__":
    engine = BenchmarkEngine()
    engine.run()