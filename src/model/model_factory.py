from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

def create_logistic_regression(model_config):

    """Create a Logistic Regression model from configuration."""

    lr_config = model_config["logistic_regression"]

    return LogisticRegression(
        C=lr_config["C"],
        solver=lr_config["solver"],
        penalty=lr_config["penalty"],
        max_iter=lr_config["max_iter"]
    )

def create_random_forest(model_config):

    """Create a Random Forest model from configuration."""

    rf_config = model_config["random_forest"]

    return RandomForestClassifier(
        n_estimators=rf_config["n_estimators"],
        criterion=rf_config["criterion"],
        max_depth=rf_config["max_depth"],
        min_samples_split=rf_config["min_samples_split"],
        min_samples_leaf=rf_config["min_samples_leaf"],
        random_state=rf_config["random_state"]
    )
def create_svm(model_config):

    """Create a SVM model from configuration."""

    svm_config = model_config["svm"]

    return SVC(
        C=svm_config["C"],
        kernel=svm_config["kernel"],
        gamma=svm_config["gamma"],
        probability=svm_config["probability"],
        random_state=svm_config["random_state"]
    )

def create_naive_bayes(model_config):

    """Create a naive bayes model from configuration."""

    nb_config = model_config["naive_bayes"]

    return MultinomialNB(
        alpha=nb_config["alpha"],
        fit_prior=nb_config["fit_prior"]
    )

def create_xgboost(model_config):

    """Create a XGBoost Classifier model from configuration."""
    xgb_config = model_config["xgboost"]

    return XGBClassifier(
        n_estimators=xgb_config["n_estimators"],
        learning_rate=xgb_config["learning_rate"],
        max_depth=xgb_config["max_depth"],
        subsample=xgb_config["subsample"],
        colsample_bytree=xgb_config["colsample_bytree"],
        random_state=xgb_config["random_state"],
        eval_metric=xgb_config["eval_metric"]
    )


# Registry mapping algorithm names to model creation functions
MODEL_REGISTRY = {
    "logistic_regression": create_logistic_regression,
    "random_forest": create_random_forest,
    "svm": create_svm,
    "naive_bayes": create_naive_bayes,
    "xgboost": create_xgboost,
}


def get_model(model_config):

    algorithm = model_config.get("algorithm")

    if algorithm is None:
        raise ValueError("'algorithm' is missing in model configuration.")

    model_creator = MODEL_REGISTRY.get(algorithm)

    if model_creator is None:

        available_models = ", ".join(MODEL_REGISTRY.keys())

        raise ValueError(
            f"Unsupported model '{algorithm}'. "
            f"Available models: {available_models}"
        )

    return model_creator(model_config)