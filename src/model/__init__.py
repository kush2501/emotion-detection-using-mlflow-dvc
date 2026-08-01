from sklearn.linear_model import LogisticRegression


def get_model(model_config):
    """
    Returns the ML model based on params.yaml configuration.
    """

    algorithm = model_config["algorithm"]

    if algorithm == "logistic_regression":

        lr_config = model_config["logistic_regression"]

        return LogisticRegression(
            C=lr_config["C"],
            solver=lr_config["solver"],
            penalty=lr_config["penalty"],
            max_iter=lr_config["max_iter"]
        )

    else:
        raise ValueError(
            f"Unsupported Algorithm : {algorithm}"
        )