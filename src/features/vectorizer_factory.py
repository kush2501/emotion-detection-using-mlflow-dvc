from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)


def get_vectorizer(feature_config):
    """
    Returns the vectorizer based on params.yaml configuration.
    """

    vectorizer_type = feature_config["vectorizer"]

    common_params = {
        "max_features": feature_config["max_features"],
        "lowercase": feature_config["lowercase"],
        "stop_words": feature_config["stop_words"],
        "ngram_range": tuple(feature_config["ngram_range"])
    }

    if vectorizer_type == "count":
        return CountVectorizer(**common_params)

    elif vectorizer_type == "tfidf":
        return TfidfVectorizer(**common_params)

    else:
        raise ValueError(
            f"Unsupported vectorizer: {vectorizer_type}"
        )