import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "never"
}


def lower_case(text: str) -> str:
    """Convert text to lowercase."""
    return " ".join(
        word.lower()
        for word in str(text).split()
    )


def remove_stop_words(text: str) -> str:
    """
    Remove English stopwords while preserving
    sentiment-critical negation words.
    """

    stop_words = set(stopwords.words("english"))

    stop_words = stop_words - NEGATION_WORDS

    filtered_words = [
        word
        for word in str(text).split()
        if word not in stop_words
    ]

    return " ".join(filtered_words)


def remove_numbers(text: str) -> str:
    """Remove numeric characters."""
    return "".join(
        char for char in text
        if not char.isdigit()
    )


def remove_punctuations(text: str) -> str:
    """Remove punctuation and normalize whitespace."""

    text = re.sub(
        r"""[!"#$%&'()*+,،\-./:;<=>؟?@\[\]^_`{|}~]""",
        " ",
        text
    )

    text = text.replace("؛", "")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS and WWW URLs."""

    url_pattern = re.compile(
        r"https?://\S+|www\.\S+"
    )

    return url_pattern.sub("", text)


def lemmatize_text(text: str) -> str:
    """Lemmatize individual words."""

    lemmatizer = WordNetLemmatizer()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


def preprocess_text(text: str) -> str:
    """
    Apply the complete text preprocessing pipeline.

    This function is shared by training and inference
    to prevent training-serving preprocessing skew.
    """

    text = str(text)

    text = lower_case(text)
    text = remove_stop_words(text)
    text = remove_numbers(text)
    text = remove_punctuations(text)
    text = remove_urls(text)
    text = lemmatize_text(text)

    return text.strip()