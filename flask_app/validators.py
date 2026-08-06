MAX_TEXT_LENGTH = 5000


def validate_prediction_input(data):
    """
    Validate JSON input for sentiment prediction API.

    Returns:
        tuple: (text, error_message)
    """

    if not isinstance(data, dict):
        return None, "Request body must be a valid JSON object."

    if "text" not in data:
        return None, "Field 'text' is required."

    text = data["text"]

    if not isinstance(text, str):
        return None, "Field 'text' must be a string."

    text = text.strip()

    if not text:
        return None, "Field 'text' cannot be empty."

    if len(text) > MAX_TEXT_LENGTH:
        return None, (
            f"Field 'text' must not exceed "
            f"{MAX_TEXT_LENGTH} characters."
        )

    return text, None