from flask_app.app import app


client = app.test_client()


def test_home_page():
    """Test whether home page loads successfully."""
    response = client.get("/")

    assert response.status_code == 200


def test_positive_prediction():
    """Clearly positive text should be predicted as positive."""
    response = client.post(
        "/predict",
        data={"text": "I am very happy today"}
    )

    assert response.status_code == 200
    assert b"Positive" in response.data


def test_negation_prediction():
    """
    Regression test for negation handling.

    'not happy' should not be interpreted as positive.
    """
    response = client.post(
        "/predict",
        data={"text": "I am not happy today"}
    )

    assert response.status_code == 200
    assert b"Negative" in response.data


def test_negative_prediction():
    """Clearly negative text should be predicted as negative."""
    response = client.post(
        "/predict",
        data={"text": "I never liked this movie"}
    )

    assert response.status_code == 200
    assert b"Negative" in response.data