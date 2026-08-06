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

# -------------------- Production API Tests -------------------- #

def test_health_endpoint():
    """Health endpoint should report the application as healthy."""

    response = client.get("/health")
    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["vectorizer_loaded"] is True


def test_model_info_endpoint():
    """Model information endpoint should return model metadata."""

    response = client.get("/model-info")
    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "ready"
    assert "model" in data
    assert "feature_engineering" in data


def test_api_positive_prediction():
    """JSON API should correctly predict positive sentiment."""

    response = client.post(
        "/api/predict",
        json={"text": "I am very happy today"}
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert "Positive" in data["prediction"]["prediction"]


def test_api_negation_prediction():
    """JSON API should preserve negation during prediction."""

    response = client.post(
        "/api/predict",
        json={"text": "I am not happy today"}
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert "Negative" in data["prediction"]["prediction"]


def test_api_empty_text():
    """Empty text should be rejected."""

    response = client.post(
        "/api/predict",
        json={"text": ""}
    )

    data = response.get_json()

    assert response.status_code == 400
    assert data["status"] == "error"


def test_api_invalid_text_type():
    """Non-string text should be rejected."""

    response = client.post(
        "/api/predict",
        json={"text": 123}
    )

    data = response.get_json()

    assert response.status_code == 400
    assert data["status"] == "error"


def test_not_found():
    """Unknown endpoint should return HTTP 404."""

    response = client.get("/does-not-exist")
    data = response.get_json()

    assert response.status_code == 404
    assert data["status"] == "error"


def test_method_not_allowed():
    """GET should not be allowed on prediction API."""

    response = client.get("/api/predict")
    data = response.get_json()

    assert response.status_code == 405
    assert data["status"] == "error"