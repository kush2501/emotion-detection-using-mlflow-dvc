import logging
import os


from flask import Flask, render_template, request, jsonify

from .validators import validate_prediction_input

from .predictor import (
    predict_sentiment,
    get_model_status,
    get_model_info
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


# -------------------- Create Flask App -------------------- #

def create_app():

    app = Flask(__name__)

    # -------------------- Home -------------------- #

    @app.route("/")
    def home():

        return render_template("index.html")

    # -------------------- Health -------------------- #

    @app.route("/health", methods=["GET"])
    def health():

        health_status = get_model_status()

        status_code = (
            200
            if health_status["status"] == "healthy"
            else 503
        )

        return jsonify(health_status), status_code

    # -------------------- Model Information -------------------- #

    @app.route("/model-info", methods=["GET"])
    def model_info():

        return jsonify(get_model_info()), 200

    # -------------------- Prediction -------------------- #

    @app.route("/predict", methods=["POST"])
    def predict():

        text = request.form["text"]

        result = predict_sentiment(text)

        return render_template(
            "index.html",
            prediction=result["prediction"],
            confidence=result["confidence"],
            processed_text=result["processed_text"],
            original_text=text
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():

        data = request.get_json(silent=True)

        text, error = validate_prediction_input(data)

        if error:
            return jsonify({
                "status": "error",
                "message": error
            }), 400

        try:
            result = predict_sentiment(text)

            return jsonify({
                "status": "success",
                "prediction": result
            }), 200

        except Exception:
            app.logger.exception(
                "Unexpected error during sentiment prediction."
            )

            return jsonify({
                "status": "error",
                "message": "Prediction could not be completed."
            }), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "status": "error",
            "message": "Resource not found."
        }), 404


    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "status": "error",
            "message": "Method not allowed."
        }), 405


    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            "status": "error",
            "message": "Internal server error."
        }), 500
    

    return app


# Create Flask App
app = create_app()


# -------------------- Run App -------------------- #

if __name__ == "__main__":

    debug_mode = os.getenv(
        "FLASK_DEBUG",
        "false"
    ).lower() == "true"

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=debug_mode
    )