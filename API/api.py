import logging
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_PATH = Path("Model/anomaly_model.pkl")
SCALER_PATH = Path("Model/scaler.pkl")

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load model and scaler
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

logger.info("Model and scaler loaded successfully")

# --------------------------------------------------
# Flask app
# --------------------------------------------------
app = Flask(__name__, template_folder='templates')


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        temperature = data["temperature"]
        humidity = data["humidity"]
        sound_volume = data["sound_volume"]

        features = scaler.transform([[temperature, humidity, sound_volume]])
        prediction = model.predict(features)[0]
        score = model.decision_function(features)[0]

        response = {
            "temperature": temperature,
            "humidity": humidity,
            "sound_volume": sound_volume,
            "anomaly": bool(prediction == -1),
            "anomaly_score": float(score),
        }

        logger.info("Prediction made: %s", response)
        return jsonify(response)

    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        return {"error": str(e)}, 400


if __name__ == "__main__":
    app.run(debug=True)
