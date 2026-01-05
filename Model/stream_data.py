import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .data_preprocessing import (
    load_and_clean_data,
    DATA_PATH,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "Model" / "anomaly_model.pkl"
SCALER_PATH = BASE_DIR / "Model" / "scaler.pkl"

STREAM_DELAY_SECONDS = 1
MAX_RECORDS = 100

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def utc_timestamp(timespec: str = "seconds") -> str:
    """
    Return current UTC timestamp in RFC 3339 format.

    Example: 2025-12-22T22:26:47Z
    """
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec=timespec)
        .replace("+00:00", "Z")
    )


def load_model_and_scaler(model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
    """Load trained model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Model loaded from %s", model_path)
    logger.info("Scaler loaded from %s", scaler_path)
    return model, scaler


def detect_anomalies(model, scaler, sensor_df, max_records: int = MAX_RECORDS):
    """
    Stream sensor data and detect anomalies using the trained model.
    
    Args:
        model: Trained anomaly detection model
        scaler: Fitted scaler for feature scaling
        sensor_df: DataFrame containing sensor data
        max_records: Maximum number of records to stream
    """
    logger.info("Starting sensor data stream")

    for i, row in sensor_df.iterrows():
        if i >= max_records:
            logger.info("Stream finished after %d records", max_records)
            break

        # Prepare sensor event
        temperature = float(row["temperature"])
        humidity = float(row["humidity"])
        sound_volume = float(row["sound_volume"])

        sensor_event = {
            "timestamp": utc_timestamp(),
            "temperature": temperature,
            "humidity": humidity,
            "sound_volume": sound_volume,
        }

        # Scale features and predict
        features = pd.DataFrame(
            [[temperature, humidity, sound_volume]],
            columns=["temperature", "humidity", "sound_volume"]
        )
        features_scaled = scaler.transform(features)
        anomaly_prediction = model.predict(features_scaled)[0]
        anomaly_score = model.decision_function(features_scaled)[0]

        # Add anomaly flag (-1 for anomaly, 1 for normal) and anomaly score
        sensor_event["is_anomaly"] = anomaly_prediction == -1
        sensor_event["anomaly_score"] = round(anomaly_score, 4)

        logger.info(sensor_event)
        time.sleep(STREAM_DELAY_SECONDS)


if __name__ == "__main__":
    # Load data and trained model/scaler
    sensor_df = load_and_clean_data(DATA_PATH)
    model, scaler = load_model_and_scaler()

    # Stream data and detect anomalies
    detect_anomalies(model, scaler, sensor_df)

