import logging
from pathlib import Path

from sklearn.ensemble import IsolationForest
import joblib

from .data_preprocessing import (
    load_and_clean_data,
    scale_features,
    save_scaler,
    DATA_PATH,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_OUTPUT_PATH = BASE_DIR / "Model" / "anomaly_model.pkl"

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Model training
# --------------------------------------------------
def train_anomaly_model(X_scaled, random_state: int = 42, contamination: float = 0.05):
    """
    Train an Isolation Forest model for anomaly detection.
    
    Args:
        X_scaled: Scaled feature matrix
        random_state: Random seed for reproducibility
        contamination: Expected proportion of anomalies in dataset
        
    Returns:
        Trained IsolationForest model
    """
    logger.info("Training Isolation Forest model")
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(X_scaled)
    return model


def save_model(model: IsolationForest, model_path: Path = MODEL_OUTPUT_PATH):
    """Save trained model to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    # Load and preprocess data
    sensor_df = load_and_clean_data(DATA_PATH)
    X_scaled, scaler = scale_features(sensor_df)

    # Train model
    model = train_anomaly_model(X_scaled)

    # Save model only (scaler is already saved during preprocessing)
    save_model(model)
