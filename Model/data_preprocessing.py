import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "Data" / "iot_telemetry_data.csv"
OUTPUT_PATH = BASE_DIR / "Data" / "clean_sensor_data.csv"
SCALER_OUTPUT_PATH = BASE_DIR / "Model" / "scaler.pkl"

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Data loading and cleaning
# --------------------------------------------------
def load_and_clean_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load IoT telemetry dataset and clean it."""
    logger.info("Loading IoT telemetry dataset")
    df = pd.read_csv(data_path)

    logger.info("Dataset preview:")
    logger.info("\n%s", df.head())

    logger.info("Dataset statistics:")
    logger.info("\n%s", df.describe())

    sensor_df = (
        df[["temp", "humidity", "smoke"]]
        .dropna()
        .rename(columns={"temp": "temperature", "smoke": "sound_volume"})
    )

    return sensor_df


# --------------------------------------------------
# Feature scaling
# --------------------------------------------------
def scale_features(sensor_df: pd.DataFrame) -> tuple:
    """
    Scale features using StandardScaler.
    
    Returns:
        tuple: (scaled_features, scaler_object)
    """
    logger.info("Preparing features for model training")
    X = sensor_df[["temperature", "humidity", "sound_volume"]]

    logger.info("Scaling features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# --------------------------------------------------
# Save cleaned data and scaler
# --------------------------------------------------
def save_cleaned_data(sensor_df: pd.DataFrame, output_path: Path = OUTPUT_PATH):
    """Save cleaned sensor data to CSV."""
    sensor_df.to_csv(output_path, index=False)
    logger.info("Cleaned sensor data saved to %s", output_path)


def save_scaler(scaler: StandardScaler, scaler_path: Path = SCALER_OUTPUT_PATH):
    """Save fitted scaler to disk."""
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to %s", scaler_path)


if __name__ == "__main__":
    # Load and clean data
    sensor_df = load_and_clean_data()
    save_cleaned_data(sensor_df)

    # Scale features and save scaler
    X_scaled, scaler = scale_features(sensor_df)
    save_scaler(scaler)
