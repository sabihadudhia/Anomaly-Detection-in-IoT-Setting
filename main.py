"""
Main orchestration script for IoT Anomaly Detection Pipeline.

This script runs the complete workflow in order:
1. Data preprocessing and cleaning
2. Model training
3. Data streaming with anomaly detection
"""
import logging
from pathlib import Path

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Execute the complete anomaly detection pipeline."""
    
    logger.info("="*60)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("="*60)
    
    from Model.data_preprocessing import (
        load_and_clean_data,
        save_cleaned_data,
        scale_features,
        save_scaler,
        DATA_PATH
    )
    
    # Load and clean data
    sensor_df = load_and_clean_data(DATA_PATH)
    save_cleaned_data(sensor_df)
    
    # Scale features and save scaler
    X_scaled, scaler = scale_features(sensor_df)
    save_scaler(scaler)
    
    logger.info("✓ Data preprocessing completed")
    
    # --------------------------------------------------
    
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Model Training")
    logger.info("="*60)
    
    from Model.train_model import train_anomaly_model, save_model
    
    # Train anomaly detection model
    model = train_anomaly_model(X_scaled)
    save_model(model)
    
    logger.info("✓ Model training completed")
    
    # --------------------------------------------------
    
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Stream Data and Detect Anomalies")
    logger.info("="*60)
    
    from Model.stream_data import load_model_and_scaler, detect_anomalies
    
    # Load trained model and scaler
    model, scaler = load_model_and_scaler()
    
    # Stream data and detect anomalies
    detect_anomalies(model, scaler, sensor_df, max_records=100)
    
    logger.info("✓ Data streaming completed")
    
    # --------------------------------------------------
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info("\nTo start the Flask API server, run:")
    logger.info("  python API/api.py")


if __name__ == "__main__":
    run_pipeline()
