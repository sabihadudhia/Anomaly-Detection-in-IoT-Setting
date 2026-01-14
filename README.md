# IoT Anomaly Detection System

## Overview
Anomaly detection for industrial IoT sensor data using Isolation Forest. The system includes data preprocessing, model training, streaming data simulation, and a REST API for real-time predictions.

## Features
- Preprocess IoT telemetry data
- Train Isolation Forest model for anomaly detection
- Stream simulated sensor data
- Provide REST API for real-time predictions
- Simple web interface to view predictions

## Technologies
- Python 3.8+
- scikit-learn (Isolation Forest)
- Flask
- pandas, numpy

## Setup / Installation
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main pipeline (preprocessing, training, streaming):
```bash
python main.py
```

3. Start the API server:
```bash
python API/api.py
```

## Usage
- Access the web interface at http://localhost:5000
- Send a POST request to the API to predict anomalies:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 25.5, "humidity": 60.2, "sound_volume": 0.025}'
```

## Example API response:
```json
{
  "anomaly": false,
  "anomaly_score": -0.1234
}
```

## Project Structure
```bash 
├── API/api.py
├── Model/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── stream_data.py
├── Data/iot_telemetry_data.csv
└── main.py
```
