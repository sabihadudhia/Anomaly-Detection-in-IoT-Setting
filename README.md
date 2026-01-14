# IoT Anomaly Detection System

Anomaly detection for industrial IoT sensors using Isolation Forest and Flask REST API.

## Basic Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline (preprocessing, training, streaming)
python main.py

# Start API server
python API/api.py
```

Access web interface at `http://localhost:5000`

## API Usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 25.5, "humidity": 60.2, "sound_volume": 0.025}'
```

Response:
```json
{
  "anomaly": false,
  "anomaly_score": -0.1234
}
```

## Project Structure

```
├── API/api.py              # Flask REST API
├── Model/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── stream_data.py
├── Data/iot_telemetry_data.csv
└── main.py
```

## Technologies

- Python 3.8+
- scikit-learn (Isolation Forest)
- Flask
- pandas, numpy
