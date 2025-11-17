# Airport Taxi Time Prediction Model

Machine learning model to predict aircraft taxi times at Manchester Airport using real ABSD (Airport Surface Movement) data.

## Results

**Model trained on 63,842 real operations from Manchester Airport (April-October 2021)**

### Performance Metrics
- **R² Score**: 0.928 (92.8% accuracy)
- **MAE**: 3.6 seconds
- **RMSE**: 18.6 seconds

### Key Findings
- **Queue size** is the dominant predictor (87.4% importance)
- **Traffic density** significantly impacts taxi time (5.5% importance)
- **Time of day** affects operations (2.1% importance)
- Model predicts taxi times with high accuracy across 3,174 unique aircraft

## Dataset
- 63,842 operations (32,981 arrivals, 30,861 departures)
- Manchester Airport network: 285 nodes, 305 edges, 105 gates
- Primary runway: 23R/05L (90% of operations)
- Average traffic: 5 concurrent moving aircraft

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train Model with Real Data
```bash
python train_real_model.py
```

### Analyze Data
```bash
python analyze_real_data.py
```

## Model Features

**Input Features:**
- Distance metrics (total, shortest path, gate distance)
- Queue sizes (departures, arrivals)
- Traffic density (other moving aircraft)
- Temporal features (hour of day, peak hours)
- Operation type (arrival/departure)

**Output:**
- Predicted taxi time in seconds

## Quick Example

```python
import pickle
import pandas as pd

# Load model
with open('real_turnaround_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']

# Predict
features = pd.DataFrame({
    'distance': [2000],
    'shortest path': [1500],
    'distance_gate': [100],
    'other_moving_ac': [5],
    'total_queue': [10],
    'hour': [14],
    'is_departure': [1],
    # ... other features
})

prediction = model.predict(features)
print(f"Predicted taxi time: {prediction[0]:.1f} seconds")
```

## Files

- `train_real_model.py` - Train model on real ABSD data
- `analyze_real_data.py` - Data analysis and visualization
- `process_absd_data.py` - Process airport network data
- `requirements.txt` - Python dependencies

## Note

This model predicts **taxi time** (time spent moving on taxiways), not full gate turnaround time. For full turnaround prediction, gate block-on/block-off timestamps would be needed.
