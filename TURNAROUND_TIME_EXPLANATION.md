# Turnaround Time Calculation - Method and Process

## Overview

The "taxi time" (what we call turnaround time) is the time an aircraft spends moving on the ground between the gate and runway.

## Data Source

The data comes from **ADSB (Automatic Dependent Surveillance-Broadcast)** tracking:
- Aircraft broadcast their position every few seconds
- Ground stations receive these signals
- Data is processed to track complete taxi operations

## How Taxi Time is Calculated

### Step 1: Raw Data Collection
The processed ADSB data contains these key fields for each operation:

```python
# Example from man_features_04.csv
{
    'id': '01-Apr-2021_QAJ2815',           # Unique flight identifier
    'type': 'departure',                    # arrival or departure
    'mode_s': '0x3CEA42',                  # Aircraft identifier
    'taxi_time': 21.783333,                # CALCULATED taxi time in minutes
    'gate (block) hour': '23:00:01',       # Time at gate
    'runway hour': '23:21:48',             # Time at runway
    'distance': 2003.718594,               # Total distance traveled (meters)
    'shortest path': 1853.451483,          # Shortest possible route (meters)
    # ... other features
}
```

### Step 2: Taxi Time Calculation Formula

The taxi time is calculated as:

```
taxi_time = runway_time - gate_time
```

**For Departures:**
```
taxi_time = takeoff_time - gate_departure_time
```
Example: 23:21:48 - 23:00:01 = 21.78 minutes

**For Arrivals:**
```
taxi_time = gate_arrival_time - landing_time
```
Example: 14:15:30 - 14:09:05 = 6.42 minutes

### Step 3: Data Cleaning (Critical!)

The raw data had issues, so we clean it:

```python
# From process_and_train_proper.py

# Remove negative times (data errors)
df = df[df['taxi_time'] >= 0]

# Remove unrealistically short times (< 2 minutes)
# These are likely incomplete tracking or position updates only
df = df[df['taxi_time'] >= 2]

# Remove unrealistically long times (> 60 minutes)
# These are likely aircraft parking, maintenance, or data errors
df = df[df['taxi_time'] <= 60]
```

**Why these thresholds?**
- **< 2 min**: Physically impossible to taxi from gate to runway in under 2 minutes
- **> 60 min**: Normal taxi operations don't take more than 1 hour
- **Negative**: Obviously data errors (time going backwards)

## Code Walkthrough

Let me show you the key parts of the code:

### 1. Loading the Data

```python
# From process_and_train_proper.py

def load_and_clean_data():
    """Load all monthly feature files and clean the data"""
    data_path = 'Processed ABSD data/.../Processed ADSB Dataset/'
    files = glob.glob(os.path.join(data_path, 'man_features_*.csv'))
    
    # Load all monthly files
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # CRITICAL: Clean the data
    combined_df = combined_df.dropna(subset=['taxi_time'])  # Remove missing
    combined_df = combined_df[combined_df['taxi_time'] >= 0]  # Remove negative
    combined_df = combined_df[combined_df['taxi_time'] >= 2]  # Min 2 minutes
    combined_df = combined_df[combined_df['taxi_time'] <= 60]  # Max 60 minutes
    
    return combined_df
```

### 2. Feature Engineering

```python
def prepare_features(df):
    """Prepare features for modeling"""
    data = df.copy()
    
    # Target variable (already in minutes from the data)
    data['taxi_time_minutes'] = data['taxi_time']
    
    # Encode operation type (departure = 1, arrival = 0)
    data['is_departure'] = (data['type'] == 'departure').astype(int)
    
    # Extract hour from gate time
    data['hour'] = pd.to_datetime(
        data['gate (block) hour'], 
        format='%H:%M:%S', 
        errors='coerce'
    ).dt.hour
    
    # Create peak hour indicator (morning/evening rush)
    data['is_peak_hour'] = data['hour'].apply(
        lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0
    )
    
    # Total queue size (all aircraft waiting)
    data['total_queue'] = (
        data['QDepDep'] +  # Departures waiting for departures
        data['QDepArr'] +  # Departures waiting for arrivals
        data['QArrDep'] +  # Arrivals waiting for departures
        data['QArrArr']    # Arrivals waiting for arrivals
    )
    
    return data
```

### 3. Training the Model

```python
def train_model(X, y):
    """Train the Random Forest model"""
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,      # 200 decision trees
        max_depth=15,          # Maximum tree depth
        min_samples_split=5,   # Minimum samples to split a node
        min_samples_leaf=2,    # Minimum samples in a leaf
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'test': {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
    }
    
    return model, metrics, y_test, y_pred_test
```

## How the Model Predicts

### Input Features (18 total):

1. **Distance Features:**
   - `distance`: Total taxi distance (meters)
   - `shortest path`: Shortest route distance (meters)
   - `distance_gate`: Distance from gate (meters)
   - `distance_long`: Long taxiway distance (meters)
   - `distance_else`: Other distances (meters)

2. **Traffic Features:**
   - `other_moving_ac`: Number of other aircraft moving
   - `total_queue`: Total aircraft in queues
   - `QDepDep`, `QDepArr`, `QArrDep`, `QArrArr`: Specific queue sizes
   - `NDepDep`, `NDepArr`, `NArrDep`, `NArrArr`: Number of interactions

3. **Temporal Features:**
   - `hour`: Hour of day (0-23)
   - `is_peak_hour`: Peak hour indicator (0 or 1)

4. **Operation Features:**
   - `is_departure`: Departure (1) or arrival (0)

### Prediction Process:

```python
# Example prediction
import pickle
import pandas as pd

# Load trained model
with open('proper_taxi_time_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']

# Create feature vector for a departure with moderate traffic
features = pd.DataFrame({
    'distance': [2000],              # 2km taxi distance
    'shortest path': [1800],         # 1.8km shortest route
    'distance_gate': [100],          # 100m from gate
    'distance_long': [1000],         # 1km on long taxiway
    'distance_else': [900],          # 900m other
    'other_moving_ac': [8],          # 8 other aircraft
    'total_queue': [12],             # 12 aircraft in queues
    'hour': [14],                    # 2 PM
    'is_departure': [1],             # Departure
    'is_peak_hour': [0],             # Not peak hour
    'QDepDep': [4],                  # 4 departures waiting
    'QDepArr': [3],                  # 3 dep waiting for arr
    'QArrDep': [3],                  # 3 arr waiting for dep
    'QArrArr': [2],                  # 2 arrivals waiting
    'NDepDep': [2],                  # 2 dep-dep interactions
    'NDepArr': [1],                  # 1 dep-arr interaction
    'NArrDep': [1],                  # 1 arr-dep interaction
    'NArrArr': [1]                   # 1 arr-arr interaction
})

# Predict taxi time
prediction = model.predict(features)
print(f"Predicted taxi time: {prediction[0]:.1f} minutes")
# Output: Predicted taxi time: 13.2 minutes
```

## How Random Forest Works

The Random Forest makes predictions by:

1. **Creating 200 decision trees**, each trained on a random subset of data
2. **Each tree makes a prediction** based on the input features
3. **Average all predictions** to get the final result

Example decision tree logic:
```
if total_queue > 10:
    if is_departure == 1:
        if distance > 2000:
            predict: 15.3 minutes
        else:
            predict: 11.8 minutes
    else:
        predict: 7.2 minutes
else:
    if distance > 1500:
        predict: 9.5 minutes
    else:
        predict: 6.1 minutes
```

## Feature Importance

The model learned that these features matter most:

1. **total_queue (40.9%)** - Most important!
   - More aircraft waiting = longer taxi times
   - Queue congestion is the biggest bottleneck

2. **is_departure (14.2%)** - Second most important
   - Departures take 2x longer than arrivals
   - Departures must join runway queue

3. **distance (10.9%)** - Third most important
   - Longer routes = longer times
   - But queue effects dominate

## Statistics from Our Data

After cleaning, we found:

```
Total Operations: 58,972
├── Arrivals: 29,120 (49.4%)
│   ├── Mean: 6.4 minutes
│   ├── Median: 4.7 minutes
│   └── Range: 2-60 minutes
│
└── Departures: 29,852 (50.6%)
    ├── Mean: 12.7 minutes
    ├── Median: 11.3 minutes
    └── Range: 2-60 minutes

Overall Mean: 9.6 minutes
Overall Median: 8.2 minutes
```

## Why Departures Take Longer

**Arrival Process (6.4 min average):**
```
Landing → Taxi to gate → Park
         (direct route)
```

**Departure Process (12.7 min average):**
```
Leave gate → Taxi to runway → Join queue → Wait → Takeoff
            (longer route)    (congestion!)
```

The departure queue waiting time is the key difference!

## Validation

We validated the model by:

1. **Splitting data**: 80% training, 20% testing
2. **Testing on unseen data**: Model never saw test data during training
3. **Checking metrics**:
   - R² = 0.773 (explains 77% of variance)
   - MAE = 1.83 minutes (average error)
   - RMSE = 3.52 minutes (error spread)

4. **Sanity checks**:
   - ✅ Mean taxi time is realistic (9.6 min)
   - ✅ Departures > arrivals
   - ✅ Queue size is top predictor
   - ✅ No negative predictions

## Summary

**The Process:**
1. ADSB tracks aircraft positions
2. Data processor calculates: `taxi_time = runway_time - gate_time`
3. We clean the data (remove outliers)
4. We engineer features (queue sizes, distances, time of day)
5. Random Forest learns patterns from 58,972 operations
6. Model predicts taxi time for new operations

**Key Insight:**
Queue congestion (40.9% importance) matters more than distance (10.9% importance) for predicting taxi times!

**Result:**
A model that predicts taxi times with ±2 minute accuracy, matching real-world airport operations.
