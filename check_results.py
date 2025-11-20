"""
Check that all results make sense
"""
import pandas as pd
import pickle
import glob

print("="*70)
print("VERIFICATION: Do the numbers make sense?")
print("="*70)

# 1. Check feature importance
print("\n1. FEATURE IMPORTANCE")
df_feat = pd.read_csv('proper_feature_importance.csv')
print(df_feat.head(10))
print(f"\nTotal importance: {df_feat['importance'].sum():.4f} (should be ~1.0)")

# 2. Check model metrics
print("\n2. MODEL METRICS")
with open('proper_taxi_time_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    metrics = model_data['metrics']

print(f"Test Set:")
print(f"  R² = {metrics['test']['r2']:.4f} (0.7-0.8 is good for real-world data)")
print(f"  MAE = {metrics['test']['mae']:.2f} minutes (±2 min is excellent)")
print(f"  RMSE = {metrics['test']['rmse']:.2f} minutes")

print(f"\nTraining Set:")
print(f"  R² = {metrics['train']['r2']:.4f}")
print(f"  MAE = {metrics['train']['mae']:.2f} minutes")

# 3. Check data statistics
print("\n3. DATA STATISTICS")
data_path = 'Processed ABSD data/Processed ADSB and Airport Dataset-20251117/Processed ADSB Dataset/'
files = glob.glob(data_path + 'man_features_*.csv')

all_data = []
for file in files:
    df = pd.read_csv(file)
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Clean data same way as model
df_clean = df_all[
    (df_all['taxi_time'] >= 2) & 
    (df_all['taxi_time'] <= 60) & 
    (df_all['taxi_time'].notna())
]

print(f"Total records: {len(df_all):,}")
print(f"Clean records: {len(df_clean):,} ({len(df_clean)/len(df_all)*100:.1f}%)")
print(f"\nTaxi Time Statistics (minutes):")
print(f"  Mean: {df_clean['taxi_time'].mean():.2f}")
print(f"  Median: {df_clean['taxi_time'].median():.2f}")
print(f"  Std: {df_clean['taxi_time'].std():.2f}")
print(f"  Min: {df_clean['taxi_time'].min():.2f}")
print(f"  Max: {df_clean['taxi_time'].max():.2f}")

print(f"\nBy Operation Type:")
for op_type in ['arrival', 'departure']:
    subset = df_clean[df_clean['type'] == op_type]['taxi_time']
    print(f"  {op_type.capitalize()}: {subset.mean():.2f} min (n={len(subset):,})")

# 4. Sanity checks
print("\n4. SANITY CHECKS")

checks = []

# Check 1: Mean taxi time is reasonable (5-15 minutes)
mean_taxi = df_clean['taxi_time'].mean()
checks.append(("Mean taxi time 5-15 min", 5 <= mean_taxi <= 15, f"{mean_taxi:.1f} min"))

# Check 2: Departures take longer than arrivals
arr_mean = df_clean[df_clean['type'] == 'arrival']['taxi_time'].mean()
dep_mean = df_clean[df_clean['type'] == 'departure']['taxi_time'].mean()
checks.append(("Departures > Arrivals", dep_mean > arr_mean, f"{dep_mean:.1f} > {arr_mean:.1f}"))

# Check 3: R² is reasonable (0.6-0.9)
r2 = metrics['test']['r2']
checks.append(("R² between 0.6-0.9", 0.6 <= r2 <= 0.9, f"{r2:.3f}"))

# Check 4: MAE is reasonable (1-3 minutes)
mae = metrics['test']['mae']
checks.append(("MAE between 1-3 min", 1 <= mae <= 3, f"{mae:.2f} min"))

# Check 5: Queue size is top feature
top_feature = df_feat.iloc[0]['feature']
checks.append(("Queue size is top feature", 'queue' in top_feature.lower(), top_feature))

# Check 6: No negative taxi times
has_negative = (df_clean['taxi_time'] < 0).any()
checks.append(("No negative taxi times", not has_negative, "All positive"))

for check_name, passed, value in checks:
    status = "✅" if passed else "❌"
    print(f"  {status} {check_name}: {value}")

# 5. Overall verdict
print("\n" + "="*70)
all_passed = all(check[1] for check in checks)
if all_passed:
    print("✅ ALL CHECKS PASSED - Numbers make sense!")
    print("="*70)
    print("\nThe model is predicting realistic taxi times.")
    print("Results are consistent with real-world airport operations.")
else:
    print("⚠️  SOME CHECKS FAILED")
    print("="*70)
    print("\nPlease review the failed checks above.")

print()
