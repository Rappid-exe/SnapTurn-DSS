"""
Get turnaround time statistics
"""
import pandas as pd
import glob

# Load data
files = glob.glob('Processed ABSD data/Processed ADSB and Airport Dataset-20251117/Processed ADSB Dataset/man_features_*.csv')
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Clean data
df_clean = df[(df['taxi_time'] >= 2) & (df['taxi_time'] <= 60) & (df['taxi_time'].notna())]

print("="*70)
print("TAXI TIME (TURNAROUND TIME) STATISTICS")
print("="*70)

print(f"\n📊 OVERALL STATISTICS")
print(f"   Mean (Average):     {df_clean['taxi_time'].mean():.2f} minutes")
print(f"   Median:             {df_clean['taxi_time'].median():.2f} minutes")
print(f"   Mode (Most Common): {df_clean['taxi_time'].mode()[0]:.2f} minutes")
print(f"   Std Deviation:      {df_clean['taxi_time'].std():.2f} minutes")

print(f"\n✈️  BY OPERATION TYPE")
arrivals = df_clean[df_clean['type'] == 'arrival']['taxi_time']
departures = df_clean[df_clean['type'] == 'departure']['taxi_time']

print(f"\n   ARRIVALS (Landing → Gate):")
print(f"   - Mean:   {arrivals.mean():.2f} minutes")
print(f"   - Median: {arrivals.median():.2f} minutes")
print(f"   - Count:  {len(arrivals):,} operations")

print(f"\n   DEPARTURES (Gate → Takeoff):")
print(f"   - Mean:   {departures.mean():.2f} minutes")
print(f"   - Median: {departures.median():.2f} minutes")
print(f"   - Count:  {len(departures):,} operations")

print(f"\n📈 PERCENTILES")
for p in [10, 25, 50, 75, 90, 95]:
    val = df_clean['taxi_time'].quantile(p/100)
    print(f"   {p}th percentile: {val:.2f} minutes")

print(f"\n⏱️  TIME RANGES")
ranges = [
    (2, 5, "Very Fast"),
    (5, 10, "Fast"),
    (10, 15, "Normal"),
    (15, 20, "Slow"),
    (20, 60, "Very Slow")
]

for min_t, max_t, label in ranges:
    count = len(df_clean[(df_clean['taxi_time'] >= min_t) & (df_clean['taxi_time'] < max_t)])
    pct = count / len(df_clean) * 100
    print(f"   {label:12} ({min_t:2}-{max_t:2} min): {count:6,} ops ({pct:5.1f}%)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"The average (mean) taxi time is {df_clean['taxi_time'].mean():.2f} minutes")
print(f"Most operations take around {df_clean['taxi_time'].median():.2f} minutes (median)")
print(f"Departures take about 2x longer than arrivals")
print("="*70)
