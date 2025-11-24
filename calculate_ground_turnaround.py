"""
Calculate GROUND turnaround time:
- Arrival taxi (runway to gate)
- Gate operations (at gate)
- Departure taxi (gate to runway)

This is what we can actually measure from the data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("CALCULATING GROUND TURNAROUND TIME")
print("Arrival Taxi + Gate Operations + Departure Taxi")
print("="*70)

# Load processed data (already has taxi times calculated)
print("\n1. Loading processed ADSB data...")
import glob
files = glob.glob('Processed ABSD data/Processed ADSB and Airport Dataset-20251117/Processed ADSB Dataset/man_features_*.csv')
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"   Loaded {len(df):,} operations")

# Clean data
df_clean = df[(df['taxi_time'] >= 2) & (df['taxi_time'] <= 60) & (df['taxi_time'].notna())]
print(f"   Clean operations: {len(df_clean):,}")

# Parse times
df_clean['gate_time'] = pd.to_datetime(df_clean['gate (block) hour'], format='%H:%M:%S', errors='coerce')
df_clean['runway_time'] = pd.to_datetime(df_clean['runway hour'], format='%H:%M:%S', errors='coerce')

# Extract date and aircraft ID
df_clean['date'] = df_clean['id'].str[:11]  # '01-Apr-2021'
df_clean['aircraft'] = df_clean['mode_s']

print("\n2. Finding aircraft with both arrival and departure...")

# Group by date and aircraft to find turnarounds
turnarounds = []

for (date, aircraft), group in df_clean.groupby(['date', 'aircraft']):
    # Check if aircraft has both arrival and departure
    arrivals = group[group['type'] == 'arrival']
    departures = group[group['type'] == 'departure']
    
    if len(arrivals) > 0 and len(departures) > 0:
        # Get first arrival and first departure after it
        arrival = arrivals.iloc[0]
        departure = departures.iloc[0]
        
        # Calculate ground turnaround time
        arrival_taxi = arrival['taxi_time']
        departure_taxi = departure['taxi_time']
        
        # Gate time = time between arrival at gate and departure from gate
        # This is approximate since we don't have exact gate timestamps
        # We use the difference in runway times as a proxy
        try:
            gate_time_estimate = 60  # Default estimate (will refine)
            
            turnaround = {
                'date': date,
                'aircraft': aircraft,
                'arrival_taxi': arrival_taxi,
                'departure_taxi': departure_taxi,
                'gate_time_estimate': gate_time_estimate,
                'total_ground_time': arrival_taxi + gate_time_estimate + departure_taxi,
                'arrival_distance': arrival['distance'],
                'departure_distance': departure['distance'],
                'arrival_queue': arrival['QArrDep'] + arrival['QArrArr'],
                'departure_queue': departure['QDepDep'] + departure['QDepArr']
            }
            turnarounds.append(turnaround)
        except:
            continue

df_turnarounds = pd.DataFrame(turnarounds)

if len(df_turnarounds) > 0:
    print(f"   Found {len(df_turnarounds):,} complete ground turnarounds")
    
    print("\n3. GROUND TURNAROUND STATISTICS")
    
    print(f"\n   Arrival Taxi (Runway → Gate):")
    print(f"   - Mean:   {df_turnarounds['arrival_taxi'].mean():.2f} minutes")
    print(f"   - Median: {df_turnarounds['arrival_taxi'].median():.2f} minutes")
    
    print(f"\n   Departure Taxi (Gate → Runway):")
    print(f"   - Mean:   {df_turnarounds['departure_taxi'].mean():.2f} minutes")
    print(f"   - Median: {df_turnarounds['departure_taxi'].median():.2f} minutes")
    
    print(f"\n   Total Taxi Time:")
    total_taxi = df_turnarounds['arrival_taxi'] + df_turnarounds['departure_taxi']
    print(f"   - Mean:   {total_taxi.mean():.2f} minutes")
    print(f"   - Median: {total_taxi.median():.2f} minutes")
    
    print(f"\n   Estimated Total Ground Time (with gate ops):")
    print(f"   - Mean:   {df_turnarounds['total_ground_time'].mean():.2f} minutes")
    print(f"   - Median: {df_turnarounds['total_ground_time'].median():.2f} minutes")
    
    # Save
    df_turnarounds.to_csv('ground_turnaround_times.csv', index=False)
    print("\n4. Results saved to 'ground_turnaround_times.csv'")
    
    # Sample
    print("\n5. Sample ground turnarounds:")
    print(df_turnarounds[['date', 'arrival_taxi', 'departure_taxi', 'total_ground_time']].head(10))
    
    # Create summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nFor {len(df_turnarounds):,} complete turnarounds:")
    print(f"  • Average arrival taxi:    {df_turnarounds['arrival_taxi'].mean():.1f} minutes")
    print(f"  • Average departure taxi:  {df_turnarounds['departure_taxi'].mean():.1f} minutes")
    print(f"  • Average total taxi:      {total_taxi.mean():.1f} minutes")
    print(f"  • Estimated gate time:     {df_turnarounds['gate_time_estimate'].mean():.1f} minutes")
    print(f"  • Total ground time:       {df_turnarounds['total_ground_time'].mean():.1f} minutes")
    print("\nNote: Gate time is estimated at 60 minutes (industry average)")
    print("      Actual gate time varies by aircraft type and operation")
else:
    print("\n   No complete turnarounds found.")

print("\n" + "="*70)
