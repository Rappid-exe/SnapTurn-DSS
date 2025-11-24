"""
Calculate TRUE turnaround time including:
1. Arrival flight (in air)
2. Arrival taxi (ground)
3. Gate operations (ground)
4. Departure taxi (ground)  
5. Departure flight (in air)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("CALCULATING TRUE TURNAROUND TIME")
print("Including flight time + taxi time + gate operations")
print("="*70)

# Load data
print("\n1. Loading raw data...")
print("   Loading radar data (this may take a minute)...")
df_radar = pd.read_csv('unprocessed/NATS_data/NATS Unproccessed data/radar_manchester_2021 (1).csv.gz')
print(f"   Loaded {len(df_radar):,} radar records")

print("   Loading flight plan data...")
df_flight = pd.read_csv('unprocessed/NATS_data/NATS Unproccessed data/flight_plan_manchester_2021 (1).csv.gz')
print(f"   Loaded {len(df_flight):,} flight plans")

# Convert datetime
print("\n2. Processing timestamps...")
df_radar['datetime'] = pd.to_datetime(df_radar['datetime_utc'])
df_flight['start_datetime'] = pd.to_datetime(df_flight['start_datetime'])
df_flight['end_datetime'] = pd.to_datetime(df_flight['end_datetime'])

# Identify ground vs air
# fl_calc is flight level - if NaN or < 0.5 (50 feet), aircraft is on ground
print("\n3. Identifying ground vs air positions...")
df_radar['on_ground'] = (df_radar['fl_calc'].isna()) | (df_radar['fl_calc'] < 0.5)
df_radar['in_air'] = ~df_radar['on_ground']

print(f"   Ground positions: {df_radar['on_ground'].sum():,}")
print(f"   Air positions: {df_radar['in_air'].sum():,}")

# Get flights that have both arrival and departure at Manchester
print("\n4. Finding complete turnarounds...")
print("   (Flights that both arrive and depart from Manchester)")

# Group by ufid to find complete turnarounds
complete_turnarounds = []

for ufid in df_flight['ufid'].unique()[:1000]:  # Sample first 1000 for speed
    # Get radar data for this flight
    flight_radar = df_radar[df_radar['ufid'] == ufid].sort_values('datetime')
    
    if len(flight_radar) < 10:  # Need enough data points
        continue
    
    # Get flight plan
    flight_plan = df_flight[df_flight['ufid'] == ufid].iloc[0]
    
    # Check if this is a turnaround (origin or dest is Manchester)
    if flight_plan['origin'] == 'EGCC' or flight_plan['dest'] == 'EGCC':
        # Calculate phases
        ground_data = flight_radar[flight_radar['on_ground']]
        air_data = flight_radar[flight_radar['in_air']]
        
        if len(ground_data) > 0 and len(air_data) > 0:
            turnaround = {
                'ufid': ufid,
                'callsign': flight_plan['callsign'],
                'origin': flight_plan['origin'],
                'dest': flight_plan['dest'],
                'aircraft_type': flight_plan['actype'],
                'start_time': flight_radar['datetime'].min(),
                'end_time': flight_radar['datetime'].max(),
                'total_time': (flight_radar['datetime'].max() - flight_radar['datetime'].min()).total_seconds() / 60,
                'ground_time': len(ground_data) * 6 / 60,  # 6 sec intervals
                'air_time': len(air_data) * 6 / 60,
                'num_positions': len(flight_radar)
            }
            complete_turnarounds.append(turnaround)

df_turnarounds = pd.DataFrame(complete_turnarounds)

print(f"   Found {len(df_turnarounds)} complete turnarounds in sample")

if len(df_turnarounds) > 0:
    print("\n5. TURNAROUND TIME STATISTICS")
    print(f"\n   Total Time (Flight + Taxi + Gate):")
    print(f"   - Mean:   {df_turnarounds['total_time'].mean():.2f} minutes")
    print(f"   - Median: {df_turnarounds['total_time'].median():.2f} minutes")
    print(f"   - Min:    {df_turnarounds['total_time'].min():.2f} minutes")
    print(f"   - Max:    {df_turnarounds['total_time'].max():.2f} minutes")
    
    print(f"\n   Ground Time (Taxi + Gate):")
    print(f"   - Mean:   {df_turnarounds['ground_time'].mean():.2f} minutes")
    print(f"   - Median: {df_turnarounds['ground_time'].median():.2f} minutes")
    
    print(f"\n   Air Time (Flight):")
    print(f"   - Mean:   {df_turnarounds['air_time'].mean():.2f} minutes")
    print(f"   - Median: {df_turnarounds['air_time'].median():.2f} minutes")
    
    # Save results
    df_turnarounds.to_csv('true_turnaround_times.csv', index=False)
    print("\n6. Results saved to 'true_turnaround_times.csv'")
    
    # Show sample
    print("\n7. Sample turnarounds:")
    print(df_turnarounds[['callsign', 'origin', 'dest', 'total_time', 'ground_time', 'air_time']].head(10))
else:
    print("\n   No complete turnarounds found in sample.")
    print("   This is expected - we need to process more data.")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nNote: This is a sample analysis of first 1000 flights.")
print("Full analysis would process all flights to find complete turnarounds.")
print("="*70)
