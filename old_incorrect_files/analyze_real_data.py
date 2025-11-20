"""
Analyze the real ABSD data to understand turnaround patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def load_all_data():
    """Load all monthly feature files"""
    data_path = 'Processed ABSD data/Processed ADSB and Airport Dataset-20251117/Processed ADSB Dataset/'
    files = glob.glob(os.path.join(data_path, 'man_features_*.csv'))
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def analyze_data(df):
    """Comprehensive data analysis"""
    print("=" * 70)
    print("REAL ABSD DATA ANALYSIS")
    print("Manchester Airport - January 2021")
    print("=" * 70)
    
    # Basic statistics
    print(f"\n1. DATASET OVERVIEW")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique aircraft: {df['mode_s'].nunique()}")
    print(f"   Date range: {df['id'].str[:11].min()} to {df['id'].str[:11].max()}")
    
    # Operation types
    print(f"\n2. OPERATION TYPES")
    print(df['type'].value_counts())
    
    # Taxi time analysis (in seconds)
    print(f"\n3. TAXI TIME ANALYSIS (seconds)")
    print(f"   Mean:   {df['taxi_time'].mean():.2f} seconds ({df['taxi_time'].mean()/60:.2f} minutes)")
    print(f"   Median: {df['taxi_time'].median():.2f} seconds ({df['taxi_time'].median()/60:.2f} minutes)")
    print(f"   Std:    {df['taxi_time'].std():.2f} seconds")
    print(f"   Min:    {df['taxi_time'].min():.2f} seconds")
    print(f"   Max:    {df['taxi_time'].max():.2f} seconds ({df['taxi_time'].max()/60:.2f} minutes)")
    
    # Percentiles
    print(f"\n   Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = df['taxi_time'].quantile(p/100)
        print(f"   {p}th: {val:.2f} seconds ({val/60:.2f} minutes)")
    
    # By operation type
    print(f"\n4. TAXI TIME BY OPERATION TYPE")
    for op_type in df['type'].unique():
        if pd.notna(op_type):
            subset = df[df['type'] == op_type]['taxi_time']
            print(f"\n   {op_type.upper()}:")
            print(f"   - Count:  {len(subset):,}")
            print(f"   - Mean:   {subset.mean():.2f} sec ({subset.mean()/60:.2f} min)")
            print(f"   - Median: {subset.median():.2f} sec ({subset.median()/60:.2f} min)")
    
    # Distance analysis
    print(f"\n5. DISTANCE ANALYSIS (meters)")
    print(f"   Mean distance:   {df['distance'].mean():.2f} m")
    print(f"   Median distance: {df['distance'].median():.2f} m")
    print(f"   Mean shortest path: {df['shortest path'].mean():.2f} m")
    
    # Queue analysis
    print(f"\n6. QUEUE/TRAFFIC ANALYSIS")
    print(f"   Mean other moving aircraft: {df['other_moving_ac'].mean():.2f}")
    print(f"   Mean total queue (all types): {(df['QDepDep'] + df['QDepArr'] + df['QArrDep'] + df['QArrArr']).mean():.2f}")
    
    # Runway usage
    print(f"\n7. RUNWAY USAGE")
    runway_counts = df['rwy'].value_counts()
    print(runway_counts.head(10))
    
    # Create visualizations
    create_visualizations(df)
    
    # Identify realistic turnaround times
    print(f"\n8. REALISTIC TURNAROUND TIME FILTER")
    # Filter for reasonable taxi times (5 minutes to 60 minutes)
    realistic = df[(df['taxi_time'] >= 300) & (df['taxi_time'] <= 3600)]
    print(f"   Records with 5-60 min taxi time: {len(realistic):,} ({len(realistic)/len(df)*100:.1f}%)")
    if len(realistic) > 0:
        print(f"   Mean: {realistic['taxi_time'].mean()/60:.2f} minutes")
        print(f"   Median: {realistic['taxi_time'].median()/60:.2f} minutes")
    
    return df

def create_visualizations(df):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Taxi time distribution
    axes[0, 0].hist(df['taxi_time']/60, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Taxi Time (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Taxi Time Distribution (All Data)')
    axes[0, 0].set_xlim(0, 60)
    
    # 2. Taxi time by operation type
    for op_type in df['type'].dropna().unique():
        subset = df[df['type'] == op_type]['taxi_time'] / 60
        axes[0, 1].hist(subset, bins=50, alpha=0.5, label=op_type)
    axes[0, 1].set_xlabel('Taxi Time (minutes)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Taxi Time by Operation Type')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 30)
    
    # 3. Distance vs Taxi Time
    axes[0, 2].scatter(df['distance'], df['taxi_time']/60, alpha=0.1, s=5)
    axes[0, 2].set_xlabel('Distance (meters)')
    axes[0, 2].set_ylabel('Taxi Time (minutes)')
    axes[0, 2].set_title('Distance vs Taxi Time')
    axes[0, 2].set_ylim(0, 30)
    
    # 4. Queue vs Taxi Time
    df['total_queue'] = df['QDepDep'] + df['QDepArr'] + df['QArrDep'] + df['QArrArr']
    axes[1, 0].scatter(df['total_queue'], df['taxi_time']/60, alpha=0.1, s=5)
    axes[1, 0].set_xlabel('Total Queue Size')
    axes[1, 0].set_ylabel('Taxi Time (minutes)')
    axes[1, 0].set_title('Queue Size vs Taxi Time')
    axes[1, 0].set_ylim(0, 30)
    
    # 5. Hour of day distribution
    df['hour'] = pd.to_datetime(df['gate (block) hour'], format='%H:%M:%S', errors='coerce').dt.hour
    axes[1, 1].hist(df['hour'].dropna(), bins=24, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Number of Operations')
    axes[1, 1].set_title('Operations by Hour of Day')
    axes[1, 1].set_xticks(range(0, 24, 2))
    
    # 6. Taxi time by hour
    hourly_mean = df.groupby('hour')['taxi_time'].mean() / 60
    axes[1, 2].plot(hourly_mean.index, hourly_mean.values, marker='o')
    axes[1, 2].set_xlabel('Hour of Day')
    axes[1, 2].set_ylabel('Mean Taxi Time (minutes)')
    axes[1, 2].set_title('Mean Taxi Time by Hour')
    axes[1, 2].set_xticks(range(0, 24, 2))
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Other moving aircraft distribution
    axes[2, 0].hist(df['other_moving_ac'], bins=50, edgecolor='black', alpha=0.7)
    axes[2, 0].set_xlabel('Number of Other Moving Aircraft')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Traffic Density Distribution')
    
    # 8. Runway usage
    runway_counts = df['rwy'].value_counts().head(10)
    axes[2, 1].barh(range(len(runway_counts)), runway_counts.values)
    axes[2, 1].set_yticks(range(len(runway_counts)))
    axes[2, 1].set_yticklabels(runway_counts.index)
    axes[2, 1].set_xlabel('Number of Operations')
    axes[2, 1].set_title('Top 10 Runway Usage')
    axes[2, 1].invert_yaxis()
    
    # 9. Realistic turnaround times (5-60 minutes)
    realistic = df[(df['taxi_time'] >= 300) & (df['taxi_time'] <= 3600)]
    if len(realistic) > 0:
        axes[2, 2].hist(realistic['taxi_time']/60, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[2, 2].set_xlabel('Taxi Time (minutes)')
        axes[2, 2].set_ylabel('Frequency')
        axes[2, 2].set_title(f'Realistic Turnaround Times (n={len(realistic):,})')
    
    plt.tight_layout()
    plt.savefig('real_data_analysis.png', dpi=300, bbox_inches='tight')
    print("\n   Visualizations saved to 'real_data_analysis.png'")

def main():
    # Load data
    df = load_all_data()
    
    # Analyze
    analyze_data(df)
    
    # Save summary
    summary = {
        'total_records': len(df),
        'unique_aircraft': df['mode_s'].nunique(),
        'mean_taxi_time_minutes': df['taxi_time'].mean() / 60,
        'median_taxi_time_minutes': df['taxi_time'].median() / 60,
        'arrivals': len(df[df['type'] == 'arrival']),
        'departures': len(df[df['type'] == 'departure'])
    }
    
    pd.DataFrame([summary]).to_csv('data_summary.csv', index=False)
    print("\n   Summary saved to 'data_summary.csv'")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)

if __name__ == '__main__':
    main()
