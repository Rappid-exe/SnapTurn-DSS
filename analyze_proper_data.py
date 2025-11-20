"""
Analyze the properly cleaned ADSB data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def load_and_clean_data():
    """Load all monthly feature files and clean the data"""
    data_path = 'Processed ABSD data/Processed ADSB and Airport Dataset-20251117/Processed ADSB Dataset/'
    files = glob.glob(os.path.join(data_path, 'man_features_*.csv'))
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Clean the data
    combined_df = combined_df.dropna(subset=['taxi_time'])
    combined_df = combined_df[combined_df['taxi_time'] >= 2]  # Min 2 minutes
    combined_df = combined_df[combined_df['taxi_time'] <= 60]  # Max 60 minutes
    
    return combined_df

def create_visualizations(df):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Taxi time distribution
    axes[0, 0].hist(df['taxi_time'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Taxi Time (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Taxi Time Distribution (Cleaned Data)')
    axes[0, 0].axvline(df['taxi_time'].mean(), color='red', linestyle='--', label=f'Mean: {df["taxi_time"].mean():.1f} min')
    axes[0, 0].legend()
    
    # 2. Taxi time by operation type
    for op_type in df['type'].dropna().unique():
        subset = df[df['type'] == op_type]['taxi_time']
        axes[0, 1].hist(subset, bins=40, alpha=0.6, label=op_type)
    axes[0, 1].set_xlabel('Taxi Time (minutes)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Taxi Time by Operation Type')
    axes[0, 1].legend()
    
    # 3. Distance vs Taxi Time
    axes[0, 2].scatter(df['distance'], df['taxi_time'], alpha=0.1, s=5)
    axes[0, 2].set_xlabel('Distance (meters)')
    axes[0, 2].set_ylabel('Taxi Time (minutes)')
    axes[0, 2].set_title('Distance vs Taxi Time')
    
    # 4. Queue vs Taxi Time
    df['total_queue'] = df['QDepDep'] + df['QDepArr'] + df['QArrDep'] + df['QArrArr']
    axes[1, 0].scatter(df['total_queue'], df['taxi_time'], alpha=0.1, s=5)
    axes[1, 0].set_xlabel('Total Queue Size')
    axes[1, 0].set_ylabel('Taxi Time (minutes)')
    axes[1, 0].set_title('Queue Size vs Taxi Time (Strong Correlation)')
    
    # 5. Hour of day distribution
    df['hour'] = pd.to_datetime(df['gate (block) hour'], format='%H:%M:%S', errors='coerce').dt.hour
    axes[1, 1].hist(df['hour'].dropna(), bins=24, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Number of Operations')
    axes[1, 1].set_title('Operations by Hour of Day')
    axes[1, 1].set_xticks(range(0, 24, 2))
    
    # 6. Taxi time by hour
    hourly_mean = df.groupby('hour')['taxi_time'].mean()
    axes[1, 2].plot(hourly_mean.index, hourly_mean.values, marker='o', linewidth=2, markersize=6)
    axes[1, 2].set_xlabel('Hour of Day')
    axes[1, 2].set_ylabel('Mean Taxi Time (minutes)')
    axes[1, 2].set_title('Mean Taxi Time by Hour')
    axes[1, 2].set_xticks(range(0, 24, 2))
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Other moving aircraft distribution
    axes[2, 0].hist(df['other_moving_ac'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[2, 0].set_xlabel('Number of Other Moving Aircraft')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Traffic Density Distribution')
    
    # 8. Runway usage
    runway_counts = df['rwy'].value_counts().head(5)
    axes[2, 1].barh(range(len(runway_counts)), runway_counts.values, color='purple')
    axes[2, 1].set_yticks(range(len(runway_counts)))
    axes[2, 1].set_yticklabels(runway_counts.index)
    axes[2, 1].set_xlabel('Number of Operations')
    axes[2, 1].set_title('Top 5 Runway Usage')
    axes[2, 1].invert_yaxis()
    
    # 9. Box plot by operation type
    arrival_data = df[df['type'] == 'arrival']['taxi_time']
    departure_data = df[df['type'] == 'departure']['taxi_time']
    axes[2, 2].boxplot([arrival_data, departure_data], labels=['Arrival', 'Departure'])
    axes[2, 2].set_ylabel('Taxi Time (minutes)')
    axes[2, 2].set_title('Taxi Time Distribution by Type')
    axes[2, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('proper_data_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to 'proper_data_analysis.png'")

def main():
    print("="*70)
    print("PROPER DATA ANALYSIS")
    print("="*70)
    
    df = load_and_clean_data()
    
    print(f"\nDataset: {len(df):,} clean taxi operations")
    print(f"Mean taxi time: {df['taxi_time'].mean():.2f} minutes")
    print(f"Median taxi time: {df['taxi_time'].median():.2f} minutes")
    print(f"Std dev: {df['taxi_time'].std():.2f} minutes")
    
    create_visualizations(df)
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
