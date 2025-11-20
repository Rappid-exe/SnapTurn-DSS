"""
Proper Airport Taxi Time Prediction Model
Process raw ADSB data and train a realistic model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
import os

def load_and_clean_data():
    """Load all monthly feature files and clean the data"""
    data_path = 'Processed ABSD data/Processed ADSB and Airport Dataset-20251117/Processed ADSB Dataset/'
    files = glob.glob(os.path.join(data_path, 'man_features_*.csv'))
    
    print(f"Found {len(files)} data files")
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records loaded: {len(combined_df):,}")
    
    # CRITICAL: Clean the data properly
    print("\n" + "="*70)
    print("DATA CLEANING")
    print("="*70)
    
    # Remove records with missing taxi_time
    initial_count = len(combined_df)
    combined_df = combined_df.dropna(subset=['taxi_time'])
    print(f"Removed {initial_count - len(combined_df):,} records with missing taxi_time")
    
    # Remove negative taxi times (data errors)
    negative_count = len(combined_df[combined_df['taxi_time'] < 0])
    combined_df = combined_df[combined_df['taxi_time'] >= 0]
    print(f"Removed {negative_count:,} records with negative taxi_time")
    
    # Remove unrealistically short taxi times (< 2 minutes)
    # Aircraft can't taxi from gate to runway in under 2 minutes
    too_short = len(combined_df[combined_df['taxi_time'] < 2])
    combined_df = combined_df[combined_df['taxi_time'] >= 2]
    print(f"Removed {too_short:,} records with taxi_time < 2 minutes (likely incomplete data)")
    
    # Remove unrealistically long taxi times (> 60 minutes)
    # Normal taxi operations don't take more than 1 hour
    too_long = len(combined_df[combined_df['taxi_time'] > 60])
    combined_df = combined_df[combined_df['taxi_time'] <= 60]
    print(f"Removed {too_long:,} records with taxi_time > 60 minutes (likely data errors)")
    
    print(f"\nFinal clean dataset: {len(combined_df):,} records")
    print(f"Retention rate: {len(combined_df)/initial_count*100:.1f}%")
    
    return combined_df

def prepare_features(df):
    """Prepare features for modeling"""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    data = df.copy()
    
    # Target variable (already in minutes)
    data['taxi_time_minutes'] = data['taxi_time']
    
    # Encode operation type
    data['is_departure'] = (data['type'] == 'departure').astype(int)
    
    # Extract hour from gate time
    data['hour'] = pd.to_datetime(data['gate (block) hour'], format='%H:%M:%S', errors='coerce').dt.hour
    data['hour'] = data['hour'].fillna(12)  # Default to noon if missing
    
    # Create peak hour indicator (morning and evening rush)
    data['is_peak_hour'] = data['hour'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)
    
    # Total queue size
    data['total_queue'] = data['QDepDep'] + data['QDepArr'] + data['QArrDep'] + data['QArrArr']
    
    # Fill missing distance values
    for col in ['distance', 'shortest path', 'distance_gate', 'distance_long', 'distance_else']:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # Fill missing traffic values
    data['other_moving_ac'] = data['other_moving_ac'].fillna(0)
    
    print(f"Features prepared for {len(data):,} records")
    
    return data

def analyze_cleaned_data(df):
    """Analyze the cleaned dataset"""
    print("\n" + "="*70)
    print("CLEANED DATA ANALYSIS")
    print("="*70)
    
    print(f"\n1. DATASET OVERVIEW")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique aircraft: {df['mode_s'].nunique():,}")
    
    print(f"\n2. OPERATION TYPES")
    print(df['type'].value_counts())
    
    print(f"\n3. TAXI TIME STATISTICS (minutes)")
    print(f"   Mean:   {df['taxi_time'].mean():.2f} minutes")
    print(f"   Median: {df['taxi_time'].median():.2f} minutes")
    print(f"   Std:    {df['taxi_time'].std():.2f} minutes")
    print(f"   Min:    {df['taxi_time'].min():.2f} minutes")
    print(f"   Max:    {df['taxi_time'].max():.2f} minutes")
    
    print(f"\n   Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = df['taxi_time'].quantile(p/100)
        print(f"   {p}th: {val:.2f} minutes")
    
    print(f"\n4. TAXI TIME BY OPERATION TYPE")
    for op_type in df['type'].unique():
        if pd.notna(op_type):
            subset = df[df['type'] == op_type]['taxi_time']
            print(f"\n   {op_type.upper()}:")
            print(f"   - Count:  {len(subset):,}")
            print(f"   - Mean:   {subset.mean():.2f} min")
            print(f"   - Median: {subset.median():.2f} min")
    
    print(f"\n5. DISTANCE ANALYSIS (meters)")
    print(f"   Mean distance:   {df['distance'].mean():.2f} m")
    print(f"   Median distance: {df['distance'].median():.2f} m")
    if 'shortest path' in df.columns:
        print(f"   Mean shortest path: {df['shortest path'].mean():.2f} m")
    
    print(f"\n6. TRAFFIC ANALYSIS")
    print(f"   Mean other moving aircraft: {df['other_moving_ac'].mean():.2f}")
    total_queue = df['QDepDep'] + df['QDepArr'] + df['QArrDep'] + df['QArrArr']
    print(f"   Mean total queue: {total_queue.mean():.2f}")
    
    print(f"\n7. RUNWAY USAGE")
    print(df['rwy'].value_counts().head(5))

def train_model(X, y):
    """Train the Random Forest model"""
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set:     {len(X_test):,} samples")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
    }
    
    return model, metrics, X_train, X_test, y_train, y_test, y_pred_test

def plot_results(y_test, y_pred, feature_importance_df, metrics, data_stats):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Actual vs Predicted
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, y_pred, alpha=0.3, s=10)
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', lw=2)
    ax1.set_xlabel('Actual Taxi Time (min)', fontsize=10)
    ax1.set_ylabel('Predicted Taxi Time (min)', fontsize=10)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.text(0.05, 0.95, f"R² = {metrics['test']['r2']:.3f}", 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residuals
    ax2 = plt.subplot(2, 3, 2)
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Taxi Time (min)', fontsize=10)
    ax2.set_ylabel('Residuals (min)', fontsize=10)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    
    # 3. Feature Importance
    ax3 = plt.subplot(2, 3, 3)
    top_features = feature_importance_df.head(10)
    ax3.barh(range(len(top_features)), top_features['importance'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'], fontsize=9)
    ax3.set_xlabel('Importance', fontsize=10)
    ax3.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. Distribution comparison
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(y_test, bins=50, alpha=0.5, label='Actual', density=True)
    ax4.hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
    ax4.set_xlabel('Taxi Time (min)', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    
    # 5. Error distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Prediction Error (min)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax5.text(0.05, 0.95, f"MAE = {metrics['test']['mae']:.2f} min\nRMSE = {metrics['test']['rmse']:.2f} min", 
             transform=ax5.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 6. Performance metrics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    MODEL PERFORMANCE SUMMARY
    ========================
    
    Dataset: Manchester Airport (Apr-Oct 2021)
    Records: {data_stats['total_records']:,}
    Aircraft: {data_stats['unique_aircraft']:,}
    
    Test Set Metrics:
    • R² Score:  {metrics['test']['r2']:.4f}
    • MAE:       {metrics['test']['mae']:.2f} minutes
    • RMSE:      {metrics['test']['rmse']:.2f} minutes
    
    Training Set Metrics:
    • R² Score:  {metrics['train']['r2']:.4f}
    • MAE:       {metrics['train']['mae']:.2f} minutes
    • RMSE:      {metrics['train']['rmse']:.2f} minutes
    
    Taxi Time Range: {data_stats['min_taxi']:.1f} - {data_stats['max_taxi']:.1f} min
    Mean Taxi Time: {data_stats['mean_taxi']:.1f} min
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('proper_taxi_time_prediction_results.png', dpi=300, bbox_inches='tight')
    print("\n   Visualization saved to 'proper_taxi_time_prediction_results.png'")

def main():
    print("="*70)
    print("PROPER AIRPORT TAXI TIME PREDICTION MODEL")
    print("Manchester Airport ADSB Data (April-October 2021)")
    print("="*70)
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Analyze cleaned data
    analyze_cleaned_data(df)
    
    # Prepare features
    data = prepare_features(df)
    
    # Select features for modeling
    feature_cols = [
        'distance',
        'shortest path',
        'distance_gate',
        'distance_long',
        'distance_else',
        'other_moving_ac',
        'is_departure',
        'hour',
        'is_peak_hour',
        'QDepDep',
        'QDepArr',
        'QArrDep',
        'QArrArr',
        'total_queue',
        'NDepDep',
        'NDepArr',
        'NArrDep',
        'NArrArr'
    ]
    
    # Remove any rows with NaN in selected features
    data_clean = data[feature_cols + ['taxi_time_minutes']].dropna()
    
    X = data_clean[feature_cols]
    y = data_clean['taxi_time_minutes']
    
    print(f"\nFinal modeling dataset: {len(X):,} records with {len(feature_cols)} features")
    
    # Train model
    model, metrics, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    
    # Results
    print("\n" + "="*70)
    print("MODEL PERFORMANCE")
    print("="*70)
    
    print(f"\nTraining Set:")
    print(f"  MAE:  {metrics['train']['mae']:.2f} minutes")
    print(f"  RMSE: {metrics['train']['rmse']:.2f} minutes")
    print(f"  R²:   {metrics['train']['r2']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  MAE:  {metrics['test']['mae']:.2f} minutes")
    print(f"  RMSE: {metrics['test']['rmse']:.2f} minutes")
    print(f"  R²:   {metrics['test']['r2']:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)
    print(feature_importance.to_string(index=False))
    
    # Save model
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    with open('proper_taxi_time_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }, f)
    print("✓ Model saved to 'proper_taxi_time_model.pkl'")
    
    # Save feature importance
    feature_importance.to_csv('proper_feature_importance.csv', index=False)
    print("✓ Feature importance saved to 'proper_feature_importance.csv'")
    
    # Save data summary
    data_stats = {
        'total_records': len(data_clean),
        'unique_aircraft': df['mode_s'].nunique(),
        'mean_taxi': y.mean(),
        'min_taxi': y.min(),
        'max_taxi': y.max()
    }
    
    # Visualize
    plot_results(y_test, y_pred, feature_importance, metrics, data_stats)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nKey Insights:")
    print(f"• Model explains {metrics['test']['r2']*100:.1f}% of taxi time variance")
    print(f"• Average prediction error: {metrics['test']['mae']:.1f} minutes")
    print(f"• Most important factor: {feature_importance.iloc[0]['feature']}")
    print(f"• Dataset: {len(data_clean):,} clean taxi operations")
    print("="*70)

if __name__ == '__main__':
    main()
