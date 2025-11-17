"""
Train Airport Turnaround Time Prediction Model with REAL ABSD Data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
import os

def load_all_data():
    """Load all monthly feature files"""
    data_path = 'Processed ABSD data/Processed ADSB and Airport Dataset-20251117/Processed ADSB Dataset/'
    files = glob.glob(os.path.join(data_path, 'man_features_*.csv'))
    
    print(f"Found {len(files)} data files")
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined_df)}")
    
    return combined_df

def prepare_features(df):
    """Prepare features for modeling"""
    # Create a copy
    data = df.copy()
    
    # Convert taxi_time from seconds to minutes
    data['turnaround_time_minutes'] = data['taxi_time'] / 60
    
    # Remove extreme outliers (likely data errors)
    data = data[data['turnaround_time_minutes'] > 0]
    data = data[data['turnaround_time_minutes'] < 180]  # Less than 3 hours
    
    # Encode operation mode (arrival/departure)
    data['is_departure'] = (data['type'] == 'departure').astype(int)
    
    # Extract hour from gate time
    data['hour'] = pd.to_datetime(data['gate (block) hour'], format='%H:%M:%S', errors='coerce').dt.hour
    
    # Fill missing values
    data['hour'] = data['hour'].fillna(12)  # Default to noon if missing
    data['distance'] = data['distance'].fillna(data['distance'].median())
    data['shortest path'] = data['shortest path'].fillna(data['shortest path'].median())
    data['distance_gate'] = data['distance_gate'].fillna(0)
    data['other_moving_ac'] = data['other_moving_ac'].fillna(0)
    
    # Create peak hour indicator
    data['is_peak_hour'] = data['hour'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)
    
    # Queue features
    data['total_queue'] = data['QDepDep'] + data['QDepArr'] + data['QArrDep'] + data['QArrArr']
    
    return data

def train_model(X, y):
    """Train the model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
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

def plot_results(y_test, y_pred, feature_importance_df, metrics):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Actual vs Predicted
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, y_pred, alpha=0.3, s=10)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Turnaround Time (min)', fontsize=10)
    ax1.set_ylabel('Predicted Turnaround Time (min)', fontsize=10)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.text(0.05, 0.95, f"R² = {metrics['test']['r2']:.3f}", 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residuals
    ax2 = plt.subplot(2, 3, 2)
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Turnaround Time (min)', fontsize=10)
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
    ax4.set_xlabel('Turnaround Time (min)', fontsize=10)
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
    
    Test Set Metrics:
    • R² Score:  {metrics['test']['r2']:.4f}
    • MAE:       {metrics['test']['mae']:.2f} minutes
    • RMSE:      {metrics['test']['rmse']:.2f} minutes
    
    Training Set Metrics:
    • R² Score:  {metrics['train']['r2']:.4f}
    • MAE:       {metrics['train']['mae']:.2f} minutes
    • RMSE:      {metrics['train']['rmse']:.2f} minutes
    
    Data Statistics:
    • Total Samples: {len(y_test) + len(y_pred)}
    • Test Samples:  {len(y_test)}
    • Features Used: {len(feature_importance_df)}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('real_turnaround_prediction_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to 'real_turnaround_prediction_results.png'")

def main():
    print("=" * 70)
    print("REAL Airport Turnaround Time Prediction Model")
    print("Using Actual Manchester Airport ABSD Data")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading real ABSD data...")
    df = load_all_data()
    
    # Prepare features
    print("\n2. Preparing features...")
    data = prepare_features(df)
    
    print(f"\nData after cleaning: {len(data)} records")
    print(f"Turnaround time range: {data['turnaround_time_minutes'].min():.1f} - {data['turnaround_time_minutes'].max():.1f} minutes")
    print(f"Mean turnaround time: {data['turnaround_time_minutes'].mean():.1f} minutes")
    print(f"Median turnaround time: {data['turnaround_time_minutes'].median():.1f} minutes")
    
    # Select features
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
    data_clean = data[feature_cols + ['turnaround_time_minutes']].dropna()
    
    X = data_clean[feature_cols]
    y = data_clean['turnaround_time_minutes']
    
    print(f"\nFinal dataset: {len(X)} records with {len(feature_cols)} features")
    
    # Train model
    print("\n3. Training model...")
    model, metrics, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    
    # Results
    print("\n4. Model Performance:")
    print(f"\n   Training Set:")
    print(f"   - MAE:  {metrics['train']['mae']:.2f} minutes")
    print(f"   - RMSE: {metrics['train']['rmse']:.2f} minutes")
    print(f"   - R²:   {metrics['train']['r2']:.4f}")
    
    print(f"\n   Test Set:")
    print(f"   - MAE:  {metrics['test']['mae']:.2f} minutes")
    print(f"   - RMSE: {metrics['test']['rmse']:.2f} minutes")
    print(f"   - R²:   {metrics['test']['r2']:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n5. Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model
    print("\n6. Saving model...")
    with open('real_turnaround_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics
        }, f)
    print("   Model saved to 'real_turnaround_model.pkl'")
    
    # Visualize
    print("\n7. Generating visualizations...")
    plot_results(y_test, y_pred, feature_importance, metrics)
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("   Feature importance saved to 'feature_importance.csv'")
    
    print("\n" + "=" * 70)
    print("REAL MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\nKey Insights:")
    print(f"• Model explains {metrics['test']['r2']*100:.1f}% of turnaround time variance")
    print(f"• Average prediction error: {metrics['test']['mae']:.1f} minutes")
    print(f"• Most important factor: {feature_importance.iloc[0]['feature']}")
    print("=" * 70)

if __name__ == '__main__':
    main()
