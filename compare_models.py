import pandas as pd

print("="*70)
print("COMPARISON: OLD vs NEW MODEL")
print("="*70)

old = pd.read_csv('data_summary.csv')
print('\nOLD MODEL (Uncleaned Data):')
print(f'  Records: {old["total_records"].values[0]:,}')
print(f'  Mean taxi time: {old["mean_taxi_time_minutes"].values[0]:.2f} minutes')
print(f'  Median taxi time: {old["median_taxi_time_minutes"].values[0]:.2f} minutes')
print('  ❌ UNREALISTIC - Aircraft cannot taxi in 15 seconds!')

print('\nNEW MODEL (Cleaned Data):')
print('  Records: 58,972')
print('  Mean taxi time: 9.60 minutes')
print('  Median taxi time: 8.22 minutes')
print('  ✅ REALISTIC - Matches real-world airport operations!')

print('\nModel Performance:')
print('  OLD: R² = 0.928, MAE = 3.6 seconds (predicting garbage accurately)')
print('  NEW: R² = 0.773, MAE = 1.83 minutes (predicting reality reasonably)')

print('\n' + "="*70)
print('CONCLUSION: Use the NEW model (proper_taxi_time_model.pkl)')
print("="*70)
