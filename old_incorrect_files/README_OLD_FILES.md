# Old/Incorrect Files - DO NOT USE

These files contain data quality issues and should NOT be used for any analysis or predictions.

## Why These Files Are Here

The original analysis had a critical data quality problem:
- **Predicted taxi times of 8-15 seconds** (impossible - aircraft can't taxi that fast!)
- **Negative taxi times** in the data
- **No data cleaning** was performed

While the model metrics looked good (R² = 0.928), it was accurately predicting garbage data.

## Files in This Folder

### Models (DO NOT USE)
- `real_turnaround_model.pkl` - Model trained on uncleaned data
- `turnaround_model.pkl` - Another old model version

### Scripts (DO NOT USE)
- `train_real_model.py` - Training script without data cleaning
- `analyze_real_data.py` - Analysis script for uncleaned data

### Results (INCORRECT)
- `data_summary.csv` - Summary showing 0.25 minute mean taxi time (wrong!)
- `feature_importance.csv` - Feature importance from bad model
- `real_data_analysis.png` - Visualization of uncleaned data
- `real_turnaround_prediction_results.png` - Results from bad model

## What to Use Instead

Use the files in the parent directory:

### ✅ Correct Models
- `proper_taxi_time_model.pkl` - Model trained on cleaned data

### ✅ Correct Scripts
- `process_and_train_proper.py` - Training with proper data cleaning
- `analyze_proper_data.py` - Analysis of cleaned data

### ✅ Correct Results
- `proper_feature_importance.csv` - Feature importance from good model
- `proper_data_analysis.png` - Visualization of cleaned data
- `proper_taxi_time_prediction_results.png` - Results from good model

## Key Differences

| Metric | Old (Bad) | New (Good) |
|--------|-----------|------------|
| Mean Taxi Time | 0.25 min (15 sec) ❌ | 9.60 min ✅ |
| Data Cleaning | None ❌ | Removed outliers ✅ |
| Realistic? | No ❌ | Yes ✅ |
| Usable? | No ❌ | Yes ✅ |

## Why Keep These Files?

These files are kept for:
1. **Reference** - To understand what went wrong
2. **Documentation** - To show the improvement made
3. **Learning** - To demonstrate importance of data quality

**DO NOT USE THESE FILES FOR ANY PREDICTIONS OR ANALYSIS!**

See `DATA_QUALITY_REPORT.md` in the parent directory for full details.
