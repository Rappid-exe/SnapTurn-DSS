"""
Verify project structure and files are correct
"""
import os
import pickle

print("="*70)
print("PROJECT VERIFICATION")
print("="*70)

# Check essential files exist
essential_files = [
    'proper_taxi_time_model.pkl',
    'process_and_train_proper.py',
    'analyze_proper_data.py',
    'proper_feature_importance.csv',
    'README.md',
    'QUICK_START.md'
]

print("\n1. CHECKING ESSENTIAL FILES")
all_good = True
for file in essential_files:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"   {status} {file}")
    if not exists:
        all_good = False

# Check old files are archived
old_files = [
    'real_turnaround_model.pkl',
    'train_real_model.py',
    'analyze_real_data.py'
]

print("\n2. CHECKING OLD FILES ARE ARCHIVED")
for file in old_files:
    in_main = os.path.exists(file)
    in_archive = os.path.exists(f'old_incorrect_files/{file}')
    
    if in_main:
        print(f"   ❌ {file} - Still in main directory (should be archived)")
        all_good = False
    elif in_archive:
        print(f"   ✅ {file} - Correctly archived")
    else:
        print(f"   ⚠️  {file} - Not found (may have been deleted)")

# Check model loads correctly
print("\n3. CHECKING MODEL")
try:
    with open('proper_taxi_time_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        feature_cols = model_data['feature_cols']
    print(f"   ✅ Model loads successfully")
    print(f"   ✅ Features: {len(feature_cols)} features")
    print(f"   ✅ Model type: {type(model).__name__}")
except Exception as e:
    print(f"   ❌ Model failed to load: {e}")
    all_good = False

# Check documentation
print("\n4. CHECKING DOCUMENTATION")
docs = [
    'QUICK_START.md',
    'README.md',
    'DATA_QUALITY_REPORT.md'
]

for doc in docs:
    exists = os.path.exists(doc)
    status = "✅" if exists else "❌"
    print(f"   {status} {doc}")
    if not exists:
        all_good = False

# Check data directory
print("\n5. CHECKING DATA DIRECTORY")
data_path = 'Processed ABSD data/Processed ADSB and Airport Dataset-20251117/Processed ADSB Dataset/'
if os.path.exists(data_path):
    files = [f for f in os.listdir(data_path) if f.startswith('man_features_')]
    print(f"   ✅ Data directory exists")
    print(f"   ✅ Found {len(files)} monthly data files")
else:
    print(f"   ❌ Data directory not found")
    all_good = False

# Final verdict
print("\n" + "="*70)
if all_good:
    print("✅ PROJECT VERIFICATION PASSED")
    print("="*70)
    print("\nAll files are in correct locations!")
    print("You can start using the project.")
    print("\nNext steps:")
    print("  1. Read QUICK_START.md or README.md")
    print("  2. Run: python compare_models.py")
    print("  3. Run: python process_and_train_proper.py")
else:
    print("⚠️  PROJECT VERIFICATION FAILED")
    print("="*70)
    print("\nSome files are missing or in wrong locations.")
    print("Please check the messages above.")

print()
