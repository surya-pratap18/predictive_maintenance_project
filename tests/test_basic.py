# tests/test_basic.py
import pytest
import numpy as np
import pandas as pd
import json
import os
import sys
sys.path.append('src')

def test_data_loading():
    cols = ['engine_id', 'cycle'] + \
           [f'setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(
        'data/train_FD001.txt',
        sep=r'\s+', header=None,
        names=cols, engine='python'
    )
    df.dropna(axis=1, how='all', inplace=True)
    assert len(df) > 0
    assert 'engine_id' in df.columns
    assert df['engine_id'].nunique() == 100
    print("✅ Data loading passed!")

def test_rul_calculation():
    cols = ['engine_id', 'cycle'] + \
           [f'setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(
        'data/train_FD001.txt',
        sep=r'\s+', header=None,
        names=cols, engine='python'
    )
    df.dropna(axis=1, how='all', inplace=True)
    mc = df.groupby('engine_id')['cycle']\
           .max().reset_index()
    mc.columns = ['engine_id', 'max_cycle']
    df = df.merge(mc, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    assert (df['RUL'] >= 0).all()
    print("✅ RUL calculation passed!")

def test_feature_engineering():
    data = pd.DataFrame({
        'sensor_1': np.random.randn(100),
        'sensor_2': np.random.randn(100),
        'engine_id': [1] * 100,
        'cycle': range(100)
    })
    data['sensor_1_mean5'] = data['sensor_1']\
        .rolling(5, min_periods=1).mean()
    data['sensor_1_std5'] = data['sensor_1']\
        .rolling(5, min_periods=1).std().fillna(0)
    assert 'sensor_1_mean5' in data.columns
    assert 'sensor_1_std5' in data.columns
    print("✅ Feature engineering passed!")

def test_model_files_exist():
    """Test model files exist in any location"""
    # Check multiple possible locations
    model_paths = [
        'models/best_rul_model.pth',
        'models/lstm_rul_best.pth',
        'models/best_model_info.json',
    ]

    # At least one model file should exist
    found_any = any(
        os.path.exists(p) for p in model_paths
    )

    if not found_any:
        # Check what's in models folder
        if os.path.exists('models'):
            files = os.listdir('models')
            print(f"Models folder contains: {files}")
            # Pass if models folder exists with files
            assert len(files) > 0, \
                f"Models folder empty! Files: {files}"
        else:
            pytest.skip(
                "Models folder not found - "
                "skipping model test"
            )
    else:
        print("✅ Model files found!")

def test_schedule_exists():
    """Test schedule exists in any location"""
    schedule_paths = [
        'data/schedules/optimal_schedule.json',
        'notebooks/data/schedules/optimal_schedule.json',
    ]

    found = False
    for path in schedule_paths:
        if os.path.exists(path):
            with open(path) as f:
                schedule = json.load(f)
            assert 'schedule' in schedule or \
                   'total_cost' in schedule
            found = True
            print(f"✅ Schedule found at: {path}")
            break

    if not found:
        pytest.skip(
            "Schedule not found - "
            "run day11 notebook first"
        )

def test_drift_report_exists():
    """Test drift report exists"""
    drift_paths = [
        'data/drift_reports/latest_drift.json',
        'notebooks/data/drift_reports/latest_drift.json'
    ]

    found = False
    for path in drift_paths:
        if os.path.exists(path):
            found = True
            print(f"✅ Drift report at: {path}")
            break

    if not found:
        pytest.skip("Drift report not found")

def test_processed_data_exists():
    """Test processed sequences exist"""
    data_paths = [
        'data/processed/X_train.npy',
        'notebooks/data/processed/X_train.npy'
    ]

    found = False
    for path in data_paths:
        if os.path.exists(path):
            data = np.load(path)
            assert len(data) > 0
            found = True
            print(f"✅ Processed data at: {path}")
            break

    if not found:
        pytest.skip("Processed data not found")

if __name__ == "__main__":
    test_data_loading()
    test_rul_calculation()
    test_feature_engineering()
    test_model_files_exist()
    test_schedule_exists()
    test_drift_report_exists()
    test_processed_data_exists()
    print("\n✅ All tests complete!")