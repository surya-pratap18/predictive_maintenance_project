# src/airflow_dag_simulator.py
# Simulates Apache Airflow DAG behavior
# DAG: batch CSV ingestion → validation → 
#      feature engineering → write to DB

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("🔄 AIRFLOW DAG SIMULATOR")
print("  DAG: sensor_data_pipeline")
print("="*55)

# ============================================
# DATABASE CONNECTION
# ============================================
password = quote_plus("surya@2006")
engine = create_engine(
    f"postgresql://postgres:{password}@localhost:5432/predictive_maintenance"
)

# ============================================
# TASK 1: Data Ingestion
# ============================================
def task_ingest_data():
    print("\n📥 [TASK 1/4] Ingesting CSV data...")
    start = datetime.now()

    cols = ['engine_id', 'cycle'] + \
           [f'setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(
        'data/train_FD001.txt',
        sep='\s+',
        header=None,
        names=cols,
        engine='python'
    )
    df.dropna(axis=1, how='all', inplace=True)

    duration = (datetime.now() - start).seconds
    print(f"  ✅ Loaded {len(df)} rows in {duration}s")
    print(f"  📊 Shape: {df.shape}")
    return df

# ============================================
# TASK 2: Data Validation (Great Expectations)
# ============================================
def task_validate_data(df):
    print("\n✔️  [TASK 2/4] Validating data...")
    errors = []

    # Check 1: No missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        errors.append(f"Missing values found: {missing}")
    else:
        print("  ✅ No missing values")

    # Check 2: engine_id is positive
    if (df['engine_id'] <= 0).any():
        errors.append("Invalid engine_id found")
    else:
        print("  ✅ engine_id values valid")

    # Check 3: cycle is positive
    if (df['cycle'] <= 0).any():
        errors.append("Invalid cycle values found")
    else:
        print("  ✅ cycle values valid")

    # Check 4: Expected number of engines
    n_engines = df['engine_id'].nunique()
    if n_engines < 50:
        errors.append(f"Too few engines: {n_engines}")
    else:
        print(f"  ✅ Engine count valid: {n_engines}")

    # Check 5: No duplicate rows
    dupes = df.duplicated().sum()
    if dupes > 0:
        errors.append(f"Duplicate rows: {dupes}")
    else:
        print("  ✅ No duplicate rows")

    if errors:
        print(f"  ❌ Validation failed: {errors}")
        return False
    else:
        print("  ✅ All validations passed!")
        return True

# ============================================
# TASK 3: Feature Engineering
# ============================================
def task_feature_engineering(df):
    print("\n⚙️  [TASK 3/4] Engineering features...")

    # Useful sensors only
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    sensor_cols = [c for c in sensor_cols 
                   if c in df.columns and df[c].std() > 0.001]

    # Add RUL
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycle, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)

    # Rolling features (1h, 8h, 24h windows)
    print("  🔄 Computing rolling statistics...")
    for col in sensor_cols[:5]:  # top 5 sensors
        # Rolling mean
        df[f'{col}_roll_mean_5'] = df.groupby('engine_id')[col]\
            .transform(lambda x: x.rolling(5, min_periods=1).mean())

        # Rolling std
        df[f'{col}_roll_std_5'] = df.groupby('engine_id')[col]\
            .transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))

        # Rolling max
        df[f'{col}_roll_max_5'] = df.groupby('engine_id')[col]\
            .transform(lambda x: x.rolling(5, min_periods=1).max())

    # Lag features
    print("  🔄 Computing lag features...")
    for col in sensor_cols[:3]:  # top 3 sensors
        for lag in [1, 3, 5]:
            df[f'{col}_lag_{lag}'] = df.groupby('engine_id')[col]\
                .transform(lambda x: x.shift(lag).fillna(0))

    # Cross sensor ratios
    print("  🔄 Computing cross-sensor ratios...")
    if 'sensor_2' in df.columns and 'sensor_3' in df.columns:
        df['sensor_2_3_ratio'] = df['sensor_2'] / (df['sensor_3'] + 1e-8)

    if 'sensor_4' in df.columns and 'sensor_7' in df.columns:
        df['sensor_4_7_ratio'] = df['sensor_4'] / (df['sensor_7'] + 1e-8)

    # Failure label
    df['failure_label'] = (df['RUL'] < 30).astype(int)
    df['anomaly_score']  = 0.0

    print(f"  ✅ Features engineered!")
    print(f"  📊 Final shape: {df.shape}")
    print(f"  📊 New features added: {df.shape[1] - 26}")
    return df

# ============================================
# TASK 4: Write to Database
# ============================================
def task_write_to_db(df):
    print("\n💾 [TASK 4/4] Writing to database...")

    import datetime as dt

    # Add timestamp
    base_time = dt.datetime(2024, 1, 1)
    df['time'] = df.apply(
        lambda row: base_time + dt.timedelta(
            hours=int((row['engine_id']-1)*500 + row['cycle'])
        ), axis=1
    )

    # Rename
    df = df.rename(columns={'engine_id': 'equipment_id'})
    df['run_id'] = df['equipment_id']

    # Keep only DB columns
    db_cols = [
        'time', 'equipment_id', 'run_id', 'cycle',
        'setting_1', 'setting_2', 'setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
        'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
        'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16',
        'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
        'sensor_21', 'failure_label', 'rul', 'anomaly_score'
    ]

    # Add rul column
    df['rul'] = df['RUL']

    # Keep only existing columns
    db_cols = [c for c in db_cols if c in df.columns]
    df_db = df[db_cols]

    try:
        # Clear existing data
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE sensor_readings;"))
            conn.commit()

        # Insert new data
        df_db.to_sql(
            'sensor_readings',
            engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
        print(f"  ✅ {len(df_db)} rows written to database!")
    except Exception as e:
        print(f"  ❌ DB Error: {e}")

    return df

# ============================================
# RUN DAG
# ============================================
if __name__ == "__main__":
    dag_start = datetime.now()

    # Run all tasks in sequence
    df = task_ingest_data()

    is_valid = task_validate_data(df)
    if not is_valid:
        print("\n❌ DAG failed at validation step!")
        exit(1)

    df = task_feature_engineering(df)
    df = task_write_to_db(df)

    duration = (datetime.now() - dag_start).seconds
    print("\n" + "="*55)
    print(f"✅ DAG completed successfully in {duration}s!")
    print(f"📊 Final dataset shape: {df.shape}")
    print("="*55)