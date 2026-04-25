# src/drift_detection.py
# ============================================
# Data Drift Detection - Evidently 0.7.x
# ============================================

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported!")

# ============================================
# Load Data
# ============================================
def load_cmapss_data():
    cols = ['engine_id', 'cycle'] + \
           [f'setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(
        'data/train_FD001.txt',
        sep=r'\s+',
        header=None,
        names=cols,
        engine='python'
    )
    df.dropna(axis=1, how='all', inplace=True)

    max_cycle = df.groupby('engine_id')['cycle']\
                  .max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycle, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=125)
    df.drop(columns=['max_cycle'], inplace=True)
    df['failure_label'] = (
        df['RUL'] < 30
    ).astype(int)

    return df

# ============================================
# Get Feature Columns
# ============================================
def get_feature_cols(df):
    sensor_cols = [
        f'sensor_{i}' for i in range(1, 22)
        if f'sensor_{i}' in df.columns
        and df[f'sensor_{i}'].std() > 0.001
    ]
    setting_cols = [
        c for c in ['setting_1', 'setting_2']
        if c in df.columns
    ]
    return sensor_cols + setting_cols

# ============================================
# KS Test Drift Detection (No Evidently needed)
# ============================================
def run_ks_drift_detection(reference,
                            current,
                            feature_cols,
                            significance=0.05):
    """
    Run KS-test based drift detection
    Works without Evidently imports
    """
    print("🔄 Running KS-test drift detection...")

    column_drift = {}
    drifted_cols = []

    for col in feature_cols:
        try:
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) == 0 or \
               len(cur_vals) == 0:
                continue

            ks_stat, p_val = stats.ks_2samp(
                ref_vals, cur_vals
            )
            is_drifted = p_val < significance

            if is_drifted:
                drifted_cols.append(col)

            column_drift[col] = {
                'drifted'   : bool(is_drifted),
                'p_value'   : float(p_val),
                'ks_stat'   : float(ks_stat),
                'statistic' : 'KS-test',
                'ref_mean'  : float(
                    ref_vals.mean()
                ),
                'cur_mean'  : float(
                    cur_vals.mean()
                ),
                'mean_shift': float(
                    cur_vals.mean() -
                    ref_vals.mean()
                )
            }

        except Exception as e:
            print(f"  ⚠️  Skipping {col}: {e}")
            continue

    drift_share = (
        len(drifted_cols) / len(feature_cols)
        if feature_cols else 0
    )

    dataset_drifted = drift_share > 0.2
    tests_passed    = drift_share < 0.3

    print(
        f"   Drifted columns : "
        f"{len(drifted_cols)}/{len(feature_cols)}"
    )
    print(
        f"   Drift share     : "
        f"{drift_share:.2%}"
    )
    print(
        f"   Dataset drifted : {dataset_drifted}"
    )

    return {
        'generated_at'          : datetime.now()\
                                    .isoformat(),
        'method'                : 'KS-test',
        'significance_level'    : significance,
        'dataset_drifted'       : dataset_drifted,
        'drift_share'           : float(drift_share),
        'drifted_columns'       : drifted_cols,
        'tests_passed'          : tests_passed,
        'column_drift'          : column_drift,
        'reference_size'        : int(len(reference)),
        'current_size'          : int(len(current)),
        'retraining_recommended': dataset_drifted
    }

# ============================================
# Evidently Report (Optional - 0.7.x style)
# ============================================
def run_evidently_drift(reference,
                         current,
                         feature_cols):
    """
    Try Evidently 0.7.x style drift detection
    Falls back to KS-test if import fails
    """
    try:
        # Evidently 0.7.x imports
        from evidently.report import Report
        from evidently.metric_preset import (
            DataDriftPreset
        )

        print("🔄 Running Evidently drift report...")

        report = Report(metrics=[
            DataDriftPreset()
        ])

        report.run(
            reference_data=reference[feature_cols],
            current_data=current[feature_cols]
        )

        result = report.as_dict()

        # Extract metrics
        metrics   = result.get('metrics', [])
        drift_met = {}

        for m in metrics:
            if 'DatasetDriftMetric' in str(
                m.get('metric', '')
            ):
                drift_met = m.get('result', {})
                break

        is_drifted  = drift_met.get(
            'dataset_drift', False
        )
        drift_share = drift_met.get(
            'share_of_drifted_columns', 0.0
        )

        # Save HTML report
        os.makedirs(
            'data/drift_reports', exist_ok=True
        )
        report.save_html(
            'data/drift_reports/'
            'evidently_report.html'
        )
        print(
            "✅ Evidently HTML report saved!"
        )

        return is_drifted, float(drift_share)

    except ImportError as e:
        print(
            f"⚠️  Evidently preset import failed: {e}"
        )
        print("   Falling back to KS-test method")
        return None, None

    except Exception as e:
        print(f"⚠️  Evidently error: {e}")
        print("   Falling back to KS-test method")
        return None, None

# ============================================
# Target Drift Detection
# ============================================
def check_target_drift(reference,
                        current,
                        target_col='RUL'):
    """Check if RUL distribution has drifted"""
    try:
        ref_rul = reference[target_col].dropna()
        cur_rul = current[target_col].dropna()

        ks_stat, p_val = stats.ks_2samp(
            ref_rul, cur_rul
        )
        is_drifted = p_val < 0.05

        # Additional stats
        ref_mean = float(ref_rul.mean())
        cur_mean = float(cur_rul.mean())
        ref_std  = float(ref_rul.std())
        cur_std  = float(cur_rul.std())

        return {
            'target_drifted'  : bool(is_drifted),
            'ks_statistic'    : float(ks_stat),
            'p_value'         : float(p_val),
            'ref_mean_rul'    : ref_mean,
            'cur_mean_rul'    : cur_mean,
            'ref_std_rul'     : ref_std,
            'cur_std_rul'     : cur_std,
            'mean_shift'      : cur_mean - ref_mean
        }
    except Exception as e:
        print(f"⚠️  Target drift error: {e}")
        return {'target_drifted': False}

# ============================================
# Save Report
# ============================================
def save_drift_report(results):
    os.makedirs('data/drift_reports', exist_ok=True)

    timestamp = datetime.now().strftime(
        '%Y%m%d_%H%M%S'
    )
    filepath  = (
        f'data/drift_reports/'
        f'drift_report_{timestamp}.json'
    )

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    with open(
        'data/drift_reports/latest_drift.json',
        'w'
    ) as f:
        json.dump(results, f, indent=2)

    print(f"✅ Report saved: {filepath}")
    return filepath

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    print("="*55)
    print("🔍 DRIFT DETECTION PIPELINE")
    print("   Method: KS-Test + Evidently 0.7.x")
    print("="*55)

    # Load data
    df = load_cmapss_data()
    print(f"✅ Data loaded: {df.shape}")

    # Split reference / current
    split_idx = int(len(df) * 0.5)
    reference = df.iloc[:split_idx].copy()
    current   = df.iloc[split_idx:].copy()
    print(f"📊 Reference: {len(reference)} rows")
    print(f"📊 Current  : {len(current)} rows")

    # Feature columns
    feature_cols = get_feature_cols(df)
    print(f"📊 Features : {len(feature_cols)}")

    # Run KS-test drift detection
    results = run_ks_drift_detection(
        reference, current, feature_cols
    )

    # Try Evidently (optional)
    ev_drifted, ev_share = run_evidently_drift(
        reference, current, feature_cols
    )
    if ev_drifted is not None:
        results['evidently_drifted'] = ev_drifted
        results['evidently_share']   = ev_share
        print(
            f"✅ Evidently drift: {ev_drifted}"
        )

    # Check target drift
    target_drift = check_target_drift(
        reference, current
    )
    results['target_drift'] = target_drift
    print(
        f"📊 Target drifted: "
        f"{target_drift['target_drifted']}"
    )

    # Save report
    save_drift_report(results)

    # Print summary
    print("\n" + "="*55)
    print("📊 DRIFT DETECTION SUMMARY")
    print("="*55)
    print(
        f"  Dataset Drifted  : "
        f"{results['dataset_drifted']}"
    )
    print(
        f"  Drift Share      : "
        f"{results['drift_share']:.2%}"
    )
    print(
        f"  Tests Passed     : "
        f"{results['tests_passed']}"
    )
    print(
        f"  Target Drifted   : "
        f"{target_drift['target_drifted']}"
    )
    print(
        f"  Retraining Needed: "
        f"{results['retraining_recommended']}"
    )

    print("\n📊 Column Drift Results:")
    drifted = {
        k: v for k, v in
        results['column_drift'].items()
        if v['drifted']
    }
    stable  = {
        k: v for k, v in
        results['column_drift'].items()
        if not v['drifted']
    }

    print(f"\n  🔴 Drifted ({len(drifted)}):")
    for col, data in drifted.items():
        print(
            f"    {col:12s}: "
            f"p={data['p_value']:.4f} | "
            f"shift={data['mean_shift']:+.4f}"
        )

    print(f"\n  ✅ Stable ({len(stable)}):")
    for col, data in list(stable.items())[:5]:
        print(
            f"    {col:12s}: "
            f"p={data['p_value']:.4f}"
        )

    print("\n✅ Drift detection complete!")
    print(
        "📄 Report: "
        "data/drift_reports/latest_drift.json"
    )