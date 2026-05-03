# tests/performance_test.py
# ============================================
# Performance & Stress Testing
# Without Locust (pure Python)
# ============================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import json
import os
import pickle
import threading
import concurrent.futures
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("⚡ PERFORMANCE & STRESS TESTING")
print("="*55)

# ============================================
# Model Setup
# ============================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )
        context = (
            attn_weights * lstm_out
        ).sum(dim=1)
        return context, attn_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim=64,
                 num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = AttentionLayer(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _        = self.lstm(x)
        out           = self.dropout(out)
        context, attn = self.attention(out)
        return self.fc(context).squeeze(-1), attn

# ============================================
# TEST 1: Data Loading Performance
# ============================================
def test_data_loading_performance(n_runs=10):
    print("\n📊 TEST 1: Data Loading Performance")
    print("-"*40)

    times = []
    for i in range(n_runs):
        start = time.time()

        cols = ['engine_id', 'cycle'] + \
               [f'setting_{i}' for i in range(1, 4)] + \
               [f'sensor_{i}' for i in range(1, 22)]

        df = pd.read_csv(
            'data/train_FD001.txt',
            sep=r'\s+', header=None,
            names=cols, engine='python'
        )
        df.dropna(axis=1, how='all', inplace=True)

        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"  Runs          : {n_runs}")
    print(f"  Avg load time : {avg_time:.2f} ms")
    print(f"  Min load time : {min_time:.2f} ms")
    print(f"  Max load time : {max_time:.2f} ms")

    status = "✅ PASS" if avg_time < 2000 else "❌ FAIL"
    print(f"  Status        : {status} (target < 2000ms)")

    return {
        'test'    : 'data_loading',
        'avg_ms'  : round(avg_time, 2),
        'min_ms'  : round(min_time, 2),
        'max_ms'  : round(max_time, 2),
        'passed'  : avg_time < 2000
    }

# ============================================
# TEST 2: Model Inference Performance
# ============================================
def test_inference_performance(n_requests=100):
    print("\n🤖 TEST 2: Model Inference Performance")
    print("-"*40)

    try:
        # Load model
        with open(
            'data/processed/feature_cols.pkl', 'rb'
        ) as f:
            feature_cols = pickle.load(f)

        with open(
            'models/best_model_info.json'
        ) as f:
            model_info = json.load(f)

        input_dim = model_info['input_dim']
        model     = LSTMWithAttention(
            input_dim=input_dim,
            hidden_dim=model_info['hidden_dim'],
            num_layers=model_info['num_layers'],
            dropout=model_info['dropout']
        )
        model.load_state_dict(torch.load(
            'models/best_rul_model.pth',
            map_location='cpu'
        ))
        model.eval()

        # Single inference test
        times_single = []
        for _ in range(n_requests):
            start    = time.time()
            x        = torch.randn(1, 30, input_dim)
            with torch.no_grad():
                pred, _ = model(x)
            elapsed  = (time.time() - start) * 1000
            times_single.append(elapsed)

        avg_single = np.mean(times_single)

        # Batch inference test (64 samples)
        times_batch = []
        for _ in range(20):
            start   = time.time()
            x_batch = torch.randn(64, 30, input_dim)
            with torch.no_grad():
                pred, _ = model(x_batch)
            elapsed = (time.time() - start) * 1000
            times_batch.append(elapsed)

        avg_batch = np.mean(times_batch)

        print(f"  Single inference avg : {avg_single:.2f} ms")
        print(f"  Batch (64) avg       : {avg_batch:.2f} ms")
        print(f"  Throughput           : "
              f"{1000/avg_single:.0f} req/sec")

        status = (
            "✅ PASS" if avg_single < 300
            else "❌ FAIL"
        )
        print(
            f"  Status               : "
            f"{status} (target < 300ms)"
        )

        return {
            'test'           : 'inference',
            'single_avg_ms'  : round(avg_single, 2),
            'batch_avg_ms'   : round(avg_batch,  2),
            'throughput_rps' : round(
                1000/avg_single, 0
            ),
            'passed'         : avg_single < 300
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            'test'  : 'inference',
            'error' : str(e),
            'passed': False
        }

# ============================================
# TEST 3: Feature Engineering Performance
# ============================================
def test_feature_engineering_performance():
    print("\n⚙️  TEST 3: Feature Engineering Performance")
    print("-"*40)

    cols = ['engine_id', 'cycle'] + \
           [f'setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(
        'data/train_FD001.txt',
        sep=r'\s+', header=None,
        names=cols, engine='python'
    )
    df.dropna(axis=1, how='all', inplace=True)

    sensor_cols = [
        c for c in df.columns
        if 'sensor' in c and df[c].std() > 0.001
    ]

    start = time.time()

    # Rolling features
    for col in sensor_cols:
        df[f'{col}_mean5'] = df[col]\
            .rolling(5, min_periods=1).mean()
        df[f'{col}_std5']  = df[col]\
            .rolling(5, min_periods=1).std()\
            .fillna(0)

    # Lag features
    for col in sensor_cols[:3]:
        for lag in [1, 3, 5]:
            df[f'{col}_lag{lag}'] = df\
                .groupby('engine_id')[col]\
                .transform(lambda x: x.shift(lag).fillna(0))

    elapsed = (time.time() - start) * 1000

    print(f"  Input rows      : {len(df)}")
    print(f"  Features created: {len(df.columns)}")
    print(f"  Time taken      : {elapsed:.2f} ms")

    status = (
        "✅ PASS" if elapsed < 10000
        else "❌ FAIL"
    )
    print(
        f"  Status          : "
        f"{status} (target < 10000ms)"
    )

    return {
        'test'      : 'feature_engineering',
        'time_ms'   : round(elapsed, 2),
        'n_features': len(df.columns),
        'passed'    : elapsed < 10000
    }

# ============================================
# TEST 4: Concurrent Request Simulation
# ============================================
def test_concurrent_requests(n_concurrent=10):
    print(f"\n🔀 TEST 4: Concurrent Requests ({n_concurrent})")
    print("-"*40)

    try:
        with open(
            'models/best_model_info.json'
        ) as f:
            model_info = json.load(f)

        input_dim = model_info['input_dim']

        def single_inference():
            model = LSTMWithAttention(
                input_dim=input_dim,
                hidden_dim=model_info['hidden_dim'],
                num_layers=model_info['num_layers'],
                dropout=model_info['dropout']
            )
            model.load_state_dict(torch.load(
                'models/best_rul_model.pth',
                map_location='cpu'
            ))
            model.eval()

            start = time.time()
            x     = torch.randn(1, 30, input_dim)
            with torch.no_grad():
                pred, _ = model(x)
            return (time.time() - start) * 1000

        # Run concurrent requests
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_concurrent
        ) as executor:
            futures = [
                executor.submit(single_inference)
                for _ in range(n_concurrent)
            ]
            times = [
                f.result()
                for f in concurrent.futures.as_completed(
                    futures
                )
            ]

        total_time = (time.time() - start) * 1000
        avg_time   = np.mean(times)

        print(
            f"  Concurrent users : {n_concurrent}"
        )
        print(
            f"  Total time       : {total_time:.2f} ms"
        )
        print(
            f"  Avg per request  : {avg_time:.2f} ms"
        )
        print(
            f"  Max latency      : {max(times):.2f} ms"
        )

        status = (
            "✅ PASS" if avg_time < 1000
            else "⚠️  WARN"
        )
        print(
            f"  Status           : "
            f"{status} (target < 1000ms)"
        )

        return {
            'test'         : 'concurrent',
            'n_concurrent' : n_concurrent,
            'total_ms'     : round(total_time, 2),
            'avg_ms'       : round(avg_time,   2),
            'max_ms'       : round(max(times),  2),
            'passed'       : avg_time < 1000
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            'test'  : 'concurrent',
            'error' : str(e),
            'passed': False
        }

# ============================================
# TEST 5: Drift Detection Performance
# ============================================
def test_drift_detection_performance():
    print("\n🔍 TEST 5: Drift Detection Performance")
    print("-"*40)

    from scipy import stats

    cols = ['engine_id', 'cycle'] + \
           [f'setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(
        'data/train_FD001.txt',
        sep=r'\s+', header=None,
        names=cols, engine='python'
    )
    df.dropna(axis=1, how='all', inplace=True)

    sensor_cols = [
        c for c in df.columns
        if 'sensor' in c and df[c].std() > 0.001
    ]

    split     = int(len(df) * 0.5)
    reference = df.iloc[:split]
    current   = df.iloc[split:]

    start = time.time()

    results = {}
    for col in sensor_cols:
        ks_stat, p_val = stats.ks_2samp(
            reference[col].dropna(),
            current[col].dropna()
        )
        results[col] = {
            'drifted': p_val < 0.05,
            'p_value': p_val
        }

    elapsed = (time.time() - start) * 1000

    drifted = sum(
        1 for v in results.values()
        if v['drifted']
    )

    print(f"  Sensors tested   : {len(sensor_cols)}")
    print(f"  Drifted sensors  : {drifted}")
    print(f"  Detection time   : {elapsed:.2f} ms")

    status = (
        "✅ PASS" if elapsed < 60000
        else "❌ FAIL"
    )
    print(
        f"  Status           : "
        f"{status} (target < 60000ms)"
    )

    return {
        'test'           : 'drift_detection',
        'n_sensors'      : len(sensor_cols),
        'drifted_sensors': drifted,
        'time_ms'        : round(elapsed, 2),
        'passed'         : elapsed < 60000
    }

# ============================================
# TEST 6: Scheduler Performance
# ============================================
def test_scheduler_performance():
    print("\n📅 TEST 6: Maintenance Scheduler Performance")
    print("-"*40)

    try:
        import pulp

        n_equipment = 10
        horizon     = 14

        np.random.seed(42)
        equipment = {
            i: {
                'rul'      : np.random.randint(5, 120),
                'prev_cost': np.random.randint(1000, 5000),
                'emrg_cost': np.random.randint(5000, 20000)
            }
            for i in range(n_equipment)
        }

        start = time.time()

        prob = pulp.LpProblem(
            "Perf_Test", pulp.LpMinimize
        )
        days   = list(range(1, horizon + 1))
        eq_ids = list(range(n_equipment))

        x = {
            (i,d): pulp.LpVariable(
                f"x_{i}_{d}", cat='Binary'
            )
            for i in eq_ids for d in days
        }
        f = {
            i: pulp.LpVariable(
                f"f_{i}", cat='Binary'
            )
            for i in eq_ids
        }

        prob += (
            pulp.lpSum([
                x[i,d] *
                equipment[i]['prev_cost']
                for i in eq_ids for d in days
            ]) +
            pulp.lpSum([
                f[i] * equipment[i]['emrg_cost']
                for i in eq_ids
            ])
        )

        for i in eq_ids:
            prob += pulp.lpSum(
                [x[i,d] for d in days]
            ) <= 1
            rul      = equipment[i]['rul']
            deadline = min(int(rul), horizon)
            if deadline > 0:
                prob += (
                    pulp.lpSum([
                        x[i,d]
                        for d in range(1, deadline+1)
                    ]) + f[i] >= 1
                )
            else:
                prob += f[i] == 1

        for d in days:
            prob += pulp.lpSum(
                [x[i,d] for i in eq_ids]
            ) <= 2

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        elapsed = (time.time() - start) * 1000

        print(
            f"  Equipment        : {n_equipment}"
        )
        print(f"  Horizon          : {horizon} days")
        print(
            f"  Solve time       : {elapsed:.2f} ms"
        )
        print(
            f"  Status           : "
            f"{pulp.LpStatus[prob.status]}"
        )

        passed = elapsed < 30000
        status = "✅ PASS" if passed else "❌ FAIL"
        print(
            f"  Test Status      : "
            f"{status} (target < 30000ms)"
        )

        return {
            'test'      : 'scheduler',
            'time_ms'   : round(elapsed, 2),
            'lp_status' : pulp.LpStatus[prob.status],
            'passed'    : passed
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            'test'  : 'scheduler',
            'error' : str(e),
            'passed': False
        }

# ============================================
# MAIN: Run All Tests
# ============================================
if __name__ == "__main__":

    test_start = datetime.now()
    results    = []

    # Run all tests
    results.append(
        test_data_loading_performance(n_runs=5)
    )
    results.append(
        test_inference_performance(n_requests=50)
    )
    results.append(
        test_feature_engineering_performance()
    )
    results.append(
        test_concurrent_requests(n_concurrent=5)
    )
    results.append(
        test_drift_detection_performance()
    )
    results.append(
        test_scheduler_performance()
    )

    # Summary
    duration = (
        datetime.now() - test_start
    ).seconds

    passed = sum(
        1 for r in results
        if r.get('passed', False)
    )
    total  = len(results)

    print("\n" + "="*55)
    print("📊 PERFORMANCE TEST SUMMARY")
    print("="*55)

    for r in results:
        status = (
            "✅ PASS" if r.get('passed')
            else "❌ FAIL"
        )
        print(f"  {r['test']:30s}: {status}")

    print(f"\n  Tests Passed : {passed}/{total}")
    print(f"  Duration     : {duration}s")
    print("="*55)

    # Save report
    os.makedirs(
        'data/performance_reports', exist_ok=True
    )

    # Convert all results to JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                k: make_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    report = {
        'generated_at': datetime.now().isoformat(),
        'duration_sec': int(duration),
        'passed'      : int(passed),
        'total'       : int(total),
        'results'     : make_serializable(results)
    }

    with open(
        'data/performance_reports/'
        'latest_perf_report.json', 'w'
    ) as f:
        json.dump(report, f, indent=2)

    print(
        "\n✅ Report saved!"
    )
    print("\n✅ Day 26 Complete!")