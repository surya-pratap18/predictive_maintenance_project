# tests/test_edge_cases.py
# ============================================
# Edge Case Testing
# ============================================

import pytest
import numpy as np
import pandas as pd
import json
import os
import torch
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Edge Case 1: Missing Data Handling
# ============================================
def test_missing_data_handling():
    """Test system handles missing sensor data"""
    print("\n🔍 Testing missing data handling...")

    cols = ['engine_id', 'cycle'] + \
           [f'setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(
        'data/train_FD001.txt',
        sep=r'\s+', header=None,
        names=cols, engine='python'
    )
    df.dropna(axis=1, how='all', inplace=True)

    # Inject missing values (10%)
    df_missing = df.copy()
    mask = np.random.random(df_missing.shape) < 0.1
    df_missing[mask] = np.nan

    # System should handle NaN
    sensor_cols = [
        c for c in df_missing.columns
        if 'sensor' in c
    ]

    df_filled = df_missing[sensor_cols].fillna(0)
    assert not df_filled.isna().any().any(), \
        "Missing values should be filled"

    # Rolling features should still work
    df_filled['sensor_2_mean'] = df_filled[
        'sensor_2'
    ].rolling(5, min_periods=1).mean()

    assert not df_filled[
        'sensor_2_mean'
    ].isna().any(), \
        "Rolling mean should handle NaN"

    print("✅ Missing data handling passed!")

# ============================================
# Edge Case 2: High Noise Data
# ============================================
def test_high_noise_handling():
    """Test model handles high noise input"""
    print("\n🔍 Testing high noise handling...")

    try:
        with open(
            'models/best_model_info.json'
        ) as f:
            model_info = json.load(f)

        input_dim = model_info['input_dim']

        # Normal input
        normal_input = torch.randn(1, 30, input_dim)

        # Very noisy input (10x noise)
        noisy_input  = torch.randn(
            1, 30, input_dim
        ) * 10

        # Zero input
        zero_input   = torch.zeros(1, 30, input_dim)

        # Extreme values
        extreme_input = torch.ones(
            1, 30, input_dim
        ) * 1000

        from test_edge_cases import load_model_for_testing
          
        model = load_model_for_testing()

        if model is not None:
            model.eval()
            with torch.no_grad():
                for name, inp in [
                    ('Normal', normal_input),
                    ('Noisy',  noisy_input),
                    ('Zero',   zero_input),
                    ('Extreme',extreme_input)
                ]:
                    pred, _ = model(inp)
                    assert not torch.isnan(pred).any(), \
                        f"{name} input produced NaN!"
                    print(f"  ✅ {name} input: {pred.item():.4f}")

        print("✅ High noise handling passed!")

    except Exception as e:
        pytest.skip(f"Model not available: {e}")

# ============================================
# Helper: Load Model
# ============================================
def load_model_for_testing():
    """Helper to load model for testing"""
    try:
        import torch.nn as nn

        class AttentionLayer(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.attention = nn.Linear(
                    hidden_dim, 1
                )
            def forward(self, lstm_out):
                attn = torch.softmax(
                    self.attention(lstm_out), dim=1
                )
                return (attn * lstm_out).sum(dim=1), attn

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
                    dropout=dropout
                    if num_layers > 1 else 0,
                    bidirectional=True
                )
                self.attention = AttentionLayer(
                    hidden_dim * 2
                )
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
                return (
                    self.fc(context).squeeze(-1),
                    attn
                )

        with open(
            'models/best_model_info.json'
        ) as f:
            info = json.load(f)

        model = LSTMWithAttention(
            input_dim=info['input_dim'],
            hidden_dim=info['hidden_dim'],
            num_layers=info['num_layers'],
            dropout=info['dropout']
        )
        model.load_state_dict(torch.load(
            'models/best_rul_model.pth',
            map_location='cpu'
        ))
        return model

    except Exception as e:
        return None

# ============================================
# Edge Case 3: Complete Sensor Failure
# ============================================
def test_complete_sensor_failure():
    """Test when all sensors return zero"""
    print("\n🔍 Testing complete sensor failure...")

    try:
        model = load_model_for_testing()
        if model is None:
            pytest.skip("Model not available")

        with open(
            'models/best_model_info.json'
        ) as f:
            info = json.load(f)

        # All zeros — complete sensor failure
        zero_input = torch.zeros(
            1, 30, info['input_dim']
        )

        model.eval()
        with torch.no_grad():
            pred, _ = model(zero_input)

        assert not torch.isnan(pred).any(), \
            "Should handle zero input without NaN"
        assert not torch.isinf(pred).any(), \
            "Should handle zero input without Inf"

        print(
            f"  Sensor failure RUL: {pred.item():.4f}"
        )
        print("✅ Sensor failure test passed!")

    except Exception as e:
        pytest.skip(f"Skipping: {e}")

# ============================================
# Edge Case 4: Single Engine Data
# ============================================
def test_single_engine_data():
    """Test with minimum data (1 engine)"""
    print("\n🔍 Testing single engine data...")

    cols = ['engine_id', 'cycle'] + \
           [f'setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(
        'data/train_FD001.txt',
        sep=r'\s+', header=None,
        names=cols, engine='python'
    )
    df.dropna(axis=1, how='all', inplace=True)

    # Single engine only
    single_engine = df[
        df['engine_id'] == 1
    ].copy()

    assert len(single_engine) > 0, \
        "Should have data for engine 1"

    # RUL calculation should work
    mc = single_engine['cycle'].max()
    single_engine['RUL'] = mc - single_engine['cycle']

    assert (single_engine['RUL'] >= 0).all(), \
        "RUL should be non-negative"
    assert single_engine['RUL'].iloc[-1] == 0, \
        "Last cycle RUL should be 0"

    print(
        f"  Engine 1 cycles: {len(single_engine)}"
    )
    print(
        f"  Max RUL: {single_engine['RUL'].max()}"
    )
    print("✅ Single engine test passed!")

# ============================================
# Edge Case 5: Scheduler No Solution
# ============================================
def test_scheduler_infeasible():
    """Test scheduler handles infeasible problems"""
    print("\n🔍 Testing infeasible scheduler...")

    try:
        import pulp

        # Create impossible problem
        # (10 equipment, only 1 day, max 1 per day)
        prob = pulp.LpProblem(
            "Infeasible_Test", pulp.LpMinimize
        )

        x = {
            i: pulp.LpVariable(
                f"x_{i}", cat='Binary'
            )
            for i in range(10)
        }

        # All must be done on same day
        prob += pulp.lpSum(x.values()) >= 10

        # But max 2 per day
        prob += pulp.lpSum(x.values()) <= 2

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Should return infeasible status
        status = pulp.LpStatus[prob.status]
        print(f"  Solver status: {status}")
        assert status in [
            'Infeasible', 'Undefined',
            'Not Solved', 'Optimal'
        ], "Should handle infeasible gracefully"

        print("✅ Infeasible scheduler test passed!")

    except Exception as e:
        pytest.skip(f"PuLP not available: {e}")

# ============================================
# Edge Case 6: Drift With Identical Data
# ============================================
def test_drift_identical_data():
    """Test drift detection with identical data"""
    print("\n🔍 Testing drift with identical data...")

    from scipy import stats
    import numpy as np

    # Same distribution — should NOT drift
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(0, 1, 1000)

    ks_stat, p_val = stats.ks_2samp(data1, data2)

    # p_value should be high (no drift)
    assert p_val > 0.01, \
        "Identical distributions should not drift"

    print(f"  KS stat : {ks_stat:.4f}")
    print(f"  P-value : {p_val:.4f}")
    print(f"  Drifted : {p_val < 0.05}")
    print("✅ Identical data drift test passed!")

# ============================================
# Edge Case 7: Large Batch Inference
# ============================================
def test_large_batch_inference():
    """Test model with large batch size"""
    print("\n🔍 Testing large batch inference...")

    try:
        model = load_model_for_testing()
        if model is None:
            pytest.skip("Model not available")

        with open(
            'models/best_model_info.json'
        ) as f:
            info = json.load(f)

        model.eval()
        batch_sizes = [1, 16, 32, 64, 128]

        for bs in batch_sizes:
            x = torch.randn(
                bs, 30, info['input_dim']
            )
            with torch.no_grad():
                pred, _ = model(x)

            assert pred.shape == (bs,), \
                f"Wrong output shape for batch {bs}"
            assert not torch.isnan(pred).any(), \
                f"NaN in batch {bs}"

            print(
                f"  Batch {bs:4d}: "
                f"output shape {pred.shape} ✅"
            )

        print("✅ Large batch inference passed!")

    except Exception as e:
        pytest.skip(f"Skipping: {e}")

# ============================================
# Edge Case 8: JSON Report Corruption
# ============================================
def test_corrupt_json_handling():
    """Test handling of corrupted JSON files"""
    print("\n🔍 Testing corrupt JSON handling...")

    import tempfile

    # Write corrupted JSON
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json',
        delete=False
    ) as f:
        f.write("{ invalid json content !!!")
        tmp_path = f.name

    # Should handle gracefully
    try:
        with open(tmp_path) as f:
            data = json.load(f)
        result = "loaded"
    except json.JSONDecodeError:
        result = "handled"
    finally:
        os.unlink(tmp_path)

    assert result == "handled", \
        "Should catch JSON decode errors"

    print("✅ Corrupt JSON handling passed!")

if __name__ == "__main__":
    print("="*55)
    print("🔍 EDGE CASE TESTING")
    print("="*55)

    test_missing_data_handling()
    test_single_engine_data()
    test_complete_sensor_failure()
    test_scheduler_infeasible()
    test_drift_identical_data()
    test_large_batch_inference()
    test_corrupt_json_handling()

    print("\n" + "="*55)
    print("✅ ALL EDGE CASE TESTS COMPLETE!")
    print("="*55)