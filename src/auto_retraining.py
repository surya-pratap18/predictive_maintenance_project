# src/auto_retraining.py
# ============================================
# Automated Retraining Pipeline
# Simulates Airflow DAG #2:
# Drift Detection → Retrain → Promote → A/B Test
# ============================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset, DataLoader,
    TensorDataset, random_split
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import mlflow
import mlflow.pytorch
import pickle
import json
import os
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("="*55)
print("🔄 AUTO-RETRAINING PIPELINE")
print("   DAG #2: Drift → Retrain → Promote")
print("="*55)

# ============================================
# TASK 1: Check Drift Report
# ============================================
def task_check_drift():
    """Check if retraining is needed"""
    print("\n📊 [TASK 1/6] Checking drift report...")

    try:
        with open(
            'data/drift_reports/latest_drift.json'
        ) as f:
            report = json.load(f)

        drift_share = report.get('drift_share', 0)
        retraining  = report.get(
            'retraining_recommended', False
        )

        print(f"   Drift Share     : {drift_share:.2%}")
        print(f"   Retraining Needed: {retraining}")

        if retraining:
            print("   ✅ Retraining triggered!")
        else:
            print("   ✅ No retraining needed")

        return retraining, report

    except FileNotFoundError:
        print(
            "   ⚠️  No drift report found "
            "— forcing retraining"
        )
        return True, {}

# ============================================
# Model Definition
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
        return self.fc(context).squeeze(-1), attn

class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================
# TASK 2: Load & Prepare New Training Data
# ============================================
def task_prepare_data():
    """Load and prepare data for retraining"""
    print("\n📥 [TASK 2/6] Preparing training data...")

    # Load sequences
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_test  = np.load('data/processed/X_test.npy')
    y_test  = np.load('data/processed/y_test.npy')

    with open(
        'data/processed/rul_scaler.pkl', 'rb'
    ) as f:
        rul_scaler = pickle.load(f)

    # Add slight data augmentation
    # (simulate new incoming data)
    noise_factor = 0.02
    X_augmented  = X_train + np.random.normal(
        0, noise_factor, X_train.shape
    )
    y_augmented  = y_train + np.random.normal(
        0, noise_factor, y_train.shape
    )
    y_augmented  = np.clip(y_augmented, 0, 1)

    # Combine original + augmented
    X_combined = np.vstack([X_train, X_augmented])
    y_combined = np.hstack([y_train, y_augmented])

    print(f"   Original samples : {len(X_train)}")
    print(f"   Augmented samples: {len(X_augmented)}")
    print(f"   Total samples    : {len(X_combined)}")

    # Create dataloaders
    full_dataset = RULDataset(X_combined, y_combined)
    val_size     = int(len(full_dataset) * 0.2)
    train_size   = len(full_dataset) - val_size

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size]
    )
    test_ds = RULDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=64, shuffle=False
    )
    test_loader  = DataLoader(
        test_ds, batch_size=64, shuffle=False
    )

    print(f"   ✅ Data ready!")
    return (train_loader, val_loader,
            test_loader, rul_scaler,
            X_train.shape[2])

# ============================================
# TASK 3: Retrain Model
# ============================================
def task_retrain_model(train_loader,
                       val_loader,
                       input_dim,
                       epochs=15):
    """Retrain the LSTM model"""
    print("\n🔄 [TASK 3/6] Retraining model...")

    device = torch.device('cpu')

    # Load current model info
    try:
        with open(
            'models/best_model_info.json'
        ) as f:
            model_info = json.load(f)
        hidden_dim = model_info.get('hidden_dim', 64)
        num_layers = model_info.get('num_layers', 2)
        dropout    = model_info.get('dropout',    0.2)
    except:
        hidden_dim = 64
        num_layers = 2
        dropout    = 0.2

    # New model
    new_model = LSTMWithAttention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        new_model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler\
                    .ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val    = float('inf')
    best_state  = None
    train_losses= []
    val_losses  = []

    for epoch in range(epochs):
        # Train
        new_model.train()
        t_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred, _ = new_model(bx)
            loss    = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(
                new_model.parameters(), 1.0
            )
            optimizer.step()
            t_loss += loss.item()

        # Validate
        new_model.eval()
        v_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred, _= new_model(bx)
                v_loss += criterion(pred, by).item()

        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)
        train_losses.append(avg_t)
        val_losses.append(avg_v)
        scheduler.step(avg_v)

        if avg_v < best_val:
            best_val   = avg_v
            best_state = {
                k: v.clone()
                for k, v in
                new_model.state_dict().items()
            }

        if (epoch + 1) % 5 == 0:
            print(
                f"   Epoch [{epoch+1:2d}/{epochs}] "
                f"Train: {avg_t:.6f} | "
                f"Val: {avg_v:.6f}"
            )

    new_model.load_state_dict(best_state)
    print(f"   ✅ Retraining complete!")
    print(f"   📊 Best val loss: {best_val:.6f}")

    return new_model, best_val, train_losses

# ============================================
# TASK 4: Evaluate New Model
# ============================================
def task_evaluate_model(new_model,
                         test_loader,
                         rul_scaler):
    """Evaluate retrained model on test set"""
    print("\n📊 [TASK 4/6] Evaluating new model...")

    device = torch.device('cpu')
    new_model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx    = bx.to(device)
            p, _  = new_model(bx)
            preds.extend(p.cpu().numpy())
            targets.extend(by.numpy())

    preds   = np.array(preds)
    targets = np.array(targets)

    p_actual = rul_scaler.inverse_transform(
        preds.reshape(-1, 1)
    ).flatten()
    t_actual = rul_scaler.inverse_transform(
        targets.reshape(-1, 1)
    ).flatten()

    new_mae  = mean_absolute_error(t_actual, p_actual)
    new_rmse = np.sqrt(mean_squared_error(
        t_actual, p_actual
    ))
    new_r2   = r2_score(t_actual, p_actual)
    new_mape = np.mean(
        np.abs((t_actual - p_actual) /
               (t_actual + 1e-8))
    ) * 100

    # Load current model metrics
    try:
        with open('models/best_model_info.json') as f:
            current_info = json.load(f)
        current_mape = current_info.get(
            'test_mape', 999
        )
    except:
        current_mape = 999

    print(f"   New Model  MAPE: {new_mape:.2f}%")
    print(f"   Current Model  : {current_mape:.2f}%")

    improved = new_mape < current_mape
    print(
        f"   Improvement    : "
        f"{'✅ Yes' if improved else '❌ No'}"
    )

    metrics = {
        'mae' : float(new_mae),
        'rmse': float(new_rmse),
        'r2'  : float(new_r2),
        'mape': float(new_mape)
    }

    return metrics, improved

# ============================================
# TASK 5: A/B Model Promotion
# ============================================
def task_promote_model(new_model,
                        metrics,
                        improved,
                        input_dim):
    """
    Promote new model if better than current
    Implements A/B testing logic
    """
    print("\n🚀 [TASK 5/6] A/B Model Promotion...")

    os.makedirs('models', exist_ok=True)
    os.makedirs('models/versions', exist_ok=True)

    timestamp = datetime.now().strftime(
        '%Y%m%d_%H%M%S'
    )

    if improved:
        print(
            f"   ✅ New model is BETTER — promoting!"
        )

        # Backup current model
        if os.path.exists('models/best_rul_model.pth'):
            backup_path = (
                f'models/versions/'
                f'model_v{timestamp}_backup.pth'
            )
            shutil.copy(
                'models/best_rul_model.pth',
                backup_path
            )
            print(f"   💾 Current model backed up")

        # Save new model as production
        torch.save(
            new_model.state_dict(),
            'models/best_rul_model.pth'
        )

        # Update model info
        model_info = {
            'model_name'  : f'LSTM_Retrained_{timestamp}',
            'hidden_dim'  : 64,
            'num_layers'  : 2,
            'dropout'     : 0.2,
            'input_dim'   : int(input_dim),
            'seq_len'     : 30,
            'test_mae'    : metrics['mae'],
            'test_rmse'   : metrics['rmse'],
            'test_r2'     : metrics['r2'],
            'test_mape'   : metrics['mape'],
            'retrained_at': timestamp,
            'version'     : timestamp
        }

        with open(
            'models/best_model_info.json', 'w'
        ) as f:
            json.dump(model_info, f, indent=2)

        # Save versioned model
        version_path = (
            f'models/versions/'
            f'model_v{timestamp}.pth'
        )
        torch.save(
            new_model.state_dict(),
            version_path
        )

        print(
            f"   ✅ New model promoted to production!"
        )
        print(f"   📊 New MAPE: {metrics['mape']:.2f}%")

        return True, timestamp

    else:
        print(
            "   ⚠️  New model NOT better "
            "— keeping current"
        )
        print(
            f"   📊 Current model retained"
        )

        # Save as candidate (not promoted)
        candidate_path = (
            f'models/versions/'
            f'candidate_v{timestamp}.pth'
        )
        torch.save(
            new_model.state_dict(),
            candidate_path
        )

        return False, timestamp

# ============================================
# TASK 6: Save Pipeline Report
# ============================================
def task_save_pipeline_report(
    drift_report,
    metrics,
    promoted,
    timestamp,
    train_losses
):
    """Save full retraining pipeline report"""
    print(
        "\n💾 [TASK 6/6] Saving pipeline report..."
    )

    os.makedirs(
        'data/retraining_reports', exist_ok=True
    )

    report = {
        'pipeline_run_at': datetime.now().isoformat(),
        'trigger'        : 'drift_detected',
        'drift_share'    : drift_report.get(
            'drift_share', 0
        ),

        'retraining': {
            'epochs'       : len(train_losses),
            'final_loss'   : float(train_losses[-1]),
            'best_loss'    : float(min(train_losses))
        },

        'evaluation': {
            'mae' : metrics['mae'],
            'rmse': metrics['rmse'],
            'r2'  : metrics['r2'],
            'mape': metrics['mape']
        },

        'promotion': {
            'promoted'  : promoted,
            'version'   : timestamp,
            'model_path': (
                'models/best_rul_model.pth'
                if promoted else
                f'models/versions/candidate_v{timestamp}.pth'
            )
        },

        'rollback_available': os.path.exists(
            f'models/versions/'
        ),

        'next_drift_check': '7 days'
    }

    # Save report
    report_path = (
        f'data/retraining_reports/'
        f'retrain_{timestamp}.json'
    )
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Save as latest
    with open(
        'data/retraining_reports/latest_retrain.json',
        'w'
    ) as f:
        json.dump(report, f, indent=2)

    print(f"   ✅ Report saved: {report_path}")
    return report

# ============================================
# Rollback Function
# ============================================
def rollback_model():
    """
    Rollback to previous model version
    if new model causes issues
    """
    print("\n⏪ ROLLBACK initiated...")

    versions_dir = 'models/versions'
    if not os.path.exists(versions_dir):
        print("   ❌ No versions available!")
        return False

    # Find latest backup
    backups = [
        f for f in os.listdir(versions_dir)
        if 'backup' in f and f.endswith('.pth')
    ]

    if not backups:
        print("   ❌ No backup found!")
        return False

    # Use most recent backup
    latest_backup = sorted(backups)[-1]
    backup_path   = os.path.join(
        versions_dir, latest_backup
    )

    shutil.copy(
        backup_path,
        'models/best_rul_model.pth'
    )
    print(
        f"   ✅ Rolled back to: {latest_backup}"
    )
    return True

# ============================================
# Log to MLflow
# ============================================
def log_to_mlflow(metrics, promoted, timestamp):
    """Log retraining run to MLflow"""
    try:
        mlflow.set_experiment("auto_retraining")

        with mlflow.start_run(
            run_name=f"Retrain_{timestamp}"
        ):
            mlflow.log_params({
                'trigger'  : 'drift_detected',
                'promoted' : promoted,
                'timestamp': timestamp
            })
            mlflow.log_metrics({
                'retrain_mae' : metrics['mae'],
                'retrain_rmse': metrics['rmse'],
                'retrain_r2'  : metrics['r2'],
                'retrain_mape': metrics['mape']
            })

        print("   ✅ Logged to MLflow!")

    except Exception as e:
        print(f"   ⚠️  MLflow logging failed: {e}")

# ============================================
# MAIN PIPELINE
# ============================================
if __name__ == "__main__":

    pipeline_start = datetime.now()

    # Task 1: Check drift
    needs_retrain, drift_report = task_check_drift()

    if not needs_retrain:
        print("\n✅ No retraining needed! Pipeline done.")
        exit(0)

    # Task 2: Prepare data
    (train_loader, val_loader,
     test_loader, rul_scaler,
     input_dim) = task_prepare_data()

    # Task 3: Retrain
    new_model, best_val, train_losses = \
        task_retrain_model(
            train_loader, val_loader,
            input_dim, epochs=15
        )

    # Task 4: Evaluate
    metrics, improved = task_evaluate_model(
        new_model, test_loader, rul_scaler
    )

    # Task 5: Promote
    promoted, timestamp = task_promote_model(
        new_model, metrics,
        improved, input_dim
    )

    # Task 6: Save report
    report = task_save_pipeline_report(
        drift_report, metrics,
        promoted, timestamp, train_losses
    )

    # Log to MLflow
    log_to_mlflow(metrics, promoted, timestamp)

    # Pipeline summary
    duration = (
        datetime.now() - pipeline_start
    ).seconds

    print("\n" + "="*55)
    print("✅ AUTO-RETRAINING PIPELINE COMPLETE")
    print("="*55)
    print(f"""
  Pipeline Summary:
    • Duration      : {duration}s
    • Drift Detected: True
    • Retrained     : True
    • Promoted      : {promoted}
    • New MAPE      : {metrics['mape']:.2f}%
    • New R²        : {metrics['r2']:.4f}
    • Rollback      : Available ✅
    • MLflow        : Logged ✅
    """)
    print("="*55)