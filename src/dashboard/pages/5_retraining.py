# src/dashboard/pages/5_retraining.py
# ============================================
# Auto-Retraining Dashboard Page
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
import subprocess
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Auto-Retraining",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Auto-Retraining Pipeline")
st.markdown(
    "Automated model retraining triggered "
    "by drift detection"
)
st.markdown("---")

# ============================================
# Load Reports
# ============================================
def load_drift_report():
    try:
        with open(
            'data/drift_reports/latest_drift.json'
        ) as f:
            return json.load(f)
    except:
        return None

def load_retrain_report():
    try:
        with open(
            'data/retraining_reports/'
            'latest_retrain.json'
        ) as f:
            return json.load(f)
    except:
        return None

def load_model_info():
    try:
        with open('models/best_model_info.json') as f:
            return json.load(f)
    except:
        return None

def get_model_versions():
    try:
        versions_dir = 'models/versions'
        if not os.path.exists(versions_dir):
            return []
        files = os.listdir(versions_dir)
        return sorted(files, reverse=True)
    except:
        return []

drift_report   = load_drift_report()
retrain_report = load_retrain_report()
model_info     = load_model_info()
versions       = get_model_versions()

# ============================================
# Status Banner
# ============================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    drift_status = "🔴 Drifted" \
        if (drift_report and
            drift_report.get('dataset_drifted'))  \
        else "🟢 Stable"
    st.metric("Data Status", drift_status)

with col2:
    drift_share = drift_report.get(
        'drift_share', 0
    ) if drift_report else 0
    st.metric(
        "Drift Share",
        f"{drift_share:.1%}"
    )

with col3:
    retrain_needed = drift_report.get(
        'retraining_recommended', False
    ) if drift_report else False
    st.metric(
        "Retraining",
        "⚠️ Needed" if retrain_needed
        else "✅ Not needed"
    )

with col4:
    mape = model_info.get(
        'test_mape', 0
    ) if model_info else 0
    st.metric(
        "Current MAPE",
        f"{mape:.2f}%"
    )

with col5:
    st.metric(
        "Model Versions",
        len(versions)
    )

st.markdown("---")

# ============================================
# Tabs
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 Pipeline Control",
    "📊 Training History",
    "🗂️ Model Versions",
    "⏪ Rollback"
])

# ============================================
# TAB 1: Pipeline Control
# ============================================
with tab1:
    st.subheader("🔄 Pipeline Control Center")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Pipeline Settings")

        trigger_type = st.selectbox(
            "Trigger Type:",
            [
                "Drift Detected (Auto)",
                "Manual Override",
                "Scheduled (Weekly)"
            ]
        )

        epochs = st.slider(
            "Training Epochs:",
            min_value=5,
            max_value=50,
            value=15
        )

        promote_threshold = st.slider(
            "Promote if MAPE improves by (%):",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )

        st.markdown("### 🔔 Notifications")
        notify_email = st.checkbox(
            "Email on completion", value=True
        )
        notify_slack = st.checkbox(
            "Slack on promotion", value=True
        )

        st.markdown("---")

        # Pipeline trigger buttons
        if st.button(
            "🚀 Run Full Pipeline",
            use_container_width=True,
            type="primary"
        ):
            with st.spinner(
                "Running retraining pipeline..."
            ):
                try:
                    result = subprocess.run(
                        ['python',
                         'src/auto_retraining.py'],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result.returncode == 0:
                        st.success(
                            "✅ Pipeline completed!"
                        )
                        st.code(result.stdout)
                    else:
                        st.error(
                            "❌ Pipeline failed!"
                        )
                        st.code(result.stderr)

                    st.rerun()

                except subprocess.TimeoutExpired:
                    st.warning(
                        "⏱️ Pipeline timed out — "
                        "check terminal for status"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        if st.button(
            "🔍 Run Drift Check Only",
            use_container_width=True
        ):
            with st.spinner(
                "Checking drift..."
            ):
                try:
                    result = subprocess.run(
                        ['python',
                         'src/drift_detection.py'],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        st.success(
                            "✅ Drift check done!"
                        )
                    else:
                        st.error("❌ Failed!")
                        st.code(result.stderr)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with col2:
        st.markdown("### 📊 Pipeline Status")

        if retrain_report:
            # Pipeline steps visualization
            steps = [
                {
                    'name'  : '1. Drift Detection',
                    'status': '✅ Complete',
                    'detail': f"Drift: {drift_share:.1%}"
                },
                {
                    'name'  : '2. Data Preparation',
                    'status': '✅ Complete',
                    'detail': 'Data augmented'
                },
                {
                    'name'  : '3. Model Retraining',
                    'status': '✅ Complete',
                    'detail': f"Loss: {retrain_report['retraining']['final_loss']:.6f}"
                },
                {
                    'name'  : '4. Evaluation',
                    'status': '✅ Complete',
                    'detail': f"MAPE: {retrain_report['evaluation']['mape']:.2f}%"
                },
                {
                    'name'  : '5. A/B Promotion',
                    'status': (
                        '✅ Promoted'
                        if retrain_report['promotion']['promoted']
                        else '⚠️ Not promoted'
                    ),
                    'detail': f"Version: {retrain_report['promotion']['version']}"
                },
                {
                    'name'  : '6. Report Saved',
                    'status': '✅ Complete',
                    'detail': 'JSON + MLflow'
                }
            ]

            for step in steps:
                col_a, col_b, col_c = \
                    st.columns([2, 1, 2])
                with col_a:
                    st.markdown(
                        f"**{step['name']}**"
                    )
                with col_b:
                    st.markdown(step['status'])
                with col_c:
                    st.markdown(
                        f"_{step['detail']}_"
                    )
                st.markdown("---")

            # Last run info
            st.markdown(
                f"**Last run:** "
                f"{retrain_report['pipeline_run_at'][:19]}"
            )
            st.markdown(
                f"**Trigger:** "
                f"{retrain_report['trigger']}"
            )

        else:
            st.info(
                "💡 No retraining history found.\n"
                "Click 'Run Full Pipeline' to start!"
            )

            # Show pipeline diagram
            st.markdown("""
            ### Pipeline Flow:
            ```
            Drift Report
                ↓
            Data Preparation
                ↓
            Model Retraining
                ↓
            Evaluation
                ↓
            A/B Promotion
                ↓
            Report & MLflow
            ```
            """)

# ============================================
# TAB 2: Training History
# ============================================
with tab2:
    st.subheader("📊 Training History")

    if retrain_report:
        col1, col2 = st.columns(2)

        with col1:
            # Metrics comparison
            eval_metrics = retrain_report['evaluation']
            current_mape = model_info.get(
                'test_mape', 0
            ) if model_info else 0
            current_mae  = model_info.get(
                'test_mae', 0
            ) if model_info else 0

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                name='Before Retraining',
                x=['MAE', 'MAPE (%)'],
                y=[current_mae, current_mape],
                marker_color='steelblue',
                opacity=0.8
            ))
            fig1.add_trace(go.Bar(
                name='After Retraining',
                x=['MAE', 'MAPE (%)'],
                y=[
                    eval_metrics['mae'],
                    eval_metrics['mape']
                ],
                marker_color='green',
                opacity=0.8
            ))
            fig1.update_layout(
                title='Before vs After Retraining',
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=350
            )
            st.plotly_chart(
                fig1, use_container_width=True
            )

        with col2:
            # Metrics table
            st.markdown("### 📋 Metrics")
            metrics_data = {
                'Metric'   : ['MAE', 'RMSE',
                               'R²', 'MAPE (%)'],
                'New Model': [
                    round(eval_metrics['mae'],  4),
                    round(eval_metrics['rmse'], 4),
                    round(eval_metrics['r2'],   4),
                    round(eval_metrics['mape'], 2)
                ],
                'Promoted' : [
                    '✅' if retrain_report[
                        'promotion'
                    ]['promoted'] else '❌'
                ] * 4
            }
            st.dataframe(
                pd.DataFrame(metrics_data),
                use_container_width=True
            )

            # Promotion decision
            if retrain_report['promotion']['promoted']:
                st.success(
                    "✅ Model was PROMOTED to "
                    "production!"
                )
            else:
                st.warning(
                    "⚠️ Model was NOT promoted "
                    "(not better than current)"
                )

    else:
        st.info(
            "No training history yet. "
            "Run the pipeline first!"
        )

# ============================================
# TAB 3: Model Versions
# ============================================
with tab3:
    st.subheader("🗂️ Model Version Registry")

    if versions:
        version_data = []
        for v in versions:
            vtype = (
                'Backup'    if 'backup' in v
                else 'Candidate' if 'candidate' in v
                else 'Production'
            )
            version_data.append({
                'File'      : v,
                'Type'      : vtype,
                'Status'    : (
                    '🟢 Active'
                    if 'backup' not in v
                    and 'candidate' not in v
                    else '📦 Archived'
                ),
                'Size (KB)' : round(
                    os.path.getsize(
                        f'models/versions/{v}'
                    ) / 1024, 1
                )
            })

        ver_df = pd.DataFrame(version_data)
        st.dataframe(
            ver_df,
            use_container_width=True
        )

        st.metric(
            "Total Versions", len(versions)
        )
        total_size = sum(
            os.path.getsize(
                f'models/versions/{v}'
            )
            for v in versions
        ) / (1024*1024)
        st.metric(
            "Total Storage",
            f"{total_size:.1f} MB"
        )

    else:
        st.info(
            "No model versions yet. "
            "Run the pipeline to create versions!"
        )

    # Current model info
    if model_info:
        st.markdown("---")
        st.subheader("🏆 Current Production Model")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Model",
                model_info.get('model_name', 'N/A')
            )
        with col2:
            st.metric(
                "MAPE",
                f"{model_info.get('test_mape', 0):.2f}%"
            )
        with col3:
            st.metric(
                "R²",
                f"{model_info.get('test_r2', 0):.4f}"
            )
        with col4:
            retrained = model_info.get(
                'retrained_at', 'Original'
            )
            st.metric("Version", str(retrained)[:10])

# ============================================
# TAB 4: Rollback
# ============================================
with tab4:
    st.subheader("⏪ Model Rollback")

    st.warning("""
    ⚠️ **Use rollback only if:**
    - New model is producing bad predictions
    - Critical errors in production
    - Unexpected behavior detected
    """)

    backups = [
        v for v in versions
        if 'backup' in v
    ]

    if backups:
        st.success(
            f"✅ {len(backups)} backup(s) available"
        )

        selected_backup = st.selectbox(
            "Select version to rollback to:",
            backups
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"**Selected:** `{selected_backup}`"
            )

        with col2:
            confirm = st.checkbox(
                "I confirm this rollback"
            )

        if st.button(
            "⏪ Execute Rollback",
            disabled=not confirm,
            type="primary"
        ):
            try:
                import shutil
                backup_path = (
                    f'models/versions/{selected_backup}'
                )
                shutil.copy(
                    backup_path,
                    'models/best_rul_model.pth'
                )
                st.success(
                    f"✅ Rolled back to: "
                    f"{selected_backup}"
                )
                st.balloons()

            except Exception as e:
                st.error(f"❌ Rollback failed: {e}")

    else:
        st.info(
            "No backups available. "
            "Run the pipeline first to "
            "create model versions."
        )

    # Rollback history
    st.markdown("---")
    st.subheader("📜 Rollback History")
    st.info(
        "Rollback history will appear here "
        "after executing rollbacks."
    )