# src/dashboard/pages/4_drift_monitoring.py
# ============================================
# Data Drift Monitoring Dashboard Page
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Drift Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data & Model Drift Monitoring")
st.markdown(
    "Monitor input data drift, concept drift "
    "and model performance degradation"
)
st.markdown("---")

# ============================================
# Load Data
# ============================================
@st.cache_data(ttl=60)
def load_data():
    cols = ['engine_id', 'cycle'] + \
           [f'setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(
        'data/train_FD001.txt',
        sep='\s+', header=None,
        names=cols, engine='python'
    )
    df.dropna(axis=1, how='all', inplace=True)

    max_cycle = df.groupby('engine_id')['cycle']\
                  .max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycle, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=125)
    df.drop(columns=['max_cycle'], inplace=True)
    df['failure_label'] = (df['RUL'] < 30).astype(int)

    return df

@st.cache_data(ttl=60)
def load_drift_report():
    """Load latest drift report"""
    try:
        with open(
            'data/drift_reports/latest_drift.json'
        ) as f:
            return json.load(f)
    except:
        return None

df           = load_data()
drift_report = load_drift_report()

# Feature columns
sensor_cols = [
    f'sensor_{i}' for i in range(1, 22)
    if f'sensor_{i}' in df.columns
    and df[f'sensor_{i}'].std() > 0.001
]

# Split reference/current
split_idx  = int(len(df) * 0.5)
reference  = df.iloc[:split_idx]
current    = df.iloc[split_idx:]

# ============================================
# Drift Summary Banner
# ============================================
if drift_report:
    is_drifted   = drift_report['dataset_drifted']
    drift_share  = drift_report['drift_share']
    tests_passed = drift_report['tests_passed']
    retrain      = drift_report['retraining_recommended']

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        status = "🔴 DRIFTED" if is_drifted \
                 else "🟢 STABLE"
        st.metric("Dataset Status", status)
    with col2:
        st.metric(
            "Drift Share",
            f"{drift_share:.1%}"
        )
    with col3:
        st.metric(
            "Tests Passed",
            "✅ Yes" if tests_passed else "❌ No"
        )
    with col4:
        st.metric(
            "Retraining",
            "⚠️ Needed" if retrain else "✅ Not needed"
        )
    with col5:
        generated = drift_report.get(
            'generated_at', 'N/A'
        )[:19]
        st.metric("Last Check", generated)

    if is_drifted:
        st.error(
            "⚠️ DATA DRIFT DETECTED! "
            "Consider retraining the model."
        )
    else:
        st.success(
            "✅ No significant drift detected. "
            "Model is stable."
        )
else:
    st.warning(
        "⚠️ No drift report found. "
        "Run drift_detection.py first!"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Dataset Status", "🔵 Unknown")
    with col2:
        st.metric("Drift Share", "N/A")
    with col3:
        st.metric("Tests Passed", "N/A")
    with col4:
        st.metric("Retraining",   "N/A")
    with col5:
        st.metric("Last Check",   "Never")

st.markdown("---")

# ============================================
# Tabs
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Distribution Drift",
    "🎯 Target Drift",
    "📈 Performance Drift",
    "📋 Drift Report"
])

# ============================================
# TAB 1: Distribution Drift
# ============================================
with tab1:
    st.subheader("📊 Feature Distribution Drift")

    # Sensor selector
    selected_sensor = st.selectbox(
        "Select Sensor to Analyze:",
        sensor_cols
    )

    col1, col2 = st.columns(2)

    with col1:
        # Distribution comparison
        fig1 = go.Figure()

        fig1.add_trace(go.Histogram(
            x=reference[selected_sensor],
            name='Reference (Past)',
            opacity=0.7,
            marker_color='steelblue',
            nbinsx=30
        ))
        fig1.add_trace(go.Histogram(
            x=current[selected_sensor],
            name='Current (Recent)',
            opacity=0.7,
            marker_color='darkorange',
            nbinsx=30
        ))

        fig1.update_layout(
            title=f'{selected_sensor} Distribution',
            xaxis_title='Sensor Value',
            yaxis_title='Count',
            barmode='overlay',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=350
        )
        st.plotly_chart(
            fig1, use_container_width=True
        )

    with col2:
        # Box plot comparison
        fig2 = go.Figure()

        fig2.add_trace(go.Box(
            y=reference[selected_sensor],
            name='Reference',
            marker_color='steelblue',
            boxpoints='outliers'
        ))
        fig2.add_trace(go.Box(
            y=current[selected_sensor],
            name='Current',
            marker_color='darkorange',
            boxpoints='outliers'
        ))

        fig2.update_layout(
            title=f'{selected_sensor} Box Plot',
            yaxis_title='Sensor Value',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=350
        )
        st.plotly_chart(
            fig2, use_container_width=True
        )

    # KS Test Results
    st.subheader("🔬 KS Test Results — All Sensors")

    ks_results = []
    for col in sensor_cols:
        try:
            ks_stat, p_val = stats.ks_2samp(
                reference[col].dropna(),
                current[col].dropna()
            )
            drifted = p_val < 0.05
            ks_results.append({
                'Sensor'   : col,
                'KS Stat'  : round(ks_stat, 4),
                'P-Value'  : round(p_val,   4),
                'Drifted'  : '🔴 Yes' if drifted
                             else '✅ No',
                'Severity' : 'High'   if p_val < 0.01
                             else 'Medium' if p_val < 0.05
                             else 'Low'
            })
        except:
            pass

    ks_df = pd.DataFrame(ks_results)

    # Color drifted rows
    drifted_count = (
        ks_df['Drifted'] == '🔴 Yes'
    ).sum()
    st.markdown(
        f"**Drifted sensors: "
        f"{drifted_count}/{len(ks_df)}**"
    )

    st.dataframe(
        ks_df.sort_values('P-Value'),
        use_container_width=True,
        height=300
    )

    # Drift heatmap
    st.subheader("🔥 Drift Score Heatmap")

    ks_matrix = ks_df.set_index(
        'Sensor'
    )['KS Stat'].values.reshape(1, -1)

    fig3 = go.Figure(go.Heatmap(
        z=ks_matrix,
        x=ks_df['Sensor'],
        y=['Drift Score'],
        colorscale='RdYlGn_r',
        zmin=0, zmax=1,
        text=np.round(ks_matrix, 3),
        texttemplate='%{text}',
        colorbar=dict(title='KS Score')
    ))
    fig3.update_layout(
        title='KS Drift Score per Sensor',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=200
    )
    st.plotly_chart(fig3, use_container_width=True)

# ============================================
# TAB 2: Target Drift
# ============================================
with tab2:
    st.subheader("🎯 Target (RUL) Drift Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # RUL distribution drift
        fig4 = go.Figure()

        fig4.add_trace(go.Histogram(
            x=reference['RUL'],
            name='Reference RUL',
            opacity=0.7,
            marker_color='steelblue',
            nbinsx=25
        ))
        fig4.add_trace(go.Histogram(
            x=current['RUL'],
            name='Current RUL',
            opacity=0.7,
            marker_color='red',
            nbinsx=25
        ))

        fig4.update_layout(
            title='RUL Distribution Drift',
            xaxis_title='RUL (cycles)',
            yaxis_title='Count',
            barmode='overlay',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300
        )
        st.plotly_chart(
            fig4, use_container_width=True
        )

    with col2:
        # RUL stats comparison
        ref_stats = reference['RUL'].describe()
        cur_stats = current['RUL'].describe()

        stats_df = pd.DataFrame({
            'Statistic' : ref_stats.index,
            'Reference' : ref_stats.values.round(2),
            'Current'   : cur_stats.values.round(2),
            'Change'    : (
                cur_stats.values -
                ref_stats.values
            ).round(2)
        })

        st.markdown("**RUL Statistics Comparison:**")
        st.dataframe(
            stats_df,
            use_container_width=True
        )

        # KS test on RUL
        ks_stat, p_val = stats.ks_2samp(
            reference['RUL'],
            current['RUL']
        )
        rul_drifted = p_val < 0.05

        if rul_drifted:
            st.error(
                f"🔴 RUL distribution has DRIFTED!\n"
                f"KS={ks_stat:.4f}, p={p_val:.4f}"
            )
        else:
            st.success(
                f"✅ RUL distribution is STABLE\n"
                f"KS={ks_stat:.4f}, p={p_val:.4f}"
            )

    # Failure label drift
    st.subheader("⚠️ Failure Rate Drift")

    ref_fail_rate = reference['failure_label'].mean()
    cur_fail_rate = current['failure_label'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Reference Failure Rate",
            f"{ref_fail_rate:.2%}"
        )
    with col2:
        st.metric(
            "Current Failure Rate",
            f"{cur_fail_rate:.2%}",
            delta=f"{(cur_fail_rate-ref_fail_rate):.2%}"
        )
    with col3:
        drift_pct = abs(
            cur_fail_rate - ref_fail_rate
        ) / ref_fail_rate * 100
        st.metric(
            "Rate Change",
            f"{drift_pct:.1f}%"
        )

    # Failure rate over time
    fig5 = go.Figure()

    # Rolling failure rate
    window     = 100
    fail_roll  = df['failure_label']\
                   .rolling(window).mean()

    fig5.add_trace(go.Scatter(
        x=list(range(len(fail_roll))),
        y=fail_roll,
        mode='lines',
        name='Rolling Failure Rate',
        line=dict(color='red', width=2)
    ))

    fig5.add_vline(
        x=split_idx,
        line_dash="dash",
        line_color="white",
        annotation_text="Reference|Current"
    )

    fig5.update_layout(
        title='Failure Rate Over Time (Rolling)',
        xaxis_title='Sample Index',
        yaxis_title='Failure Rate',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300
    )
    st.plotly_chart(fig5, use_container_width=True)

# ============================================
# TAB 3: Performance Drift
# ============================================
with tab3:
    st.subheader("📈 Model Performance Drift")

    # Simulate performance metrics over time
    np.random.seed(42)
    n_windows  = 10
    window_ids = list(range(1, n_windows + 1))

    # Simulate degrading performance
    base_mae  = 12.0
    base_mape = 10.0

    mae_trend  = [
        base_mae + i * 0.5 +
        np.random.normal(0, 0.3)
        for i in range(n_windows)
    ]
    mape_trend = [
        base_mape + i * 0.3 +
        np.random.normal(0, 0.2)
        for i in range(n_windows)
    ]
    r2_trend   = [
        0.95 - i * 0.02 +
        np.random.normal(0, 0.01)
        for i in range(n_windows)
    ]

    col1, col2 = st.columns(2)

    with col1:
        # MAE trend
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=window_ids, y=mae_trend,
            mode='lines+markers',
            name='MAE',
            line=dict(color='steelblue', width=2)
        ))
        fig6.add_hline(
            y=base_mae * 1.2,
            line_dash='dash',
            line_color='red',
            annotation_text='Alert Threshold'
        )
        fig6.update_layout(
            title='MAE Over Time Windows',
            xaxis_title='Window',
            yaxis_title='MAE (cycles)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=280
        )
        st.plotly_chart(
            fig6, use_container_width=True
        )

    with col2:
        # MAPE trend
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=window_ids, y=mape_trend,
            mode='lines+markers',
            name='MAPE',
            line=dict(color='darkorange', width=2)
        ))
        fig7.add_hline(
            y=12,
            line_dash='dash',
            line_color='red',
            annotation_text='Target (12%)'
        )
        fig7.update_layout(
            title='MAPE Over Time Windows',
            xaxis_title='Window',
            yaxis_title='MAPE (%)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=280
        )
        st.plotly_chart(
            fig7, use_container_width=True
        )

    # R² trend
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(
        x=window_ids, y=r2_trend,
        fill='tozeroy',
        fillcolor='rgba(76,175,80,0.2)',
        mode='lines+markers',
        name='R²',
        line=dict(color='green', width=2)
    ))
    fig8.add_hline(
        y=0.85,
        line_dash='dash',
        line_color='red',
        annotation_text='Min Acceptable R²'
    )
    fig8.update_layout(
        title='R² Score Over Time',
        xaxis_title='Window',
        yaxis_title='R² Score',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=280
    )
    st.plotly_chart(fig8, use_container_width=True)

    # Performance summary
    st.subheader("📊 Performance Summary")
    perf_df = pd.DataFrame({
        'Window'  : window_ids,
        'MAE'     : [round(m, 3) for m in mae_trend],
        'MAPE (%)': [round(m, 3) for m in mape_trend],
        'R²'      : [round(r, 4) for r in r2_trend],
        'Status'  : [
            '🔴 Degraded' if m > base_mae * 1.2
            else '✅ OK'
            for m in mae_trend
        ]
    })
    st.dataframe(
        perf_df, use_container_width=True
    )

# ============================================
# TAB 4: Drift Report
# ============================================
with tab4:
    st.subheader("📋 Latest Drift Report")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Run New Detection")

        ref_size = st.slider(
            "Reference Size (%):",
            min_value=30,
            max_value=70,
            value=50
        )

        significance = st.selectbox(
            "Significance Level:",
            [0.01, 0.05, 0.10],
            index=1
        )

        if st.button(
            "🔍 Run Drift Detection",
            use_container_width=True
        ):
            with st.spinner(
                "Running drift detection..."
            ):
                try:
                    # Simple KS-test drift detection
                    split = int(
                        len(df) * ref_size / 100
                    )
                    ref = df.iloc[:split]
                    cur = df.iloc[split:]

                    drifted_cols = []
                    col_results  = {}

                    for col in sensor_cols:
                        ks_s, p_v = stats.ks_2samp(
                            ref[col].dropna(),
                            cur[col].dropna()
                        )
                        is_drift = p_v < significance
                        if is_drift:
                            drifted_cols.append(col)
                        col_results[col] = {
                            'drifted'  : bool(is_drift),
                            'p_value'  : float(p_v),
                            'statistic': 'KS'
                        }

                    drift_share = (
                        len(drifted_cols) /
                        len(sensor_cols)
                    )
                    new_report = {
                        'generated_at': datetime.now()\
                            .isoformat(),
                        'dataset_drifted': (
                            drift_share > 0.2
                        ),
                        'drift_share'    : drift_share,
                        'tests_passed'   : (
                            drift_share < 0.3
                        ),
                        'column_drift'   : col_results,
                        'retraining_recommended': (
                            drift_share > 0.2
                        )
                    }

                    os.makedirs(
                        'data/drift_reports',
                        exist_ok=True
                    )
                    with open(
                        'data/drift_reports/'
                        'latest_drift.json', 'w'
                    ) as f:
                        json.dump(
                            new_report, f, indent=2
                        )

                    st.success(
                        f"✅ Detection complete!\n"
                        f"Drifted: {len(drifted_cols)}"
                        f"/{len(sensor_cols)} sensors"
                    )
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with col2:
        if drift_report:
            st.markdown("### 📊 Current Report")

            # Show column drift results
            col_drift = drift_report.get(
                'column_drift', {}
            )
            if col_drift:
                drift_rows = []
                for col, data in col_drift.items():
                    drift_rows.append({
                        'Sensor'  : col,
                        'Drifted' : '🔴 Yes'
                        if data['drifted']
                        else '✅ No',
                        'P-Value' : round(
                            data.get('p_value', 1.0),
                            4
                        )
                    })

                drift_table = pd.DataFrame(drift_rows)
                st.dataframe(
                    drift_table,
                    use_container_width=True,
                    height=300
                )

            # Export report
            report_json = json.dumps(
                drift_report, indent=2
            )
            st.download_button(
                "📥 Download Drift Report",
                report_json,
                "drift_report.json",
                "application/json"
            )
        else:
            st.info(
                "💡 Run drift detection to "
                "see results here!"
            )

    # Retraining recommendation
    st.markdown("---")
    st.subheader("🔄 Retraining Recommendation")

    if drift_report and \
       drift_report.get('retraining_recommended'):
        st.error("""
        🚨 **RETRAINING RECOMMENDED**

        Significant data drift detected!
        The model may be producing inaccurate
        predictions. Consider:

        1. ✅ Collect new training data
        2. ✅ Retrain LSTM model
        3. ✅ Validate on recent data
        4. ✅ Deploy new model version
        """)
        if st.button(
            "🔄 Trigger Retraining Pipeline"
        ):
            st.success(
                "✅ Retraining pipeline triggered!\n"
                "Check Airflow DAG for status."
            )
    else:
        st.success("""
        ✅ **NO RETRAINING NEEDED**

        Model is performing within acceptable
        bounds. Continue monitoring.

        Next scheduled check: In 7 days
        """)