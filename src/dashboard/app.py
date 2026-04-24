# src/dashboard/app.py
# ============================================
# Main Streamlit Dashboard App
# Predictive Maintenance Platform
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Predictive Maintenance Platform",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS Styling
# ============================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(
            135deg, #1e2130, #2d3250
        );
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #4CAF50;
        margin: 5px 0;
    }

    /* Alert cards */
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-warning {
        background-color: #ffaa00;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-normal {
        background-color: #00aa44;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }

    /* Header styling */
    h1 { color: #4CAF50; }
    h2 { color: #2196F3; }
    h3 { color: #FF9800; }
</style>
""", unsafe_allow_html=True)

# ============================================
# Sidebar Navigation
# ============================================
st.sidebar.image(
    "https://img.icons8.com/color/96/000000/maintenance.png",
    width=80
)
st.sidebar.title("🔧 PredMaint Platform")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "📍 Navigate to:",
    [
        "🏠 Overview",
        "🔍 Equipment Detail",
        "🚨 Alerts",
        "📊 Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"🕐 Last Updated: "
    f"{datetime.now().strftime('%H:%M:%S')}"
)

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# ============================================
# Helper Functions
# ============================================

@st.cache_data(ttl=60)
def load_sensor_data():
    """Load CMAPSS sensor data"""
    try:
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

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_schedule():
    """Load maintenance schedule"""
    try:
        with open(
            'data/schedules/optimal_schedule.json'
        ) as f:
            return json.load(f)
    except:
        return None

@st.cache_data(ttl=60)
def load_model_info():
    """Load best model info"""
    try:
        with open('models/best_model_info.json') as f:
            return json.load(f)
    except:
        return None

def get_health_status(rul):
    """Get health status based on RUL"""
    if rul < 20:
        return "🔴 Critical", "red"
    elif rul < 50:
        return "🟡 Warning", "orange"
    else:
        return "🟢 Healthy", "green"

# ============================================
# LOAD DATA
# ============================================
df           = load_sensor_data()
schedule     = load_schedule()
model_info   = load_model_info()

# ============================================
# PAGE: OVERVIEW
# ============================================
if page == "🏠 Overview":

    st.title("🔧 Predictive Maintenance Dashboard")
    st.markdown(
        "**Real-time Asset Health Monitoring & "
        "RUL Forecasting Platform**"
    )
    st.markdown("---")

    if df.empty:
        st.error("❌ Could not load sensor data!")
        st.stop()

    # KPI Metrics
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_eq    = df['engine_id'].nunique()
    critical    = df[df['RUL'] < 20]['engine_id']\
                    .nunique()
    warning     = df[
        (df['RUL'] >= 20) & (df['RUL'] < 50)
    ]['engine_id'].nunique()
    healthy     = total_eq - critical - warning
    avg_rul     = df.groupby('engine_id')['RUL']\
                    .last().mean()

    with col1:
        st.metric(
            "Total Equipment",
            total_eq,
            delta=None
        )
    with col2:
        st.metric(
            "🔴 Critical",
            critical,
            delta=f"-{critical} need attention",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            "🟡 Warning",
            warning,
            delta=None
        )
    with col4:
        st.metric(
            "🟢 Healthy",
            healthy,
            delta=None
        )
    with col5:
        st.metric(
            "Avg RUL",
            f"{avg_rul:.0f} cycles",
            delta=None
        )

    st.markdown("---")

    # Equipment Health Table
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🏭 Equipment Health Status")

        # Get latest RUL per engine
        latest_rul = df.groupby('engine_id').last()\
                       .reset_index()

        health_data = []
        for _, row in latest_rul.iterrows():
            status, color = get_health_status(row['RUL'])
            health_data.append({
                'Equipment'  : f"EQ-{int(row['engine_id']):03d}",
                'Cycle'      : int(row['cycle']),
                'RUL'        : int(row['RUL']),
                'Status'     : status
            })

        health_df = pd.DataFrame(health_data)\
                      .sort_values('RUL')

        st.dataframe(
            health_df,
            use_container_width=True,
            height=400
        )

    with col_right:
        st.subheader("📈 RUL Distribution")

        latest_rul_vals = df.groupby('engine_id')['RUL']\
                           .last()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=latest_rul_vals,
            nbinsx=20,
            marker_color=[
                'red'    if r < 20 else
                'orange' if r < 50 else
                'green'
                for r in latest_rul_vals
            ],
            name='RUL Distribution'
        ))
        fig.add_vline(
            x=20, line_dash="dash",
            line_color="red",
            annotation_text="Critical"
        )
        fig.add_vline(
            x=50, line_dash="dash",
            line_color="orange",
            annotation_text="Warning"
        )
        fig.update_layout(
            title='Equipment RUL Distribution',
            xaxis_title='RUL (cycles)',
            yaxis_title='Count',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(
            fig, use_container_width=True
        )

    st.markdown("---")

    # Sensor Trends
    st.subheader("📡 Real-Time Sensor Trends")

    sensor_cols = [
        f'sensor_{i}' for i in range(1, 22)
        if f'sensor_{i}' in df.columns
        and df[f'sensor_{i}'].std() > 0.001
    ]

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        selected_sensor = st.selectbox(
            "Select Sensor:",
            sensor_cols
        )

    with col_s2:
        selected_engine = st.selectbox(
            "Select Equipment:",
            [f"EQ-{i:03d}" for i in
             df['engine_id'].unique()[:20]]
        )

    eng_id   = int(
        selected_engine.replace('EQ-', '')
    )
    eng_data = df[df['engine_id'] == eng_id]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=eng_data['cycle'],
        y=eng_data[selected_sensor],
        mode='lines',
        name=selected_sensor,
        line=dict(color='#2196F3', width=2)
    ))

    # Add failure zone
    max_cycle = eng_data['cycle'].max()
    fig2.add_vrect(
        x0=max_cycle - 30,
        x1=max_cycle,
        fillcolor="red",
        opacity=0.2,
        annotation_text="⚠️ Failure Zone"
    )

    fig2.update_layout(
        title=f'{selected_sensor} — {selected_engine}',
        xaxis_title='Cycle',
        yaxis_title='Sensor Value',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Maintenance Schedule Summary
    if schedule:
        st.markdown("---")
        st.subheader("📅 Upcoming Maintenance")
        sched_df = pd.DataFrame(schedule['schedule'])
        sched_df = sched_df[
            sched_df['scheduled_day'] != 'N/A'
        ].sort_values('scheduled_day')
        st.dataframe(
            sched_df[[
                'equipment_id', 'equipment_type',
                'criticality', 'predicted_rul',
                'scheduled_day', 'status'
            ]],
            use_container_width=True
        )

# ============================================
# PAGE: EQUIPMENT DETAIL
# ============================================
elif page == "🔍 Equipment Detail":

    st.title("🔍 Equipment Detail Analysis")
    st.markdown("---")

    if df.empty:
        st.error("❌ No data available!")
        st.stop()

    # Equipment selector
    col1, col2 = st.columns([1, 3])
    with col1:
        selected = st.selectbox(
            "Select Equipment:",
            [f"EQ-{i:03d}" for i in
             df['engine_id'].unique()]
        )

    eng_id   = int(selected.replace('EQ-', ''))
    eng_data = df[df['engine_id'] == eng_id].copy()

    # Equipment KPIs
    last_row   = eng_data.iloc[-1]
    status, _  = get_health_status(last_row['RUL'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Equipment ID", selected)
    with col2:
        st.metric("Current Cycle",
                  int(last_row['cycle']))
    with col3:
        st.metric("Predicted RUL",
                  f"{int(last_row['RUL'])} cycles")
    with col4:
        st.metric("Health Status", status)

    st.markdown("---")

    # Sensor grid
    st.subheader("📡 Sensor Readings Over Time")

    sensor_cols = [
        f'sensor_{i}' for i in range(1, 22)
        if f'sensor_{i}' in df.columns
        and df[f'sensor_{i}'].std() > 0.001
    ]

    # Show top 6 sensors
    fig = go.Figure()
    colors = [
        '#2196F3', '#4CAF50', '#FF9800',
        '#9C27B0', '#F44336', '#00BCD4'
    ]

    top_sensors = sensor_cols[:6]
    for i, sensor in enumerate(top_sensors):
        fig.add_trace(go.Scatter(
            x=eng_data['cycle'],
            y=eng_data[sensor],
            name=sensor,
            line=dict(
                color=colors[i % len(colors)],
                width=1.5
            )
        ))

    fig.update_layout(
        title=f'Sensor Trends — {selected}',
        xaxis_title='Cycle',
        yaxis_title='Sensor Value',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # RUL trend
    st.subheader("📉 RUL Degradation Curve")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=eng_data['cycle'],
        y=eng_data['RUL'],
        fill='tozeroy',
        fillcolor='rgba(33,150,243,0.2)',
        line=dict(color='#2196F3', width=2),
        name='RUL'
    ))
    fig2.add_hline(
        y=30, line_dash="dash",
        line_color="red",
        annotation_text="⚠️ Critical Threshold"
    )
    fig2.update_layout(
        title=f'RUL Over Time — {selected}',
        xaxis_title='Cycle',
        yaxis_title='RUL (cycles)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Sensor stats table
    st.subheader("📊 Sensor Statistics")
    stats = eng_data[sensor_cols].describe()\
                                  .round(4)
    st.dataframe(stats, use_container_width=True)

# ============================================
# PAGE: ALERTS
# ============================================
elif page == "🚨 Alerts":

    st.title("🚨 Alert Management")
    st.markdown("---")

    if df.empty:
        st.error("❌ No data available!")
        st.stop()

    # Alert summary
    latest_rul = df.groupby('engine_id')['RUL']\
                   .last().reset_index()

    critical_eq = latest_rul[
        latest_rul['RUL'] < 20
    ]['engine_id'].tolist()
    warning_eq  = latest_rul[
        (latest_rul['RUL'] >= 20) &
        (latest_rul['RUL'] < 50)
    ]['engine_id'].tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(
            f"🔴 CRITICAL ALERTS: {len(critical_eq)}"
        )
    with col2:
        st.warning(
            f"🟡 WARNING ALERTS: {len(warning_eq)}"
        )
    with col3:
        st.success(
            f"🟢 HEALTHY: "
            f"{latest_rul['engine_id'].nunique() - len(critical_eq) - len(warning_eq)}"
        )

    st.markdown("---")

    # Critical alerts
    if critical_eq:
        st.subheader("🔴 Critical Alerts")
        for eq_id in critical_eq[:5]:
            rul = latest_rul[
                latest_rul['engine_id'] == eq_id
            ]['RUL'].values[0]
            st.markdown(
                f"""
                <div class="alert-critical">
                ⚠️ <b>EQ-{eq_id:03d}</b> —
                RUL: <b>{int(rul)} cycles</b> —
                IMMEDIATE MAINTENANCE REQUIRED!
                </div>
                """,
                unsafe_allow_html=True
            )

    # Warning alerts
    if warning_eq:
        st.subheader("🟡 Warning Alerts")
        for eq_id in warning_eq[:5]:
            rul = latest_rul[
                latest_rul['engine_id'] == eq_id
            ]['RUL'].values[0]
            st.markdown(
                f"""
                <div class="alert-warning">
                ⚠️ <b>EQ-{eq_id:03d}</b> —
                RUL: <b>{int(rul)} cycles</b> —
                Schedule maintenance soon
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Alert Rule Builder
    st.subheader("⚙️ Alert Rule Builder")

    col1, col2, col3 = st.columns(3)
    with col1:
        alert_sensor = st.selectbox(
            "Select Sensor:",
            [f'sensor_{i}' for i in range(2, 22)
             if f'sensor_{i}' in df.columns]
        )
    with col2:
        threshold = st.number_input(
            "Threshold Value:",
            value=float(
                df[alert_sensor].mean()
                if alert_sensor in df.columns
                else 0
            )
        )
    with col3:
        severity = st.selectbox(
            "Severity:",
            ["Critical", "Warning", "Info"]
        )

    channel = st.multiselect(
        "Notification Channel:",
        ["Email", "SMS", "Slack", "PagerDuty"],
        default=["Email"]
    )

    if st.button("💾 Save Alert Rule"):
        st.success(
            f"✅ Alert rule saved!\n"
            f"   Sensor: {alert_sensor} > {threshold}\n"
            f"   Severity: {severity}\n"
            f"   Channels: {', '.join(channel)}"
        )

# ============================================
# PAGE: REPORTS
# ============================================
elif page == "📊 Reports":

    st.title("📊 Reports & Analytics")
    st.markdown("---")

    if df.empty:
        st.error("❌ No data available!")
        st.stop()

    # Report type
    report_type = st.selectbox(
        "Select Report Type:",
        [
            "Weekly Summary",
            "Equipment Health Report",
            "Maintenance Cost Report",
            "Model Performance Report"
        ]
    )

    st.markdown("---")

    if report_type == "Weekly Summary":
        st.subheader("📋 Weekly Summary Report")

        col1, col2 = st.columns(2)

        with col1:
            # Failures this week
            critical_count = df[
                df['RUL'] < 20
            ]['engine_id'].nunique()
            st.metric(
                "Critical Equipment",
                critical_count
            )

            # Average RUL
            avg_rul = df.groupby('engine_id')[
                'RUL'
            ].last().mean()
            st.metric(
                "Average Fleet RUL",
                f"{avg_rul:.0f} cycles"
            )

        with col2:
            # Fleet health pie chart
            latest = df.groupby('engine_id')[
                'RUL'
            ].last()
            critical = (latest < 20).sum()
            warning  = ((latest >= 20) &
                        (latest < 50)).sum()
            healthy  = (latest >= 50).sum()

            fig = go.Figure(go.Pie(
                labels=['Critical', 'Warning', 'Healthy'],
                values=[critical, warning, healthy],
                marker_colors=['red', 'orange', 'green']
            ))
            fig.update_layout(
                title='Fleet Health Status',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

    elif report_type == "Model Performance Report":
        st.subheader("🤖 Model Performance Report")

        if model_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Best Model",
                    model_info['model_name']
                )
            with col2:
                st.metric(
                    "Test MAE",
                    f"{model_info['test_mae']:.4f}"
                )
            with col3:
                st.metric(
                    "Test MAPE",
                    f"{model_info['test_mape']:.2f}%"
                )
            with col4:
                st.metric(
                    "Test R²",
                    f"{model_info['test_r2']:.4f}"
                )
        else:
            st.warning("⚠️ Model info not found!")

    elif report_type == "Maintenance Cost Report":
        st.subheader("💰 Maintenance Cost Report")

        if schedule:
            st.metric(
                "Total Optimized Cost",
                f"${schedule['total_cost']:,.2f}"
            )
            sched_df = pd.DataFrame(
                schedule['schedule']
            )
            st.dataframe(
                sched_df,
                use_container_width=True
            )
        else:
            st.warning("⚠️ Schedule not found!")

    # Download button
    st.markdown("---")
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'report_type' : report_type,
        'model_info'  : model_info,
    }

    st.download_button(
        label="📥 Download Report (JSON)",
        data=json.dumps(report_data, indent=2),
        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )   