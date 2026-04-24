# src/dashboard/pages/1_sensor_monitoring.py
# ============================================
# Real-Time Sensor Monitoring Page
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import queue
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Sensor Monitoring",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Real-Time Sensor Monitoring")
st.markdown(
    "Live sensor data streaming with "
    "anomaly detection highlights"
)
st.markdown("---")

# ============================================
# Load Data
# ============================================
@st.cache_data(ttl=30)
def load_sensor_data():
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

df = load_sensor_data()

# Get useful sensors
sensor_cols = [
    f'sensor_{i}' for i in range(1, 22)
    if f'sensor_{i}' in df.columns
    and df[f'sensor_{i}'].std() > 0.001
]

# ============================================
# Sidebar Controls
# ============================================
st.sidebar.title("🎛️ Controls")

selected_engine = st.sidebar.selectbox(
    "Select Equipment:",
    [f"EQ-{i:03d}" for i in df['engine_id'].unique()]
)

selected_sensors = st.sidebar.multiselect(
    "Select Sensors:",
    sensor_cols,
    default=sensor_cols[:4]
)

show_anomalies = st.sidebar.checkbox(
    "Show Anomaly Zones", value=True
)

show_threshold = st.sidebar.checkbox(
    "Show Thresholds", value=True
)

auto_refresh = st.sidebar.checkbox(
    "Auto Refresh (5s)", value=False
)

refresh_rate = st.sidebar.slider(
    "Refresh Rate (seconds):",
    min_value=2,
    max_value=30,
    value=5
)

# ============================================
# Get Engine Data
# ============================================
eng_id   = int(selected_engine.replace('EQ-', ''))
eng_data = df[df['engine_id'] == eng_id].copy()

last_row   = eng_data.iloc[-1]
rul        = int(last_row['RUL'])
max_cycle  = int(eng_data['cycle'].max())

# Health status
if rul < 20:
    health = "🔴 CRITICAL"
    health_color = "red"
elif rul < 50:
    health = "🟡 WARNING"
    health_color = "orange"
else:
    health = "🟢 HEALTHY"
    health_color = "green"

# ============================================
# Equipment Status Banner
# ============================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Equipment", selected_engine)
with col2:
    st.metric("Current Cycle", max_cycle)
with col3:
    st.metric("Predicted RUL", f"{rul} cycles")
with col4:
    st.metric("Health Status", health)
with col5:
    anomaly_rate = eng_data['failure_label'].mean()
    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate*100:.1f}%"
    )

st.markdown("---")

# ============================================
# SECTION 1: Multi-Sensor Line Charts
# ============================================
st.subheader("📈 Sensor Trend Analysis")

if not selected_sensors:
    st.warning("Please select at least one sensor!")
else:
    # Create subplots
    n_sensors = len(selected_sensors)
    fig = make_subplots(
        rows=n_sensors, cols=1,
        shared_xaxes=True,
        subplot_titles=selected_sensors,
        vertical_spacing=0.05
    )

    colors = [
        '#2196F3', '#4CAF50', '#FF9800',
        '#9C27B0', '#F44336', '#00BCD4',
        '#FF5722', '#795548', '#607D8B'
    ]

    for i, sensor in enumerate(selected_sensors):
        row = i + 1

        # Main sensor line
        fig.add_trace(
            go.Scatter(
                x=eng_data['cycle'],
                y=eng_data[sensor],
                mode='lines',
                name=sensor,
                line=dict(
                    color=colors[i % len(colors)],
                    width=1.5
                )
            ),
            row=row, col=1
        )

        # Anomaly highlights
        if show_anomalies:
            anomaly_data = eng_data[
                eng_data['failure_label'] == 1
            ]
            if not anomaly_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_data['cycle'],
                        y=anomaly_data[sensor],
                        mode='markers',
                        name=f'{sensor} Anomaly',
                        marker=dict(
                            color='red',
                            size=6,
                            symbol='x'
                        ),
                        showlegend=(i == 0)
                    ),
                    row=row, col=1
                )

        # Threshold lines
        if show_threshold:
            mean_val = eng_data[sensor].mean()
            std_val  = eng_data[sensor].std()
            upper    = mean_val + 2 * std_val
            lower    = mean_val - 2 * std_val

            fig.add_hline(
                y=upper,
                line_dash="dot",
                line_color="red",
                line_width=1,
                row=row, col=1
            )
            fig.add_hline(
                y=lower,
                line_dash="dot",
                line_color="orange",
                line_width=1,
                row=row, col=1
            )

    # Failure zone annotation
    if show_anomalies:
        for i in range(n_sensors):
            fig.add_vrect(
                x0=max_cycle - 30,
                x1=max_cycle,
                fillcolor="red",
                opacity=0.1,
                annotation_text="⚠️ Failure Zone"
                if i == 0 else "",
                row=i+1, col=1
            )

    fig.update_layout(
        height=200 * n_sensors,
        title=f"Sensor Trends — {selected_engine}",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02
        )
    )
    fig.update_xaxes(title_text="Cycle",
                     row=n_sensors, col=1)

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 2: Anomaly Detection Timeline
# ============================================
st.subheader("🚨 Anomaly Detection Timeline")

col1, col2 = st.columns([2, 1])

with col1:
    # Timeline with anomaly flags
    fig2 = go.Figure()

    # RUL line
    fig2.add_trace(go.Scatter(
        x=eng_data['cycle'],
        y=eng_data['RUL'],
        fill='tozeroy',
        fillcolor='rgba(33,150,243,0.15)',
        line=dict(color='#2196F3', width=2),
        name='RUL'
    ))

    # Anomaly points on RUL
    anomaly_data = eng_data[
        eng_data['failure_label'] == 1
    ]
    if not anomaly_data.empty:
        fig2.add_trace(go.Scatter(
            x=anomaly_data['cycle'],
            y=anomaly_data['RUL'],
            mode='markers',
            marker=dict(
                color='red', size=8,
                symbol='triangle-down'
            ),
            name='Anomaly Detected'
        ))

    fig2.add_hline(
        y=30, line_dash="dash",
        line_color="red",
        annotation_text="Critical (RUL=30)"
    )
    fig2.add_hline(
        y=50, line_dash="dash",
        line_color="orange",
        annotation_text="Warning (RUL=50)"
    )

    fig2.update_layout(
        title='RUL with Anomaly Flags',
        xaxis_title='Cycle',
        yaxis_title='RUL (cycles)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("📊 Anomaly Stats")

    total_cycles   = len(eng_data)
    anomaly_cycles = eng_data['failure_label'].sum()
    normal_cycles  = total_cycles - anomaly_cycles

    st.metric("Total Cycles",   total_cycles)
    st.metric("Normal Cycles",  int(normal_cycles))
    st.metric("Anomaly Cycles", int(anomaly_cycles))
    st.metric(
        "Anomaly Rate",
        f"{anomaly_cycles/total_cycles*100:.1f}%"
    )

    # Mini pie chart
    fig3 = go.Figure(go.Pie(
        labels=['Normal', 'Anomaly'],
        values=[normal_cycles, anomaly_cycles],
        marker_colors=['green', 'red'],
        hole=0.4
    ))
    fig3.update_layout(
        height=250,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=True,
        margin=dict(t=0, b=0, l=0, r=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 3: Sensor Correlation Heatmap
# ============================================
st.subheader("🔥 Sensor Correlation Heatmap")

corr_sensors = st.multiselect(
    "Select sensors for correlation:",
    sensor_cols,
    default=sensor_cols[:8]
)

if corr_sensors:
    corr_matrix = eng_data[corr_sensors].corr()

    fig4 = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10)
    ))
    fig4.update_layout(
        title=f'Sensor Correlation — {selected_engine}',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 4: Multi-Equipment Comparison
# ============================================
st.subheader("🏭 Multi-Equipment Comparison")

compare_engines = st.multiselect(
    "Select Equipment to Compare:",
    [f"EQ-{i:03d}" for i in df['engine_id'].unique()],
    default=[f"EQ-{i:03d}" for i in
             df['engine_id'].unique()[:5]]
)

compare_sensor = st.selectbox(
    "Sensor to Compare:",
    sensor_cols,
    key='compare_sensor'
)

if compare_engines and compare_sensor:
    fig5 = go.Figure()
    colors_cmp = px.colors.qualitative.Set1

    for i, eq in enumerate(compare_engines):
        eq_id   = int(eq.replace('EQ-', ''))
        eq_data = df[df['engine_id'] == eq_id]

        fig5.add_trace(go.Scatter(
            x=eq_data['cycle'],
            y=eq_data[compare_sensor],
            mode='lines',
            name=eq,
            line=dict(
                color=colors_cmp[i % len(colors_cmp)],
                width=1.5
            ),
            opacity=0.8
        ))

    fig5.update_layout(
        title=f'{compare_sensor} — Multi-Equipment',
        xaxis_title='Cycle',
        yaxis_title='Sensor Value',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=350
    )
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 5: Live Stream Simulator
# ============================================
st.subheader("🔄 Live Sensor Stream Simulator")

col1, col2 = st.columns([1, 3])

with col1:
    stream_sensor = st.selectbox(
        "Stream Sensor:",
        sensor_cols,
        key='stream_sensor'
    )
    n_points = st.slider(
        "Points to stream:",
        10, 100, 30
    )
    stream_speed = st.slider(
        "Speed (ms):",
        100, 1000, 200
    )

with col2:
    if st.button("▶️ Start Live Stream"):
        # Get last n_points of data
        stream_data = eng_data[
            stream_sensor
        ].values[-n_points:]

        # Add noise to simulate real-time
        noise = np.random.normal(
            0, stream_data.std() * 0.05,
            n_points
        )
        stream_data_noisy = stream_data + noise

        # Animate
        placeholder = st.empty()

        for i in range(1, n_points + 1):
            fig_stream = go.Figure()

            fig_stream.add_trace(go.Scatter(
                y=stream_data_noisy[:i],
                mode='lines+markers',
                line=dict(
                    color='#4CAF50', width=2
                ),
                marker=dict(size=4),
                name='Live Data'
            ))

            # Last point highlighted
            fig_stream.add_trace(go.Scatter(
                x=[i-1],
                y=[stream_data_noisy[i-1]],
                mode='markers',
                marker=dict(
                    color='red', size=12,
                    symbol='star'
                ),
                name='Current'
            ))

            fig_stream.update_layout(
                title=f'Live: {stream_sensor}',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300,
                showlegend=False
            )

            placeholder.plotly_chart(
                fig_stream,
                use_container_width=True
            )
            time.sleep(stream_speed / 1000)

        st.success("✅ Stream complete!")

# ============================================
# Auto Refresh
# ============================================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()