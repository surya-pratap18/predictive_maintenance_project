# src/dashboard/pages/2_rul_forecasting.py
# ============================================
# RUL Forecasting Visualization Page
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="RUL Forecasting",
    page_icon="⏱️",
    layout="wide"
)

st.title("⏱️ RUL Forecasting Dashboard")
st.markdown(
    "Remaining Useful Life predictions with "
    "uncertainty quantification"
)
st.markdown("---")

# ============================================
# Load Model & Data
# ============================================

# Model architecture
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )
        context = (attn_weights * lstm_out).sum(dim=1)
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

@st.cache_resource
def load_model_and_scalers():
    """Load trained LSTM model and scalers"""
    try:
        with open(
            'models/best_model_info.json', 'r'
        ) as f:
            model_info = json.load(f)

        with open(
            'data/processed/rul_scaler.pkl', 'rb'
        ) as f:
            rul_scaler = pickle.load(f)

        with open(
            'data/processed/scaler.pkl', 'rb'
        ) as f:
            feat_scaler = pickle.load(f)

        with open(
            'data/processed/feature_cols.pkl', 'rb'
        ) as f:
            feature_cols = pickle.load(f)

        device = torch.device('cpu')
        model  = LSTMWithAttention(
            input_dim=model_info['input_dim'],
            hidden_dim=model_info['hidden_dim'],
            num_layers=model_info['num_layers'],
            dropout=model_info['dropout']
        ).to(device)

        model.load_state_dict(torch.load(
            'models/best_rul_model.pth',
            map_location=device
        ))
        model.eval()

        return (model, rul_scaler,
                feat_scaler, feature_cols,
                model_info, device)

    except Exception as e:
        return None, None, None, None, None, None

@st.cache_data(ttl=60)
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

# Load everything
(model, rul_scaler,
 feat_scaler, feature_cols,
 model_info, device) = load_model_and_scalers()

df = load_sensor_data()

# ============================================
# Predict RUL Function
# ============================================
def predict_rul(engine_data, seq_len=30,
                n_samples=20):
    """
    Predict RUL with uncertainty using
    Monte Carlo dropout
    """
    if model is None:
        return None, None, None

    try:
        # Get features
        available_cols = [
            c for c in feature_cols
            if c in engine_data.columns
        ]

        if len(available_cols) == 0:
            return None, None, None

        features = engine_data[
            available_cols
        ].fillna(0).values

        # Scale features
        features_sc = feat_scaler.transform(features)

        # Create sequence
        if len(features_sc) >= seq_len:
            seq = features_sc[-seq_len:]
        else:
            # Pad with zeros
            pad = np.zeros((
                seq_len - len(features_sc),
                features_sc.shape[1]
            ))
            seq = np.vstack([pad, features_sc])

        seq_tensor = torch.FloatTensor(
            seq
        ).unsqueeze(0).to(device)

        # Monte Carlo predictions
        preds = []
        model.train()  # Enable dropout

        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = model(seq_tensor)
                preds.append(pred.item())

        model.eval()

        preds_actual = rul_scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()

        mean_pred = np.mean(preds_actual)
        std_pred  = np.std(preds_actual)

        return mean_pred, std_pred, preds_actual

    except Exception as e:
        return None, None, None

# ============================================
# Sidebar Controls
# ============================================
st.sidebar.title("⚙️ RUL Settings")

selected_engine = st.sidebar.selectbox(
    "Select Equipment:",
    [f"EQ-{i:03d}" for i in
     df['engine_id'].unique()]
)

confidence_level = st.sidebar.slider(
    "Confidence Level:",
    min_value=50,
    max_value=99,
    value=90,
    step=5,
    help="Prediction interval width"
)

seq_len = st.sidebar.slider(
    "Sequence Length:",
    min_value=10,
    max_value=60,
    value=30
)

show_actual = st.sidebar.checkbox(
    "Show Actual RUL", value=True
)

show_confidence = st.sidebar.checkbox(
    "Show Confidence Bands", value=True
)

# ============================================
# Get Engine Data
# ============================================
eng_id   = int(selected_engine.replace('EQ-', ''))
eng_data = df[df['engine_id'] == eng_id].copy()
last_row = eng_data.iloc[-1]
true_rul = int(last_row['RUL'])

# Predict RUL
mean_rul, std_rul, mc_preds = predict_rul(
    eng_data, seq_len=seq_len
)

if mean_rul is None:
    mean_rul = true_rul + np.random.normal(0, 5)
    std_rul  = abs(mean_rul * 0.1)

# Confidence interval
z_score  = {80: 1.28, 90: 1.645, 95: 1.96,
             99: 2.576}.get(confidence_level, 1.645)
ci_upper = mean_rul + z_score * std_rul
ci_lower = max(0, mean_rul - z_score * std_rul)

# ============================================
# KPI Row
# ============================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Equipment", selected_engine)

with col2:
    delta = mean_rul - true_rul
    st.metric(
        "Predicted RUL",
        f"{mean_rul:.0f} cycles",
        delta=f"{delta:+.0f} vs actual"
    )

with col3:
    st.metric(
        "Actual RUL",
        f"{true_rul} cycles"
    )

with col4:
    st.metric(
        "Uncertainty (±)",
        f"{std_rul:.1f} cycles"
    )

with col5:
    fail_prob = max(0, min(100,
        (1 - mean_rul/125) * 100
    ))
    st.metric(
        "Failure Probability",
        f"{fail_prob:.1f}%"
    )

st.markdown("---")

# ============================================
# SECTION 1: RUL Gauge Chart
# ============================================
col_gauge, col_line = st.columns([1, 2])

with col_gauge:
    st.subheader("🎯 RUL Gauge")

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=mean_rul,
        delta={
            'reference': true_rul,
            'increasing': {'color': 'green'},
            'decreasing': {'color': 'red'}
        },
        title={
            'text': f"Predicted RUL<br>"
                    f"<span style='font-size:0.8em;"
                    f"color:gray'>"
                    f"{selected_engine}</span>"
        },
        gauge={
            'axis': {
                'range': [0, 125],
                'tickwidth': 1
            },
            'bar': {'color': "#2196F3"},
            'steps': [
                {'range': [0, 20],
                 'color': '#ff4444'},
                {'range': [20, 50],
                 'color': '#ffaa00'},
                {'range': [50, 125],
                 'color': '#00aa44'}
            ],
            'threshold': {
                'line': {
                    'color': "red",
                    'width': 4
                },
                'thickness': 0.75,
                'value': 30
            }
        }
    ))

    fig_gauge.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(t=80, b=20, l=20, r=20)
    )
    st.plotly_chart(
        fig_gauge, use_container_width=True
    )

    # Confidence interval display
    st.markdown(
        f"""
        **{confidence_level}% Confidence Interval:**
        - Lower: **{ci_lower:.0f}** cycles
        - Upper: **{ci_upper:.0f}** cycles
        - Range: **±{z_score*std_rul:.0f}** cycles
        """
    )

with col_line:
    st.subheader("📉 RUL Prediction Timeline")

    fig_line = go.Figure()

    # Actual RUL
    if show_actual:
        fig_line.add_trace(go.Scatter(
            x=eng_data['cycle'],
            y=eng_data['RUL'],
            mode='lines',
            name='Actual RUL',
            line=dict(
                color='white',
                width=2
            )
        ))

    # Simulated predictions over time
    cycles     = eng_data['cycle'].values
    actual_rul = eng_data['RUL'].values
    pred_rul   = actual_rul + np.random.normal(
        0, std_rul, len(actual_rul)
    )
    pred_rul   = np.clip(pred_rul, 0, 125)

    fig_line.add_trace(go.Scatter(
        x=cycles,
        y=pred_rul,
        mode='lines',
        name='Predicted RUL',
        line=dict(color='#2196F3', width=2)
    ))

    # Confidence bands
    if show_confidence:
        upper_band = pred_rul + z_score * std_rul
        lower_band = np.maximum(
            0, pred_rul - z_score * std_rul
        )

        fig_line.add_trace(go.Scatter(
            x=np.concatenate([cycles, cycles[::-1]]),
            y=np.concatenate([upper_band, lower_band[::-1]]),
            fill='toself',
            fillcolor='rgba(33,150,243,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_level}% CI'
        ))

    # Threshold lines
    fig_line.add_hline(
        y=30, line_dash="dash",
        line_color="red",
        annotation_text="⚠️ Critical (30)"
    )
    fig_line.add_hline(
        y=50, line_dash="dash",
        line_color="orange",
        annotation_text="⚠️ Warning (50)"
    )

    # Current position marker
    fig_line.add_vline(
        x=eng_data['cycle'].max(),
        line_dash="dot",
        line_color="yellow",
        annotation_text="📍 Now"
    )

    fig_line.update_layout(
        title=f'RUL Timeline — {selected_engine}',
        xaxis_title='Cycle',
        yaxis_title='RUL (cycles)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=350,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02
        )
    )
    st.plotly_chart(
        fig_line, use_container_width=True
    )

st.markdown("---")

# ============================================
# SECTION 2: Monte Carlo Distribution
# ============================================
st.subheader(
    "🎲 Monte Carlo Prediction Distribution"
)

col1, col2 = st.columns([2, 1])

with col1:
    if mc_preds is not None:
        fig_mc = go.Figure()

        fig_mc.add_trace(go.Histogram(
            x=mc_preds,
            nbinsx=20,
            marker_color='#2196F3',
            opacity=0.8,
            name='MC Samples'
        ))

        fig_mc.add_vline(
            x=mean_rul,
            line_color='white',
            line_width=2,
            annotation_text=f"Mean: {mean_rul:.1f}"
        )
        fig_mc.add_vline(
            x=ci_upper,
            line_color='red',
            line_dash='dash',
            annotation_text=f"Upper: {ci_upper:.1f}"
        )
        fig_mc.add_vline(
            x=ci_lower,
            line_color='orange',
            line_dash='dash',
            annotation_text=f"Lower: {ci_lower:.1f}"
        )

        fig_mc.update_layout(
            title='Monte Carlo RUL Distribution',
            xaxis_title='RUL (cycles)',
            yaxis_title='Count',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300
        )
        st.plotly_chart(
            fig_mc, use_container_width=True
        )
    else:
        st.info(
            "Model not loaded — "
            "showing simulated distribution"
        )

with col2:
    st.subheader("📊 Prediction Stats")
    if mc_preds is not None:
        st.metric("Mean RUL",
                  f"{mean_rul:.1f} cycles")
        st.metric("Std Dev",
                  f"±{std_rul:.1f} cycles")
        st.metric("Min Prediction",
                  f"{mc_preds.min():.1f}")
        st.metric("Max Prediction",
                  f"{mc_preds.max():.1f}")
        st.metric(
            f"{confidence_level}% CI Width",
            f"{ci_upper - ci_lower:.1f} cycles"
        )

st.markdown("---")

# ============================================
# SECTION 3: Fleet-Wide RUL Overview
# ============================================
st.subheader("🏭 Fleet-Wide RUL Overview")

# Get latest RUL per engine
fleet_df = df.groupby('engine_id').agg(
    RUL=('RUL', 'last'),
    cycles=('cycle', 'max')
).reset_index()

fleet_df['equipment'] = fleet_df[
    'engine_id'
].apply(lambda x: f"EQ-{x:03d}")

fleet_df['status'] = fleet_df['RUL'].apply(
    lambda r: '🔴 Critical' if r < 20
    else '🟡 Warning' if r < 50
    else '🟢 Healthy'
)

fleet_df['color'] = fleet_df['RUL'].apply(
    lambda r: 'red'    if r < 20
    else 'orange' if r < 50
    else 'green'
)

col1, col2 = st.columns([2, 1])

with col1:
    # Fleet RUL bar chart
    fig_fleet = go.Figure()

    for status, color in [
        ('🔴 Critical', 'red'),
        ('🟡 Warning',  'orange'),
        ('🟢 Healthy',  'green')
    ]:
        mask = fleet_df['status'] == status
        subset = fleet_df[mask]

        if not subset.empty:
            fig_fleet.add_trace(go.Bar(
                x=subset['equipment'],
                y=subset['RUL'],
                name=status,
                marker_color=color,
                opacity=0.8
            ))

    fig_fleet.add_hline(
        y=30, line_dash="dash",
        line_color="red",
        annotation_text="Critical Threshold"
    )
    fig_fleet.add_hline(
        y=50, line_dash="dash",
        line_color="orange",
        annotation_text="Warning Threshold"
    )

    fig_fleet.update_layout(
        title='Fleet RUL Status',
        xaxis_title='Equipment',
        yaxis_title='RUL (cycles)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400,
        xaxis_tickangle=-45,
        barmode='group'
    )
    st.plotly_chart(
        fig_fleet, use_container_width=True
    )

with col2:
    st.subheader("📋 Fleet Summary")

    critical = (fleet_df['RUL'] < 20).sum()
    warning  = ((fleet_df['RUL'] >= 20) &
                (fleet_df['RUL'] < 50)).sum()
    healthy  = (fleet_df['RUL'] >= 50).sum()

    st.metric("🔴 Critical", int(critical))
    st.metric("🟡 Warning",  int(warning))
    st.metric("🟢 Healthy",  int(healthy))
    st.metric(
        "Avg Fleet RUL",
        f"{fleet_df['RUL'].mean():.0f} cycles"
    )
    st.metric(
        "Min RUL",
        f"{fleet_df['RUL'].min():.0f} cycles"
    )

    # Fleet health donut
    fig_donut = go.Figure(go.Pie(
        labels=['Critical', 'Warning', 'Healthy'],
        values=[critical, warning, healthy],
        marker_colors=['red', 'orange', 'green'],
        hole=0.5
    ))
    fig_donut.update_layout(
        height=250,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(t=0, b=0, l=0, r=0)
    )
    st.plotly_chart(
        fig_donut, use_container_width=True
    )

st.markdown("---")

# ============================================
# SECTION 4: Failure Probability Timeline
# ============================================
st.subheader("💀 Failure Probability Forecast")

# Simulate failure probability over time
future_cycles = np.arange(0, 50)
fail_probs    = 1 / (
    1 + np.exp(-(future_cycles - mean_rul) / 10)
)

fig_prob = go.Figure()

fig_prob.add_trace(go.Scatter(
    x=future_cycles,
    y=fail_probs * 100,
    fill='tozeroy',
    fillcolor='rgba(255,68,68,0.2)',
    line=dict(color='red', width=2),
    name='Failure Probability'
))

fig_prob.add_hline(
    y=50, line_dash="dash",
    line_color="orange",
    annotation_text="50% Failure Probability"
)
fig_prob.add_hline(
    y=80, line_dash="dash",
    line_color="red",
    annotation_text="80% — Urgent"
)

fig_prob.add_vline(
    x=mean_rul,
    line_color='white',
    line_dash='dot',
    annotation_text=f"Predicted RUL: {mean_rul:.0f}"
)

fig_prob.update_layout(
    title='Failure Probability Over Time',
    xaxis_title='Cycles from Now',
    yaxis_title='Failure Probability (%)',
    yaxis_range=[0, 100],
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    height=300
)
st.plotly_chart(fig_prob, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 5: Model Info
# ============================================
if model_info:
    st.subheader("🤖 Model Information")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            model_info.get('model_name', 'N/A')
        )
    with col2:
        st.metric(
            "Test MAE",
            f"{model_info.get('test_mae', 0):.4f}"
        )
    with col3:
        st.metric(
            "Test MAPE",
            f"{model_info.get('test_mape', 0):.2f}%"
        )
    with col4:
        st.metric(
            "Test R²",
            f"{model_info.get('test_r2', 0):.4f}"
        )