# src/dashboard/pages/3_alerts.py
# ============================================
# Alert Management & Rule Builder Page
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Alert Management",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Alert Management Center")
st.markdown(
    "Configure alert rules, manage notifications "
    "and monitor active alerts"
)
st.markdown("---")

# ============================================
# Initialize Session State
# ============================================
if 'alert_rules' not in st.session_state:
    st.session_state.alert_rules = [
        {
            'id'         : 1,
            'name'       : 'Critical RUL Alert',
            'sensor'     : 'RUL',
            'condition'  : 'less_than',
            'threshold'  : 20,
            'severity'   : 'Critical',
            'channels'   : ['Email', 'SMS'],
            'active'     : True,
            'created_at' : '2024-01-01 08:00:00'
        },
        {
            'id'         : 2,
            'name'       : 'Warning RUL Alert',
            'sensor'     : 'RUL',
            'condition'  : 'less_than',
            'threshold'  : 50,
            'severity'   : 'Warning',
            'channels'   : ['Email'],
            'active'     : True,
            'created_at' : '2024-01-01 08:00:00'
        },
        {
            'id'         : 3,
            'name'       : 'High Temp Alert',
            'sensor'     : 'sensor_2',
            'condition'  : 'greater_than',
            'threshold'  : 650.0,
            'severity'   : 'Warning',
            'channels'   : ['Slack'],
            'active'     : True,
            'created_at' : '2024-01-01 08:00:00'
        }
    ]

if 'alert_history' not in st.session_state:
    # Generate sample alert history
    np.random.seed(42)
    history  = []
    base_time= datetime.now() - timedelta(days=7)

    for i in range(30):
        severity = np.random.choice(
            ['Critical', 'Warning', 'Info'],
            p=[0.2, 0.5, 0.3]
        )
        history.append({
            'id'          : i + 1,
            'timestamp'   : (
                base_time + timedelta(
                    hours=np.random.randint(0, 168)
                )
            ).strftime('%Y-%m-%d %H:%M:%S'),
            'equipment'   : f"EQ-{np.random.randint(1,101):03d}",
            'sensor'      : np.random.choice([
                'RUL', 'sensor_2',
                'sensor_7', 'sensor_11'
            ]),
            'severity'    : severity,
            'message'     : (
                f"RUL below threshold"
                if severity == 'Critical'
                else f"Sensor anomaly detected"
            ),
            'resolved'    : np.random.choice(
                [True, False], p=[0.7, 0.3]
            )
        })

    st.session_state.alert_history = history

# ============================================
# Load Sensor Data
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

df = load_data()

sensor_options = ['RUL'] + [
    f'sensor_{i}' for i in range(1, 22)
    if f'sensor_{i}' in df.columns
    and df[f'sensor_{i}'].std() > 0.001
]

# ============================================
# Alert Summary Banner
# ============================================
rules      = st.session_state.alert_rules
history    = st.session_state.alert_history
history_df = pd.DataFrame(history)

active_rules    = sum(1 for r in rules if r['active'])
critical_alerts = sum(
    1 for h in history
    if h['severity'] == 'Critical'
    and not h['resolved']
)
warning_alerts  = sum(
    1 for h in history
    if h['severity'] == 'Warning'
    and not h['resolved']
)
resolved_alerts = sum(
    1 for h in history if h['resolved']
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Active Rules",   active_rules)
with col2:
    st.metric("🔴 Critical",   critical_alerts)
with col3:
    st.metric("🟡 Warning",    warning_alerts)
with col4:
    st.metric("✅ Resolved",   resolved_alerts)
with col5:
    st.metric("Total History", len(history))

st.markdown("---")

# ============================================
# Tabs
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Alert Rules",
    "🔔 Active Alerts",
    "📜 Alert History",
    "📊 Alert Analytics"
])

# ============================================
# TAB 1: Alert Rule Builder
# ============================================
with tab1:
    st.subheader("⚙️ Alert Rule Builder")

    col_form, col_rules = st.columns([1, 2])

    with col_form:
        st.markdown("### ➕ Create New Rule")

        with st.form("alert_rule_form"):
            rule_name = st.text_input(
                "Rule Name:",
                placeholder="e.g. Critical Vibration Alert"
            )

            sensor = st.selectbox(
                "Select Sensor:",
                sensor_options
            )

            condition = st.selectbox(
                "Condition:",
                [
                    "greater_than",
                    "less_than",
                    "equals",
                    "not_equals"
                ],
                format_func=lambda x: {
                    'greater_than': '> Greater Than',
                    'less_than'   : '< Less Than',
                    'equals'      : '= Equals',
                    'not_equals'  : '≠ Not Equals'
                }[x]
            )

            # Smart default threshold
            if sensor == 'RUL':
                default_thresh = 30.0
            elif sensor in df.columns:
                default_thresh = float(
                    df[sensor].mean()
                )
            else:
                default_thresh = 0.0

            threshold = st.number_input(
                "Threshold Value:",
                value=default_thresh,
                format="%.4f"
            )

            severity = st.selectbox(
                "Severity Level:",
                ["Critical", "Warning", "Info"],
                help=(
                    "Critical=immediate action, "
                    "Warning=plan maintenance, "
                    "Info=monitor"
                )
            )

            channels = st.multiselect(
                "Notification Channels:",
                ["Email", "SMS",
                 "Slack", "PagerDuty"],
                default=["Email"]
            )

            col_a, col_b = st.columns(2)
            with col_a:
                active = st.checkbox(
                    "Active", value=True
                )
            with col_b:
                cooldown = st.number_input(
                    "Cooldown (min):",
                    min_value=1,
                    max_value=60,
                    value=5
                )

            submitted = st.form_submit_button(
                "💾 Save Rule",
                use_container_width=True
            )

            if submitted:
                if rule_name and channels:
                    new_rule = {
                        'id'        : len(rules) + 1,
                        'name'      : rule_name,
                        'sensor'    : sensor,
                        'condition' : condition,
                        'threshold' : threshold,
                        'severity'  : severity,
                        'channels'  : channels,
                        'active'    : active,
                        'cooldown'  : cooldown,
                        'created_at': datetime.now()\
                            .strftime('%Y-%m-%d %H:%M:%S')
                    }
                    st.session_state\
                      .alert_rules.append(new_rule)
                    st.success(
                        f"✅ Rule '{rule_name}' saved!"
                    )
                    st.rerun()
                else:
                    st.error(
                        "❌ Please fill all fields!"
                    )

    with col_rules:
        st.markdown("### 📋 Existing Rules")

        for rule in st.session_state.alert_rules:
            severity_color = {
                'Critical': '🔴',
                'Warning' : '🟡',
                'Info'    : '🔵'
            }.get(rule['severity'], '⚪')

            status = (
                "✅ Active" if rule['active']
                else "⏸️ Paused"
            )

            condition_str = {
                'greater_than': '>',
                'less_than'   : '<',
                'equals'      : '=',
                'not_equals'  : '≠'
            }.get(rule['condition'], '?')

            with st.expander(
                f"{severity_color} "
                f"{rule['name']} — {status}"
            ):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(
                        f"**Sensor:** `{rule['sensor']}`"
                    )
                    st.markdown(
                        f"**Condition:** "
                        f"`{rule['sensor']} "
                        f"{condition_str} "
                        f"{rule['threshold']}`"
                    )
                    st.markdown(
                        f"**Severity:** "
                        f"{rule['severity']}"
                    )

                with col_b:
                    st.markdown(
                        f"**Channels:** "
                        f"`{', '.join(rule['channels'])}`"
                    )
                    st.markdown(
                        f"**Created:** "
                        f"{rule['created_at']}"
                    )

                col_toggle, col_delete = \
                    st.columns(2)

                with col_toggle:
                    if st.button(
                        "⏸️ Pause" if rule['active']
                        else "▶️ Activate",
                        key=f"toggle_{rule['id']}"
                    ):
                        for r in st.session_state\
                                    .alert_rules:
                            if r['id'] == rule['id']:
                                r['active'] = \
                                    not r['active']
                        st.rerun()

                with col_delete:
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{rule['id']}"
                    ):
                        st.session_state\
                          .alert_rules = [
                            r for r in
                            st.session_state.alert_rules
                            if r['id'] != rule['id']
                        ]
                        st.rerun()

# ============================================
# TAB 2: Active Alerts
# ============================================
with tab2:
    st.subheader("🔔 Active Alerts")

    # Simulate active alerts from data
    latest_rul = df.groupby('engine_id')[
        'RUL'
    ].last().reset_index()

    active_alerts = []

    for _, row in latest_rul.iterrows():
        rul = row['RUL']
        eq  = f"EQ-{int(row['engine_id']):03d}"

        if rul < 20:
            active_alerts.append({
                'Equipment' : eq,
                'Sensor'    : 'RUL',
                'Value'     : f"{rul:.0f} cycles",
                'Severity'  : '🔴 Critical',
                'Rule'      : 'Critical RUL Alert',
                'Since'     : datetime.now()\
                    .strftime('%H:%M:%S'),
                'Action'    : '🔧 Immediate'
            })
        elif rul < 50:
            active_alerts.append({
                'Equipment' : eq,
                'Sensor'    : 'RUL',
                'Value'     : f"{rul:.0f} cycles",
                'Severity'  : '🟡 Warning',
                'Rule'      : 'Warning RUL Alert',
                'Since'     : datetime.now()\
                    .strftime('%H:%M:%S'),
                'Action'    : '📅 Schedule'
            })

    if active_alerts:
        alerts_df = pd.DataFrame(active_alerts)

        # Filter by severity
        filter_sev = st.multiselect(
            "Filter by Severity:",
            ['🔴 Critical', '🟡 Warning'],
            default=['🔴 Critical', '🟡 Warning']
        )

        filtered = alerts_df[
            alerts_df['Severity'].isin(filter_sev)
        ]

        st.dataframe(
            filtered,
            use_container_width=True,
            height=400
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "✅ Acknowledge All Critical"
            ):
                st.success(
                    "✅ All critical alerts acknowledged!"
                )
        with col2:
            if st.button("📧 Send Summary Email"):
                st.success(
                    "📧 Alert summary sent!"
                )
    else:
        st.success("✅ No active alerts!")

    # Alert notification test
    st.markdown("---")
    st.subheader("🔔 Test Notification")

    col1, col2, col3 = st.columns(3)
    with col1:
        test_channel = st.selectbox(
            "Channel:",
            ["Email", "SMS", "Slack", "PagerDuty"]
        )
    with col2:
        test_severity = st.selectbox(
            "Severity:",
            ["Critical", "Warning", "Info"]
        )
    with col3:
        test_msg = st.text_input(
            "Message:",
            value="Test alert from PredMaint"
        )

    if st.button("📤 Send Test Notification"):
        st.success(
            f"✅ Test {test_severity} alert sent "
            f"via {test_channel}!\n"
            f"Message: '{test_msg}'"
        )

# ============================================
# TAB 3: Alert History
# ============================================
with tab3:
    st.subheader("📜 Alert History")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_severity = st.multiselect(
            "Severity:",
            ['Critical', 'Warning', 'Info'],
            default=['Critical', 'Warning', 'Info']
        )
    with col2:
        filter_status = st.multiselect(
            "Status:",
            ['Resolved', 'Unresolved'],
            default=['Resolved', 'Unresolved']
        )
    with col3:
        filter_eq = st.text_input(
            "Equipment ID (optional):",
            placeholder="e.g. EQ-001"
        )

    # Apply filters
    filtered_history = history_df[
        history_df['severity'].isin(filter_severity)
    ]

    if 'Resolved' in filter_status and \
       'Unresolved' not in filter_status:
        filtered_history = filtered_history[
            filtered_history['resolved'] == True
        ]
    elif 'Unresolved' in filter_status and \
         'Resolved' not in filter_status:
        filtered_history = filtered_history[
            filtered_history['resolved'] == False
        ]

    if filter_eq:
        filtered_history = filtered_history[
            filtered_history['equipment'].str.contains(
                filter_eq.upper()
            )
        ]

    # Format for display
    display_df = filtered_history.copy()
    display_df['status'] = display_df[
        'resolved'
    ].apply(
        lambda x: '✅ Resolved'
        if x else '🔴 Active'
    )
    display_df['severity'] = display_df[
        'severity'
    ].apply(
        lambda x:
        '🔴 Critical' if x == 'Critical'
        else '🟡 Warning' if x == 'Warning'
        else '🔵 Info'
    )

    st.dataframe(
        display_df[[
            'timestamp', 'equipment',
            'sensor', 'severity',
            'message', 'status'
        ]].sort_values(
            'timestamp', ascending=False
        ),
        use_container_width=True,
        height=400
    )

    # Export button
    csv = display_df.to_csv(index=False)
    st.download_button(
        "📥 Export History (CSV)",
        csv,
        "alert_history.csv",
        "text/csv"
    )

# ============================================
# TAB 4: Alert Analytics
# ============================================
with tab4:
    st.subheader("📊 Alert Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Alerts by severity
        sev_counts = history_df[
            'severity'
        ].value_counts()

        fig1 = go.Figure(go.Pie(
            labels=sev_counts.index,
            values=sev_counts.values,
            marker_colors=['red', 'orange', 'blue'],
            hole=0.4
        ))
        fig1.update_layout(
            title='Alerts by Severity',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300
        )
        st.plotly_chart(
            fig1, use_container_width=True
        )

    with col2:
        # Resolution rate
        resolved   = history_df['resolved'].sum()
        unresolved = len(history_df) - resolved

        fig2 = go.Figure(go.Pie(
            labels=['Resolved', 'Unresolved'],
            values=[resolved, unresolved],
            marker_colors=['green', 'red'],
            hole=0.4
        ))
        fig2.update_layout(
            title='Resolution Rate',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300
        )
        st.plotly_chart(
            fig2, use_container_width=True
        )

    # Alerts over time
    st.subheader("📈 Alert Frequency Over Time")

    history_df['date'] = pd.to_datetime(
        history_df['timestamp']
    ).dt.date

    daily_alerts = history_df.groupby(
        ['date', 'severity']
    ).size().reset_index(name='count')

    fig3 = go.Figure()

    for sev, color in [
        ('Critical', 'red'),
        ('Warning',  'orange'),
        ('Info',     'blue')
    ]:
        mask = daily_alerts['severity'] == sev
        data = daily_alerts[mask]

        if not data.empty:
            fig3.add_trace(go.Bar(
                x=data['date'],
                y=data['count'],
                name=sev,
                marker_color=color,
                opacity=0.8
            ))

    fig3.update_layout(
        title='Daily Alert Frequency',
        xaxis_title='Date',
        yaxis_title='Alert Count',
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Top alerting equipment
    st.subheader("🏭 Top Alerting Equipment")

    eq_alerts = history_df.groupby(
        'equipment'
    ).size().reset_index(name='count')\
     .sort_values('count', ascending=False)\
     .head(10)

    fig4 = go.Figure(go.Bar(
        x=eq_alerts['count'],
        y=eq_alerts['equipment'],
        orientation='h',
        marker_color='steelblue',
        opacity=0.8
    ))
    fig4.update_layout(
        title='Top 10 Equipment by Alert Count',
        xaxis_title='Alert Count',
        yaxis_title='Equipment',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=350
    )
    st.plotly_chart(fig4, use_container_width=True)

    # MTTR (Mean Time to Resolve)
    st.subheader("⏱️ Key Alert Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        resolution_rate = (
            resolved / len(history_df) * 100
        )
        st.metric(
            "Resolution Rate",
            f"{resolution_rate:.1f}%"
        )
    with col2:
        st.metric(
            "Total Alerts (7 days)",
            len(history_df)
        )
    with col3:
        st.metric(
            "Critical Alerts",
            int(sev_counts.get('Critical', 0))
        )
    with col4:
        daily_avg = len(history_df) / 7
        st.metric(
            "Daily Average",
            f"{daily_avg:.1f} alerts/day"
        )

    # Save rules button
    st.markdown("---")
    if st.button("💾 Export Alert Rules"):
        rules_json = json.dumps(
            st.session_state.alert_rules,
            indent=2
        )
        st.download_button(
            "📥 Download Rules (JSON)",
            rules_json,
            "alert_rules.json",
            "application/json"
        )