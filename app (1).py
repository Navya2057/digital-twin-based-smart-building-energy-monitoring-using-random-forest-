"""
app.py
======
Streamlit Dashboard: Digital Twin Based Smart Building Energy Monitoring
Powered by Scikit-Learn Random Forest, Plotly, and 3-Sigma Anomaly Detection.
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Add src to system path so twin and features can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from twin import BuildingDigitalTwin
from preprocess import load_or_generate_data, clean_data
from features import create_features


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Digital Twin | Smart Building Energy",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-normal {
        color: #059669;
        font-weight: bold;
    }
    .status-anomaly {
        color: #DC2626;
        font-weight: bold;
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.3; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Caching & Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_digital_twin_system():
    model_path = os.path.join("models", "random_forest_energy_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please run 'python src/train.py' first.")
        st.stop()
    twin = BuildingDigitalTwin(model_path=model_path)
    return twin


@st.cache_data
def load_stream_dataset():
    stream_path = os.path.join("data", "test_evaluation_stream.csv")
    if os.path.exists(stream_path):
        df = pd.read_csv(stream_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # Fallback to generating evaluation stream if not precomputed
    raw_df = load_or_generate_data()
    clean_df = clean_data(raw_df)
    feat_df = create_features(clean_df)
    test_split_idx = int(len(feat_df) * 0.8)
    test_df = feat_df.iloc[test_split_idx:].copy().reset_index(drop=True)
    return test_df


twin_engine = load_digital_twin_system()
stream_data = load_stream_dataset()
model_bundle = twin_engine.bundle
anomaly_params = twin_engine.anomaly_params
metrics = model_bundle.get("metrics", {})


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "stream_index" not in st.session_state:
    st.session_state.stream_index = 24  # Start with 24 hours of data
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "simulation_speed" not in st.session_state:
    st.session_state.simulation_speed = 0.5


# ---------------------------------------------------------
# Sidebar Controls & Twin Configuration
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric/100/smart-home-shield.png", width=70)
st.sidebar.title("Digital Twin Control")
st.sidebar.caption("Smart Building Telemetry Simulator")

st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ Simulation Stream Controls")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("▶️ Play / Step", use_container_width=True):
        st.session_state.stream_index = min(len(stream_data), st.session_state.stream_index + 1)
with col_btn2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.stream_index = 24

step_slider = st.sidebar.slider(
    "Timeline Scrubber (Hour)",
    min_value=24,
    max_value=len(stream_data),
    value=st.session_state.stream_index,
    step=1,
    help="Fast-forward or rewind the telemetry stream."
)
st.session_state.stream_index = step_slider

view_window = st.sidebar.slider(
    "Chart View Window (Hours)",
    min_value=12,
    max_value=168,
    value=48,
    step=6,
    help="Number of recent hours to display on the live graph."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Anomaly Detector Settings")
sigma_mult = st.sidebar.slider(
    "Deviation Sensitivity (Sigma Multiplier)",
    min_value=1.5,
    max_value=4.0,
    value=3.0,
    step=0.1,
    help="Standard 3-Sigma (99.7% confidence interval). Lower values make detection more sensitive."
)
active_threshold = anomaly_params["residual_mean"] + sigma_mult * anomaly_params["residual_std"]

st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **Model Specs:**
    - Algorithm: Random Forest Regressor
    - Estimators: {model_bundle['best_params'].get('n_estimators', 200)}
    - Max Depth: {model_bundle['best_params'].get('max_depth', 6)}
    - Baseline 3-Sigma Cutoff: `{anomaly_params['threshold_3sigma']:.3f} kWh`
    - Adjusted Cutoff: `{active_threshold:.3f} kWh`
    """
)


# ---------------------------------------------------------
# Data Slice Calculation
# ---------------------------------------------------------
current_idx = st.session_state.stream_index
window_start = max(0, current_idx - view_window)

# Active viewing slice
display_df = stream_data.iloc[window_start:current_idx].copy()
current_row = stream_data.iloc[current_idx - 1]

# Dynamic recalculation based on sensitivity slider
display_df["threshold_active"] = active_threshold
display_df["is_active_anomaly"] = display_df["abs_deviation"] > active_threshold

# Current readings
curr_actual = float(current_row["energy"])
curr_pred = float(current_row["predicted_energy"])
curr_res = float(current_row["residual"])
curr_dev = abs(curr_res)
curr_is_anomaly = curr_dev > active_threshold

prev_actual = float(stream_data.iloc[current_idx - 2]["energy"]) if current_idx > 1 else curr_actual
actual_delta = round(curr_actual - prev_actual, 2)

total_anomalies_so_far = int((stream_data.iloc[:current_idx]["abs_deviation"] > active_threshold).sum())
total_hours_so_far = current_idx


# ---------------------------------------------------------
# Header & Live KPI Cards
# ---------------------------------------------------------
st.markdown('<div class="main-header">🏢 Digital Twin: Smart Building Energy Monitoring</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Physics-Aware Energy Forecasting & Real-Time 3-Sigma Anomaly Detection via Random Forest</div>',
    unsafe_allow_html=True,
)

# Metric Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="⚡ Actual Energy",
        value=f"{curr_actual:.2f} kWh",
        delta=f"{actual_delta:+.2f} kWh/h",
        help="Real-time physical power meter telemetry"
    )

with kpi2:
    st.metric(
        label="🤖 Twin Forecast",
        value=f"{curr_pred:.2f} kWh",
        delta=f"{curr_res:+.2f} kWh (diff)",
        delta_color="inverse",
        help="Virtual digital twin expected energy load"
    )

with kpi3:
    st.metric(
        label="📏 Residual Deviation",
        value=f"{curr_dev:.2f} kWh",
        delta=f"Threshold: {active_threshold:.2f} kWh",
        delta_color="off",
        help="|Actual - Predicted| deviation"
    )

with kpi4:
    if curr_is_anomaly:
        st.metric(
            label="🚨 Twin Health Status",
            value="CRITICAL ANOMALY",
            delta=f"+{(curr_dev - active_threshold):.2f} kWh Over Limit",
            delta_color="inverse"
        )
    else:
        st.metric(
            label="🛡️ Twin Health Status",
            value="NOMINAL (NORMAL)",
            delta="Within Bounds",
            delta_color="normal"
        )

with kpi5:
    rf_r2 = metrics.get("random_forest", {}).get("r2", 0.4504)
    st.metric(
        label="🎯 Model Accuracy (R²)",
        value=f"{rf_r2*100:.1f}%",
        delta=f"Total Alerts: {total_anomalies_so_far}",
        delta_color="off",
        help="Test Set R² Score and cumulative flagged anomaly count"
    )

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Tabs Layout
# ---------------------------------------------------------
tab_stream, tab_anomalies, tab_whatif, tab_model_eval = st.tabs([
    "📈 Live Digital Twin Stream",
    "🚨 Anomaly Log & Diagnostics",
    "🧪 What-If Scenario Sandbox",
    "📊 Model Benchmark & Feature Importance",
])


# ---------------------------------------------------------
# TAB 1: Live Digital Twin Stream
# ---------------------------------------------------------
with tab_stream:
    st.subheader(f"Telemetry Stream — Last {len(display_df)} Hours (Current: {current_row['timestamp']})")
    
    # 1. Main Time Series Plot
    fig_stream = go.Figure()

    # Actual Energy
    fig_stream.add_trace(go.Scatter(
        x=display_df["timestamp"],
        y=display_df["energy"],
        mode="lines+markers",
        name="Physical Building (Actual kWh)",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=4)
    ))

    # Predicted Energy
    fig_stream.add_trace(go.Scatter(
        x=display_df["timestamp"],
        y=display_df["predicted_energy"],
        mode="lines",
        name="Digital Twin (Predicted kWh)",
        line=dict(color="#10B981", width=2.5, dash="dash")
    ))

    # Upper 3-Sigma Bound
    upper_bound = display_df["predicted_energy"] + active_threshold
    fig_stream.add_trace(go.Scatter(
        x=display_df["timestamp"],
        y=upper_bound,
        mode="lines",
        name=f"Upper Bound (+{sigma_mult}σ)",
        line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dot"),
        hoverinfo="skip"
    ))

    # Anomaly Markers
    anomalies_df = display_df[display_df["is_active_anomaly"]]
    if not anomalies_df.empty:
        fig_stream.add_trace(go.Scatter(
            x=anomalies_df["timestamp"],
            y=anomalies_df["energy"],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                color="#EF4444",
                size=12,
                symbol="triangle-up",
                line=dict(color="#991B1B", width=2)
            ),
            text=[f"ANOMALY!<br>Actual: {row.energy} kWh<br>Predicted: {row.predicted_energy} kWh<br>Deviation: {row.abs_deviation:.2f} kWh" for _, row in anomalies_df.iterrows()],
            hoverinfo="text"
        ))

    fig_stream.update_layout(
        title="Real-Time Building Telemetry vs. Digital Twin Forecast",
        xaxis_title="Timestamp",
        yaxis_title="Energy Consumption (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_stream, use_container_width=True)

    # 2. Residual Deviation Plot
    col_chart2, col_env = st.columns([2, 1])

    with col_chart2:
        fig_dev = go.Figure()
        
        # Deviation Bars
        bar_colors = ["#EF4444" if dev > active_threshold else "#94A3B8" for dev in display_df["abs_deviation"]]
        fig_dev.add_trace(go.Bar(
            x=display_df["timestamp"],
            y=display_df["abs_deviation"],
            name="|Actual - Predicted|",
            marker_color=bar_colors
        ))
        
        # Threshold line
        fig_dev.add_hline(
            y=active_threshold,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"Anomaly Threshold ({active_threshold:.2f} kWh)",
            annotation_position="top left"
        )
        
        fig_dev.update_layout(
            title="Residual Deviation & 3-Sigma Anomaly Cutoff",
            xaxis_title="Timestamp",
            yaxis_title="Absolute Error |kWh|",
            template="plotly_white",
            height=280,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_dev, use_container_width=True)

    with col_env:
        st.markdown("**Current Environmental & Occupancy Telemetry:**")
        st.write(f"- 🌡️ **Ambient Temperature:** `{current_row['temperature']:.1f} °C`")
        st.write(f"- 💧 **Relative Humidity:** `{current_row['humidity']:.1f} %`")
        st.write(f"- 👥 **Building Occupancy:** `{int(current_row['occupancy'])} people`")
        st.write(f"- 🕒 **Time:** `{pd.to_datetime(current_row['timestamp']).strftime('%A, %H:00')}`")
        st.write(f"- 📅 **Weekend:** `{'Yes' if current_row['is_weekend'] == 1 else 'No'}`")
        
        if curr_is_anomaly:
            st.error(
                f"⚠️ **ALERT TRIGGERED:** Residual deviation of `{curr_dev:.2f} kWh` exceeds the `{active_threshold:.2f} kWh` bound!"
            )
        else:
            st.success("✅ **STATUS STABLE:** Physical building matches Digital Twin forecast.")


# ---------------------------------------------------------
# TAB 2: Anomaly Log & Incident Diagnostics
# ---------------------------------------------------------
with tab_anomalies:
    st.subheader("🚨 Incident Diagnostic Log")
    st.caption("Historical record of all anomalies flagged by the Digital Twin 3-Sigma detector.")

    historical_anomalies = stream_data.iloc[:current_idx][
        stream_data.iloc[:current_idx]["abs_deviation"] > active_threshold
    ].copy()

    if historical_anomalies.empty:
        st.info("No anomalies detected in the streaming window so far.")
    else:
        # Format table
        table_df = historical_anomalies[[
            "timestamp", "energy", "predicted_energy", "residual", "abs_deviation", "temperature", "occupancy"
        ]].copy()
        
        table_df["anomaly_type"] = np.where(table_df["residual"] > 0, "⚡ POWER SURGE", "📉 ABNORMAL DROP")
        table_df["z_score"] = ((table_df["abs_deviation"] - anomaly_params["residual_mean"]) / anomaly_params["residual_std"]).round(2)
        table_df["diagnosis"] = np.where(
            table_df["residual"] > 0,
            "HVAC overload, stuck chiller, or unauthorized equipment operation",
            "Circuit trip, grid supply interruption, or sensor malfunction"
        )
        table_df.rename(columns={
            "energy": "Actual (kWh)",
            "predicted_energy": "Predicted (kWh)",
            "residual": "Residual (kWh)",
            "abs_deviation": "Deviation (kWh)",
            "temperature": "Temp (°C)",
            "occupancy": "Occupants",
            "anomaly_type": "Classification",
            "z_score": "Z-Score",
            "diagnosis": "Suspected Cause"
        }, inplace=True)

        st.dataframe(
            table_df.sort_values(by="timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        csv_data = table_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Anomaly Log (CSV)",
            data=csv_data,
            file_name=f"building_anomalies_log_{int(time.time())}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# TAB 3: What-If Scenario Sandbox
# ---------------------------------------------------------
with tab_whatif:
    st.subheader("🧪 Digital Twin 'What-If' Scenario Simulator")
    st.markdown(
        """
        The power of a **Digital Twin** lies in counterfactual testing: simulate how building energy
        consumption would react to external weather events, modified schedules, or occupancy changes
        **before** they occur in the real physical building.
        """
    )

    c_scen1, c_scen2 = st.columns(2)

    with c_scen1:
        st.markdown("##### 🎛️ Scenario Inputs")
        sim_temp = st.slider("Simulated Temperature (°C)", min_value=-5.0, max_value=40.0, value=25.0, step=0.5)
        sim_humid = st.slider("Simulated Relative Humidity (%)", min_value=15.0, max_value=100.0, value=60.0, step=1.0)
        sim_occ = st.slider("Building Occupancy (people)", min_value=0, max_value=100, value=50, step=1)
        sim_hour = st.slider("Hour of the Day", min_value=0, max_value=23, value=14, step=1)
        sim_weekend = st.radio("Day Type", options=["Weekday (Business Day)", "Weekend"], horizontal=True)
        is_wknd = 1 if "Weekend" in sim_weekend else 0

    with c_scen2:
        st.markdown("##### 🔮 Digital Twin Forecast & Insights")
        
        whatif_out = twin_engine.what_if_simulation(
            temperature=sim_temp,
            humidity=sim_humid,
            occupancy=sim_occ,
            hour=sim_hour,
            is_weekend=is_wknd,
            base_energy_reference=float(stream_data["energy"].mean())
        )
        
        sim_pred_rf = whatif_out["predicted_energy"]
        sim_pred_base = whatif_out["baseline_predicted_energy"]
        sim_cost = whatif_out["estimated_hourly_cost_usd"]
        sim_co2 = whatif_out["estimated_co2_kg"]

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.metric("Predicted Energy", f"{sim_pred_rf:.2f} kWh", help="Random Forest model forecast")
            st.metric("Estimated Cost", f"${sim_cost:.2f} / hr", help="Assuming average tariff $0.18/kWh")
        with col_w2:
            st.metric("Baseline Linear Forecast", f"{sim_pred_base:.2f} kWh", help="Linear regression reference")
            st.metric("Carbon Footprint", f"{sim_co2:.2f} kg CO₂ / hr", help="Assuming 0.42 kg CO2 / kWh")

        # Thermodynamic Insight Note
        st.markdown("---")
        st.markdown("**🧠 Thermodynamic Twin Insight:**")
        if sim_temp > 28:
            st.warning("⚠️ High ambient temperature triggers heavy HVAC chiller cooling demand.")
        elif sim_temp < 10:
            st.info("❄️ Low ambient temperature activates building space heating loops.")
        else:
            st.success("🌿 Moderate temperature operates within the thermodynamic comfort band.")

        if is_wknd and sim_occ > 20:
            st.warning("⚠️ High occupancy detected during weekend hours; ensure scheduled ventilation matches load.")


# ---------------------------------------------------------
# TAB 4: Model Benchmark & Feature Importance
# ---------------------------------------------------------
with tab_model_eval:
    st.subheader("📊 Machine Learning Performance & Explainability")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("##### 🏆 Model Comparison on Test Set")
        comp_df = pd.DataFrame([
            {
                "Model": "Baseline Linear Regression",
                "MAE (kWh)": metrics.get("baseline", {}).get("mae", 0.198),
                "RMSE (kWh)": metrics.get("baseline", {}).get("rmse", 0.310),
                "R² Score": metrics.get("baseline", {}).get("r2", 0.424),
            },
            {
                "Model": "Tuned Random Forest Regressor",
                "MAE (kWh)": metrics.get("random_forest", {}).get("mae", 0.192),
                "RMSE (kWh)": metrics.get("random_forest", {}).get("rmse", 0.303),
                "R² Score": metrics.get("random_forest", {}).get("r2", 0.450),
            }
        ])
        st.table(comp_df)

        st.markdown(
            """
            **Why Random Forest Outperforms Linear Regression:**
            - **Non-Linear Dynamics:** Building energy has non-linear thermal laws (HVAC cooling load grows non-linearly when temperature deviates from 22°C).
            - **Feature Interactions:** Human occupancy amplifies the effect of high temperature and time of day.
            - **Robustness to Spikes:** Tree ensembles do not overfit to single anomalous spikes compared to least-squares regression.
            """
        )

    with col_m2:
        st.markdown("##### 🌲 Random Forest Feature Importance")
        feat_importances = model_bundle.get("feature_importances", [])
        if feat_importances:
            fi_df = pd.DataFrame(feat_importances)
            fig_fi = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Relative Feature Importance (Gini Importance)",
                labels={"importance": "Relative Importance", "feature": "Feature Name"},
                color="importance",
                color_continuous_scale="Blues"
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=350, template="plotly_white")
            st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.subheader("📐 Anomaly Detection Formula (3-Sigma Rule)")
    res_m = anomaly_params["residual_mean"]
    res_s = anomaly_params["residual_std"]
    res_t = anomaly_params["threshold_3sigma"]

    st.markdown(
        fr"""
        The Digital Twin quantifies the model residual $e_t$ for every incoming observation:

        $$e_t = |y_t - \hat{{y}}_t|$$

        where $y_t$ is the physical sensor measurement and $\hat{{y}}_t$ is the Random Forest forecast.

        The statistical baseline parameters are calibrated during nominal training:
        - **Residual Mean ($\mu_{{|e|}}$):** `{res_m:.4f}` kWh
        - **Residual Standard Deviation ($\sigma_{{|e|}}$):** `{res_s:.4f}` kWh

        An anomaly alarm is triggered whenever:

        $$e_t > \mu_{{|e|}} + 3 \cdot \sigma_{{|e|}} \quad \Longrightarrow \quad e_t > {res_t:.4f} \text{{ kWh}}$$
        """
    )
