import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

# --- 1. Page Configuration & Theme Settings ---
st.set_page_config(
    page_title="Tesla Premium Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist CSS injection for a cleaner aesthetic
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #334155;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Header Area ---
title_col, logo_col = st.columns([4, 1])
with title_col:
    st.title("⚡ Tesla (TSLA) Stock Prediction Platform")
    st.caption("Advanced quantitative predictive modeling engine using fast-state ARIMA validation.")

st.divider()

# --- 3. Sidebar Panel Configuration ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.write("Adjust the parameters to re-calculate the predictive baseline.")
    
    # Styled slider and input options
    train_split = st.slider("Training Horizon Split (%)", min_value=60, max_value=90, value=80, step=5) / 100.0
    
    st.divider()
    st.markdown("### Model Architecture")
    p = st.number_input("AR Lag Order (p)", min_value=0, max_value=10, value=5)
    d = st.number_input("Differencing Order (d)", min_value=0, max_value=2, value=1)
    q = st.number_input("MA Window Order (q)", min_value=0, max_value=10, value=0)

# --- 4. Data Layer Caching ---
@st.cache_data
def load_data():
    # FIXED: Path changed from '/content/TSLA.csv' to 'TSLA.csv' for GitHub deployment
    df = pd.read_csv("TSLA.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
    
    # Metric Math
    latest_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    delta_val = latest_price - prev_price
    pct_change = (delta_val / prev_price) * 100
    
    # --- 5. High-Impact KPI Layout Cards ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Latest Close Value", f"${latest_price:,.2f}", f"{delta_val:+.2f} ({pct_change:+.2f}%)")
    with m2:
        st.metric("Market Sample Depth", f"{len(df):,} Trading Days")
    with m3:
        st.metric("Asset Class", "Equity / NYSE")
    with m4:
        st.metric("Prediction Sequence", f"ARIMA({p}, {d}, {q})")

    st.write("") # Spacer

    # --- 6. Tabbed Presentation Panels ---
    tab_analytics, tab_dataset = st.tabs(["📊 Interactive Forecast Analytics", "📋 Data Repository Inspection"])

    with tab_analytics:
        # Fast Predictive Modeling Simulation Loop
        with st.spinner("Optimizing computational matrix state..."):
            split_idx = int(len(df) * train_split)
            train_data = df[0:split_idx]
            test_data = df[split_idx:]
            
            train_ar = train_data['Open'].values
            test_ar = test_data['Open'].values
            
            # Fast-append execution matrix
            model = ARIMA(train_ar, order=(p, d, q))
            model_fit = model.fit()
            
            predictions = [model_fit.forecast()[0]]
            for t in range(len(test_ar) - 1):
                model_fit = model_fit.append([test_ar[t]], refit=False)
                predictions.append(model_fit.forecast()[0])
        
        # --- 7. Upgrading to Beautiful Interactive Plotly Graphs ---
        fig = go.Figure()
        
        # Training line
        fig.add_trace(go.Scatter(
            x=train_data['Date'], y=train_data['Open'],
            mode='lines', name='Historical Training Data',
            line=dict(color='#3b82f6', width=2)
        ))
        # Actual validation target line
        fig.add_trace(go.Scatter(
            x=test_data['Date'], y=test_data['Open'],
            mode='lines', name='Actual Target Trend',
            line=dict(color='#ef4444', width=2)
        ))
        # Simulated Prediction line
        fig.add_trace(go.Scatter(
            x=test_data['Date'], y=predictions,
            mode='lines+markers', name='Model Generated Path',
            line=dict(color='#10b981', width=1.5, dash='dash'),
            marker=dict(size=4)
        ))
        
        # Layout customizations for dark, sleek minimalist aesthetic
        fig.update_layout(
            title="TSLA Trend Tracking vs. Model Synthesis",
            template="plotly_dark",
            xaxis_title="Timeline Range",
            yaxis_title="Stock Asset Value ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.toast("⚡ Computation updated successfully!", icon="✅")

    with tab_dataset:
        st.subheader("Data Registry Grid Elements")
        st.dataframe(
            df.style.background_gradient(cmap='Blues', subset=['Close']),
            use_container_width=True
        )

except FileNotFoundError:
    st.error("⚠️ System Error: Unable to locate 'TSLA.csv' inside your repository directory structure.")
