# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Custom Imports (Modular Structure)
import styles
import utils

# --- 1. CONFIG ---
st.set_page_config(
    page_title="Global Growth AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
st.markdown(styles.load_css(), unsafe_allow_html=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. SIDEBAR ---
st.sidebar.title("Global Growth AI")
st.sidebar.header("📍 Region Selection")

country_map = {
    "United States": "US", "China": "CN", "India": "IN", 
    "Pakistan": "PK", "Germany": "DE", "Brazil": "BR",
    "United Kingdom": "GB", "Japan": "JP"
}
selected_country = st.sidebar.selectbox("Choose Economy", list(country_map.keys()))
country_code = country_map[selected_country]

st.sidebar.caption("⚠️ Note: Live API analysis restricted to major economies.")
st.sidebar.markdown("---")

st.sidebar.header("🎛️ Policy Simulation")
st.sidebar.info("Adjust parameters to trigger AI Forecast")

# Inputs
p_enroll = st.sidebar.slider("Primary Enrollment (%)", 50, 100, 85)
s_enroll = st.sidebar.slider("Secondary Enrollment (%)", 0, 100, 60)
hci_score = st.sidebar.slider("Human Capital Index (0-1)", 0.0, 1.0, 0.55)

# Validation Logic
if s_enroll > p_enroll:
    st.sidebar.error("⚠️ Secondary enrollment cannot exceed Primary.")

# --- 3. MAIN DASHBOARD ---
st.markdown(f"## 🌍 Economic Intelligence Unit: **{selected_country}**")

# Data Fetching
with st.spinner(f"Connecting to World Bank API for {selected_country}..."):
    df_gdp = utils.get_world_bank_data(country_code, "NY.GDP.MKTP.KD.ZG")

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    if not df_gdp.empty:
        latest_growth = df_gdp['value'].iloc[-1]
        
        # --- THE REAL AI IMPLEMENTATION ---
        # Instead of arithmetic, we ask the Random Forest model for the impact
        ai_impact = utils.ai_engine.predict_impact(p_enroll, s_enroll, hci_score)
        
        # Scaling the impact to be realistic relative to the country's baseline
        predicted_growth = latest_growth + (ai_impact / 10) 
        
        # KPIS
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"""<div class="metric-card"><div class="metric-label">Current Trend</div>
            <div class="metric-value">{latest_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            
        k2.markdown(f"""<div class="metric-card"><div class="metric-label">AI Forecast</div>
            <div class="metric-value" style="color: #1976d2;">{predicted_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            
        delta_color = "green" if ai_impact > 0 else "red"
        k3.markdown(f"""<div class="metric-card"><div class="metric-label">Policy Effect</div>
            <div class="metric-value" style="color: {delta_color};">{ai_impact/10:+.2f}%</div></div>""", unsafe_allow_html=True)
            
        # CHARTING
        st.subheader("📈 GDP Forecast Trajectory")
        fig = go.Figure()
        
        # Historical Line
        fig.add_trace(go.Scatter(x=df_gdp['date'], y=df_gdp['value'], mode='lines', 
                                 name='Historical', line=dict(color='#9e9e9e', width=2)))
        
        # Projected Point
        last_year = df_gdp['date'].iloc[-1]
        fig.add_trace(go.Scatter(x=[last_year, last_year+1, last_year+2], 
                                 y=[latest_growth, predicted_growth, predicted_growth + (ai_impact/20)],
                                 mode='lines+markers', name='AI Projection', 
                                 line=dict(color='#2e7d32', width=3, dash='dash')))
        
        fig.update_layout(plot_bgcolor='white', height=350, margin=dict(t=20, b=20, l=20, r=20),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
        st.plotly_chart(fig, use_container_width=True)
        
        # AI RECOMMENDATIONS
        st.markdown("### 🧠 Strategic Briefing")
        recommendations = utils.get_recommendations(p_enroll, s_enroll, hci_score, predicted_growth)
        
        c1, c2 = st.columns(2)
        if len(recommendations) > 0:
            c1.markdown(f"""<div class="rec-card"><div class="rec-title">{recommendations[0]['title']}</div>
            <div class="rec-body">{recommendations[0]['body']}</div></div>""", unsafe_allow_html=True)
        
        if len(recommendations) > 1:
            c2.markdown(f"""<div class="rec-card"><div class="rec-title">{recommendations[1]['title']}</div>
            <div class="rec-body">{recommendations[1]['body']}</div></div>""", unsafe_allow_html=True)

    else:
        st.error("Unable to retrieve live data. The World Bank API may be down.")

with col_right:
    st.markdown("### 🤖 Assistant")
    st.markdown("---")
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
    if prompt := st.chat_input("Ask about the forecast..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Simple response logic (can be upgraded to OpenAI later)
        response = f"Analyzing {selected_country}... Based on HCI {hci_score}, the AI predicts {predicted_growth:.2f}% growth."
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# --- 4. FOOTER ---
st.markdown("""
<div class="footer-container">
    <div class="footer-author">DEVELOPED BY ALI SHER KHAN TAREEN</div>
    <div class="footer-note">Prediction based on real-time World Bank Data & Random Forest Regression.</div>
</div>
""", unsafe_allow_html=True)