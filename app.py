# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# Custom Modules
import styles
import utils

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Global Growth AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(styles.load_css(), unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.title("Global Growth AI")
st.sidebar.caption("v2.0 Enterprise Edition")
st.sidebar.markdown("---")

# Expanded Country Selection
st.sidebar.header("📍 Economy")
# Using the full dictionary from utils.py
selected_country = st.sidebar.selectbox("Select Market", list(utils.COUNTRY_MAP.keys()))
country_code = utils.COUNTRY_MAP[selected_country]

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Policy Simulation")

# --- THE FORM (Adds "Run Simulation" Button) ---
with st.sidebar.form("policy_form"):
    st.info("Configure parameters and click Run.")
    
    p_enroll = st.slider("Primary Enrollment (%)", 50, 100, 85)
    s_enroll = st.slider("Secondary Enrollment (%)", 0, 100, 60)
    hci_score = st.slider("Human Capital Index (0-1)", 0.0, 1.0, 0.55)
    
    # The Submit Button
    run_simulation = st.form_submit_button("🚀 Run Simulation")

if run_simulation:
    st.session_state.simulation_run = True
    st.session_state.p_enroll = p_enroll
    st.session_state.s_enroll = s_enroll
    st.session_state.hci_score = hci_score

# --- 3. MAIN DASHBOARD ---
st.markdown(f"## 🌍 Economic Intelligence Unit: **{selected_country}**")

# Fetch Data
with st.spinner(f"Retrieving macro-economic data for {selected_country}..."):
    df_gdp = utils.get_world_bank_data(country_code, "NY.GDP.MKTP.KD.ZG")

col_main, col_chat = st.columns([2, 1], gap="large")

with col_main:
    if not df_gdp.empty:
        latest_growth = df_gdp['value'].iloc[-1]
        
        # DEFAULT VIEW (Before Simulation)
        if not st.session_state.simulation_run:
            st.warning("👈 Please configure the Policy Simulation in the sidebar and click 'Run Simulation'.")
            
            # Show Baseline Data Only
            k1, k2 = st.columns(2)
            k1.markdown(f"""<div class="metric-card"><div class="metric-label">Current GDP Growth</div>
                <div class="metric-value">{latest_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            k2.markdown(f"""<div class="metric-card"><div class="metric-label">Data Year</div>
                <div class="metric-value">{int(df_gdp['date'].iloc[-1])}</div></div>""", unsafe_allow_html=True)
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_gdp['date'], y=df_gdp['value'], mode='lines', 
                                     name='Historical', line=dict(color='#2d3748', width=2)))
            fig.update_layout(title="Historical Baseline", plot_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)

        # SIMULATION VIEW (After Click)
        else:
            # AI Calculation
            ai_impact = utils.ai_engine.predict_impact(
                st.session_state.p_enroll, 
                st.session_state.s_enroll, 
                st.session_state.hci_score
            )
            predicted_growth = latest_growth + (ai_impact / 10)

            # Metrics
            k1, k2, k3 = st.columns(3)
            k1.markdown(f"""<div class="metric-card"><div class="metric-label">Baseline</div>
                <div class="metric-value">{latest_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            k2.markdown(f"""<div class="metric-card"><div class="metric-label">AI Projection</div>
                <div class="metric-value" style="color: #3182ce;">{predicted_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            
            delta_color = "#38a169" if ai_impact > 0 else "#e53e3e"
            k3.markdown(f"""<div class="metric-card"><div class="metric-label">Net Impact</div>
                <div class="metric-value" style="color: {delta_color};">{ai_impact/10:+.2f}%</div></div>""", unsafe_allow_html=True)

            # Chart Comparison
            st.subheader("📈 Forecast Trajectory")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_gdp['date'], y=df_gdp['value'], mode='lines', 
                                     name='Historical', line=dict(color='#cbd5e0', width=2)))
            
            # Add Forecast Points
            last_year = df_gdp['date'].iloc[-1]
            years = [last_year, last_year+1, last_year+2]
            values = [latest_growth, predicted_growth, predicted_growth + (ai_impact/20)]
            
            fig.add_trace(go.Scatter(x=years, y=values, mode='lines+markers', name='AI Scenario', 
                                     line=dict(color='#3182ce', width=3)))
            
            fig.update_layout(plot_bgcolor='white', height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f7fafc'))
            st.plotly_chart(fig, use_container_width=True)

            # --- DOWNLOAD REPORT BUTTON ---
            st.markdown("### 📄 Executive Reporting")
            report_pdf = utils.create_pdf_report(
                selected_country, latest_growth, predicted_growth,
                st.session_state.p_enroll, st.session_state.s_enroll, st.session_state.hci_score
            )
            
            st.download_button(
                label="📥 Download Strategy Report (PDF)",
                data=report_pdf,
                file_name=f"{selected_country}_Strategy_Report.pdf",
                mime="application/pdf"
            )

    else:
        st.error("Data unavailable for this region. Please select another economy.")

# --- 4. CHATBOT SECTION ---
with col_chat:
    st.markdown("### 🤖 Strategy Assistant")
    st.markdown("---")
    
    with st.container():
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
    if prompt := st.chat_input("Ask about the simulation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Basic Context-Aware Response
        if st.session_state.simulation_run:
            response = f"Based on your simulation for {selected_country}, the HCI score of {st.session_state.hci_score} is the primary driver of the {predicted_growth:.2f}% projection."
        else:
            response = "Please run the simulation first so I can analyze the data."
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# --- 5. FOOTER ---
st.markdown("""
<div class="footer-container">
    <div class="footer-author">DEVELOPED BY ALI SHER KHAN TAREEN</div>
    <div class="footer-note">v2.0 Enterprise Edition | AI-Powered Economic Forecasting</div>
</div>
""", unsafe_allow_html=True)