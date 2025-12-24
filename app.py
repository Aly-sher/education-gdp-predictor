# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False

# --- 2. SIDEBAR ---
st.sidebar.title("Global Growth AI")
st.sidebar.caption("v2.7 Auto-Auth Edition")
st.sidebar.markdown("---")

# --- AUTHENTICATION (SECURE) ---
# 1. Check Streamlit Cloud Secrets (Automatic)
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("✅ AI Connected (Cloud)")

# 2. Fallback: Manual Input (Local Testing)
# Value is empty to prevent GitHub Security errors
else:
    groq_api_key = st.sidebar.text_input(
        "🔑 Groq API Key", 
        type="password", 
        value="", 
        help="Enter your gsk_... key here if running locally."
    )

st.sidebar.markdown("---")
st.sidebar.header("📍 Economy")
selected_country = st.sidebar.selectbox("Select Market", list(utils.COUNTRY_MAP.keys()))
country_code = utils.COUNTRY_MAP[selected_country]

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Policy Simulation")

with st.sidebar.form("policy_form"):
    st.info("👇 Adjust levers to see future impact.")
    
    # 1. Primary Enrollment
    p_enroll = st.slider("Primary School Enrollment (%)", 50, 100, 85)
    
    # 2. Secondary Enrollment (Logic Lock)
    # Max value is strictly capped at Primary Enrollment
    max_sec = p_enroll
    default_sec = min(60, max_sec)
    
    s_enroll = st.slider(
        "Secondary School Enrollment (%)", 
        0, 
        max_sec, # <--- Dynamic Max Constraint
        default_sec,
        help="Cannot be higher than Primary Enrollment."
    )
    
    if s_enroll == p_enroll:
        st.caption("⚠️ Note: Secondary capped at Primary level.")
    
    # 3. HCI
    hci_score = st.slider("Workforce Quality (HCI Index)", 0.0, 1.0, 0.55)
    
    # 4. Target Year
    target_year = st.slider("Forecast Target Year", 2025, 2030, 2028)
    
    run_simulation = st.form_submit_button("🚀 Run Simulation")

if run_simulation:
    st.session_state.simulation_run = True
    st.session_state.p_enroll = p_enroll
    st.session_state.s_enroll = s_enroll
    st.session_state.hci_score = hci_score
    st.session_state.target_year = target_year

# --- 3. MAIN DASHBOARD ---
st.markdown(f"## 🌍 Economic Intelligence: **{selected_country}**")

# Fetch Data
with st.spinner(f"Retrieving data for {selected_country}..."):
    df_gdp = utils.get_world_bank_data(country_code, "NY.GDP.MKTP.KD.ZG")

col_main, col_chat = st.columns([2, 1], gap="large")

with col_main:
    if not df_gdp.empty:
        latest_growth = df_gdp['value'].iloc[-1]
        current_data_year = int(df_gdp['date'].iloc[-1])
        
        # --- CALCULATION ---
        if st.session_state.simulation_run:
            ai_impact = utils.ai_engine.predict_impact(
                st.session_state.p_enroll, 
                st.session_state.s_enroll, 
                st.session_state.hci_score
            )
            predicted_growth = latest_growth + (ai_impact / 10)
        else:
            predicted_growth = latest_growth
            st.info("👈 **Ready:** Click 'Run Simulation' to start.")

        # --- A. KPI CARDS ---
        if st.session_state.simulation_run:
            k1, k2, k3 = st.columns(3)
            k1.markdown(f"""<div class="metric-card"><div class="metric-label">Baseline ({current_data_year})</div>
                <div class="metric-value">{latest_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            k2.markdown(f"""<div class="metric-card"><div class="metric-label">Forecast ({st.session_state.target_year})</div>
                <div class="metric-value" style="color: #3182ce;">{predicted_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            
            delta = predicted_growth - latest_growth
            color = "#38a169" if delta > 0 else "#e53e3e"
            k3.markdown(f"""<div class="metric-card"><div class="metric-label">Net Impact</div>
                <div class="metric-value" style="color: {color};">{delta:+.2f}%</div></div>""", unsafe_allow_html=True)

        # --- B. CHART (Dynamic Time Series) ---
        st.subheader("📈 Future Growth Projection")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_gdp['date'], y=df_gdp['value'], mode='lines', 
                                 name='Historical', line=dict(color='#cbd5e0', width=2)))
        
        if st.session_state.simulation_run:
            end_year = st.session_state.target_year
            future_years = list(range(current_data_year, end_year + 1))
            future_values = np.linspace(latest_growth, predicted_growth, len(future_years))
            future_values[-1] = predicted_growth

            fig.add_trace(go.Scatter(x=future_years, y=future_values, mode='lines+markers', name='AI Forecast', 
                                     line=dict(color='#3182ce', width=4)))
            
            fig.add_annotation(x=end_year, y=predicted_growth, text=f"Target: {predicted_growth:.1f}%", showarrow=True, arrowhead=2)

        fig.update_layout(plot_bgcolor='white', height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f7fafc'))
        st.plotly_chart(fig, use_container_width=True)

        # --- C. ANALYTICS ---
        if st.session_state.simulation_run:
            st.markdown("### 🔍 Analysis Breakdown")
            tab1, tab2 = st.tabs(["🎓 Dropout Analysis", "🕸️ Economy Health"])
            
            with tab1:
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    funnel_data = pd.DataFrame({
                        "Stage": ["Primary", "Secondary", "Workforce"],
                        "Percentage": [st.session_state.p_enroll, st.session_state.s_enroll, st.session_state.s_enroll * 0.8],
                        "Color": ["#4299e1", "#2b6cb0", "#2c5282"]
                    })
                    fig_funnel = px.bar(funnel_data, x="Percentage", y="Stage", orientation='h', text_auto=True, 
                                        color="Stage", color_discrete_sequence=["#4299e1", "#2b6cb0", "#2c5282"])
                    fig_funnel.update_layout(showlegend=False, plot_bgcolor='white', height=250)
                    st.plotly_chart(fig_funnel, use_container_width=True)
                with col_c2:
                    dropout = st.session_state.p_enroll - st.session_state.s_enroll
                    st.metric("Dropout Rate", f"{dropout}%")
                    if dropout > 20: st.error("⚠️ High Talent Loss")
                    else: st.success("✅ Efficient Pipeline")

            with tab2:
                categories = ['Primary', 'Secondary', 'HCI', 'Growth']
                input_vals = [st.session_state.p_enroll/10, st.session_state.s_enroll/10, st.session_state.hci_score*10, predicted_growth+5]
                fig_radar = go.Figure(go.Scatterpolar(r=input_vals, theta=categories, fill='toself', name='Scenario'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), height=250)
                st.plotly_chart(fig_radar, use_container_width=True)

            # REPORT
            st.markdown("### 📄 Export Results")
            report = utils.create_pdf_report(selected_country, latest_growth, predicted_growth, 
                                             st.session_state.p_enroll, st.session_state.s_enroll, st.session_state.hci_score)
            st.download_button("📥 Download PDF Report", report, file_name=f"{selected_country}_Strategy.pdf")

    else:
        st.error("Data unavailable.")

# --- 4. SMART ANALYST (Llama 3) ---
with col_chat:
    st.markdown("### 🤖 Analyst")
    st.markdown("---")
    
    with st.container():
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
    if prompt := st.chat_input("Ask about the simulation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        response = ""
        
        if not st.session_state.simulation_run:
            response = "Please run the simulation first so I can analyze the data."
        else:
            # 1. Use Groq if Key exists
            if groq_api_key:
                with st.spinner("Consulting Llama 3.1..."):
                    response = utils.consult_groq_ai(
                        groq_api_key, prompt, selected_country, latest_growth, predicted_growth,
                        st.session_state.p_enroll, st.session_state.s_enroll, st.session_state.hci_score,
                        st.session_state.target_year
                    )
            # 2. Fallback logic if no key
            else:
                delta = predicted_growth - latest_growth
                if "growth" in prompt.lower():
                    response = f"I project growth reaching {predicted_growth:.2f}% by {st.session_state.target_year}."
                elif "why" in prompt.lower():
                    response = f"The main driver is your Workforce Quality score of {st.session_state.hci_score}."
                else:
                    response = "I am tracking your simulation. Add a Groq API key for deeper analysis."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# --- 5. FOOTER ---
st.markdown("""<div class="footer-container"><div class="footer-author">DEVELOPED BY ALI SHER KHAN TAREEN</div></div>""", unsafe_allow_html=True)