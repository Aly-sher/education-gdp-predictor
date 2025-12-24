# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

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
st.sidebar.caption("v2.1 Enterprise Edition")
st.sidebar.markdown("---")

# Expanded Country Selection
st.sidebar.header("📍 Economy")
selected_country = st.sidebar.selectbox("Select Market", list(utils.COUNTRY_MAP.keys()))
country_code = utils.COUNTRY_MAP[selected_country]

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Policy Simulation")

# --- THE FORM ---
with st.sidebar.form("policy_form"):
    st.info("Configure parameters and click Run.")
    
    p_enroll = st.slider("Primary Enrollment (%)", 50, 100, 85)
    s_enroll = st.slider("Secondary Enrollment (%)", 0, 100, 60)
    hci_score = st.slider("Human Capital Index (0-1)", 0.0, 1.0, 0.55)
    
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
        
        # --- A. TOP KPI CARDS ---
        if st.session_state.simulation_run:
            # AI Calculation
            ai_impact = utils.ai_engine.predict_impact(
                st.session_state.p_enroll, 
                st.session_state.s_enroll, 
                st.session_state.hci_score
            )
            predicted_growth = latest_growth + (ai_impact / 10)

            k1, k2, k3 = st.columns(3)
            k1.markdown(f"""<div class="metric-card"><div class="metric-label">Baseline Growth</div>
                <div class="metric-value">{latest_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            k2.markdown(f"""<div class="metric-card"><div class="metric-label">AI Projection</div>
                <div class="metric-value" style="color: #3182ce;">{predicted_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            
            delta_color = "#38a169" if ai_impact > 0 else "#e53e3e"
            k3.markdown(f"""<div class="metric-card"><div class="metric-label">Net Policy Impact</div>
                <div class="metric-value" style="color: {delta_color};">{ai_impact/10:+.2f}%</div></div>""", unsafe_allow_html=True)
        else:
            predicted_growth = latest_growth # Default for charts before sim
            st.info("👈 Run the simulation to see AI projections.")

        # --- B. FORECAST TRAJECTORY (Main Line Chart) ---
        st.subheader("📈 GDP Forecast Trajectory")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_gdp['date'], y=df_gdp['value'], mode='lines', 
                                 name='Historical', line=dict(color='#cbd5e0', width=2)))
        
        if st.session_state.simulation_run:
            last_year = df_gdp['date'].iloc[-1]
            years = [last_year, last_year+1, last_year+2, last_year+3]
            # Smooth curve projection
            values = [latest_growth, predicted_growth, predicted_growth * 1.05, predicted_growth * 1.08]
            
            fig.add_trace(go.Scatter(x=years, y=values, mode='lines+markers', name='AI Scenario', 
                                     line=dict(color='#3182ce', width=4, shape='spline')))
            
            # Add annotation for the target
            fig.add_annotation(x=years[-1], y=values[-1], text=f"Target: {values[-1]:.2f}%", 
                               showarrow=True, arrowhead=1)

        fig.update_layout(plot_bgcolor='white', height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f7fafc'))
        st.plotly_chart(fig, use_container_width=True)

        # --- C. DEEP DIVE ANALYTICS (New Visuals) ---
        if st.session_state.simulation_run:
            st.markdown("### 🔍 Deep Dive Analytics")
            
            tab1, tab2, tab3 = st.tabs(["🎓 Education Funnel", "🕸️ Competitiveness Radar", "🧩 Human Capital vs GDP"])
            
            # 1. Education Funnel (Bar Chart with Dropout Highlight)
            with tab1:
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    dropout_rate = st.session_state.p_enroll - st.session_state.s_enroll
                    funnel_data = pd.DataFrame({
                        "Stage": ["Primary Enrollment", "Secondary Enrollment", "Workforce Ready"],
                        "Percentage": [st.session_state.p_enroll, st.session_state.s_enroll, st.session_state.s_enroll * 0.8], # Assuming 80% grad rate
                        "Color": ["#4299e1", "#2b6cb0", "#2c5282"]
                    })
                    fig_funnel = px.bar(funnel_data, x="Percentage", y="Stage", orientation='h', 
                                        text_auto=True, title="Education Pipeline Retention",
                                        color="Stage", color_discrete_sequence=["#4299e1", "#2b6cb0", "#2c5282"])
                    fig_funnel.update_layout(showlegend=False, plot_bgcolor='white')
                    st.plotly_chart(fig_funnel, use_container_width=True)
                
                with col_c2:
                    st.markdown("#### 🚨 Dropout Analysis")
                    if dropout_rate > 15:
                        st.error(f"High Dropout Rate: {dropout_rate}%")
                        st.caption("A significant portion of students are lost before Secondary school. This creates a low-skilled labor trap.")
                    else:
                        st.success(f"Healthy Retention: {dropout_rate}%")
                        st.caption("The pipeline is strong. Focus on quality of education rather than access.")

            # 2. Radar Chart (Gap Analysis)
            with tab2:
                # Comparing User Input vs "G20 Average" (Synthetic for demo)
                categories = ['Primary Ed', 'Secondary Ed', 'HCI', 'GDP Growth', 'Innovation']
                
                # Normalize values to 0-10 scale for Radar
                input_values = [
                    st.session_state.p_enroll / 10, 
                    st.session_state.s_enroll / 10, 
                    st.session_state.hci_score * 10, 
                    max(0, predicted_growth + 5), # Shift to positive scale
                    (st.session_state.hci_score * st.session_state.s_enroll) / 10
                ]
                
                avg_values = [9.5, 8.5, 7.0, 5.5, 6.0] # Synthetic "Gold Standard"
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=input_values, theta=categories, fill='toself', name=f'{selected_country} (Simulated)'))
                fig_radar.add_trace(go.Scatterpolar(r=avg_values, theta=categories, fill='toself', name='G20 Average', line=dict(dash='dot')))
                
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True, height=350)
                st.plotly_chart(fig_radar, use_container_width=True)

            # 3. Scatter Plot (Global Context)
            with tab3:
                # Generating synthetic scatter data for visual context
                np.random.seed(42)
                x_vals = np.random.uniform(0.3, 0.9, 30)
                y_vals = (x_vals * 8) + np.random.normal(0, 1, 30)
                
                fig_scatter = px.scatter(x=x_vals, y=y_vals, labels={'x': 'Human Capital Index', 'y': 'GDP Growth (%)'},
                                         title="Global HCI vs Growth Correlation", template="plotly_white")
                
                # Add "You Are Here" dot
                fig_scatter.add_trace(go.Scatter(x=[st.session_state.hci_score], y=[predicted_growth], 
                                                 mode='markers+text', marker=dict(color='red', size=15, symbol='star'),
                                                 name='You', text=[selected_country], textposition="top center"))
                
                st.plotly_chart(fig_scatter, use_container_width=True)


            # --- D. DOWNLOAD REPORT ---
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
        
        if st.session_state.simulation_run:
            response = f"Analyzing {selected_country}... The Radar Chart shows a gap in Secondary Education compared to the G20 average. Closing this would improve the Innovation score."
        else:
            response = "Please run the simulation first so I can analyze the deep dive metrics."
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# --- 5. FOOTER ---
st.markdown("""
<div class="footer-container">
    <div class="footer-author">DEVELOPED BY ALI SHER KHAN TAREEN</div>
    <div class="footer-note">v2.1 Enterprise Edition | AI-Powered Economic Forecasting</div>
</div>
""", unsafe_allow_html=True)