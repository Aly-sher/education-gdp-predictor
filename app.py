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
st.sidebar.caption("v2.2 User-Friendly Edition")
st.sidebar.markdown("---")

# Expanded Country Selection
st.sidebar.header("📍 Economy")
selected_country = st.sidebar.selectbox("Select Market", list(utils.COUNTRY_MAP.keys()))
country_code = utils.COUNTRY_MAP[selected_country]

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Policy Simulation")

# --- THE FORM (Now with Layman Explanations) ---
with st.sidebar.form("policy_form"):
    st.info("👇 Adjust these levers to see how they change the future.")
    
    # 1. Primary Enrollment (Added Help Tooltip)
    p_enroll = st.slider(
        "Primary School Enrollment (%)", 
        50, 100, 85,
        help="What % of children are attending elementary school? (Global Avg is ~89%)"
    )
    
    # 2. Secondary Enrollment (Added Context)
    s_enroll = st.slider(
        "Secondary School Enrollment (%)", 
        0, 100, 60,
        help="What % of children make it to High School? This drives the skilled workforce."
    )
    
    # 3. HCI (Renamed for clarity + Contextual Help)
    hci_score = st.slider(
        "Workforce Quality (HCI Index)", 
        0.0, 1.0, 0.55,
        help="Human Capital Index: Measures the health & skills of the next generation. 0.5 = Low Productivity, 0.88 = World Class (Singapore)."
    )
    
    run_simulation = st.form_submit_button("🚀 Run Simulation")

if run_simulation:
    st.session_state.simulation_run = True
    st.session_state.p_enroll = p_enroll
    st.session_state.s_enroll = s_enroll
    st.session_state.hci_score = hci_score

# --- 3. MAIN DASHBOARD ---
st.markdown(f"## 🌍 Economic Intelligence: **{selected_country}**")

# Fetch Data
with st.spinner(f"Retrieving data for {selected_country}..."):
    df_gdp = utils.get_world_bank_data(country_code, "NY.GDP.MKTP.KD.ZG")

col_main, col_chat = st.columns([2, 1], gap="large")

with col_main:
    if not df_gdp.empty:
        latest_growth = df_gdp['value'].iloc[-1]
        
        # --- A. TOP KPI CARDS (Simplified) ---
        if st.session_state.simulation_run:
            # AI Calculation
            ai_impact = utils.ai_engine.predict_impact(
                st.session_state.p_enroll, 
                st.session_state.s_enroll, 
                st.session_state.hci_score
            )
            predicted_growth = latest_growth + (ai_impact / 10)

            k1, k2, k3 = st.columns(3)
            k1.markdown(f"""<div class="metric-card"><div class="metric-label">Current Growth</div>
                <div class="metric-value">{latest_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            k2.markdown(f"""<div class="metric-card"><div class="metric-label">AI Prediction</div>
                <div class="metric-value" style="color: #3182ce;">{predicted_growth:.2f}%</div></div>""", unsafe_allow_html=True)
            
            delta_color = "#38a169" if ai_impact > 0 else "#e53e3e"
            k3.markdown(f"""<div class="metric-card"><div class="metric-label">Policy Effect</div>
                <div class="metric-value" style="color: {delta_color};">{ai_impact/10:+.2f}%</div></div>""", unsafe_allow_html=True)
        else:
            predicted_growth = latest_growth 
            st.info("👈 **Start Here:** Adjust the sliders in the sidebar and click 'Run Simulation'.")

        # --- B. FORECAST TRAJECTORY ---
        st.subheader("📈 Future Growth Projection")
        st.caption("The dotted line shows where the economy will go based on your policy changes.")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_gdp['date'], y=df_gdp['value'], mode='lines', 
                                 name='Past Performance', line=dict(color='#cbd5e0', width=2)))
        
        if st.session_state.simulation_run:
            last_year = df_gdp['date'].iloc[-1]
            years = [last_year, last_year+1, last_year+2, last_year+3]
            values = [latest_growth, predicted_growth, predicted_growth * 1.05, predicted_growth * 1.08]
            
            fig.add_trace(go.Scatter(x=years, y=values, mode='lines+markers', name='AI Forecast', 
                                     line=dict(color='#3182ce', width=4, shape='spline')))

        fig.update_layout(plot_bgcolor='white', height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f7fafc'))
        st.plotly_chart(fig, use_container_width=True)

        # --- C. DEEP DIVE ANALYTICS (With "How to read" notes) ---
        if st.session_state.simulation_run:
            st.markdown("### 🔍 Analysis Breakdown")
            
            tab1, tab2, tab3 = st.tabs(["🎓 Dropout Analysis", "🕸️ Economy Health", "🌍 Global Comparison"])
            
            # 1. Education Funnel
            with tab1:
                st.caption("💡 **What this means:** This shows how many students are lost between Primary school and joining the Workforce.")
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    dropout_rate = st.session_state.p_enroll - st.session_state.s_enroll
                    funnel_data = pd.DataFrame({
                        "Stage": ["Primary School", "High School", "Skilled Workforce"],
                        "Percentage": [st.session_state.p_enroll, st.session_state.s_enroll, st.session_state.s_enroll * 0.8],
                        "Color": ["#4299e1", "#2b6cb0", "#2c5282"]
                    })
                    fig_funnel = px.bar(funnel_data, x="Percentage", y="Stage", orientation='h', 
                                        text_auto=True, 
                                        color="Stage", color_discrete_sequence=["#4299e1", "#2b6cb0", "#2c5282"])
                    fig_funnel.update_layout(showlegend=False, plot_bgcolor='white')
                    st.plotly_chart(fig_funnel, use_container_width=True)
                
                with col_c2:
                    if dropout_rate > 15:
                        st.error(f"⚠️ High Dropout: {dropout_rate}%")
                        st.markdown("**Insight:** Too many children are quitting before High School. This hurts the economy.")
                    else:
                        st.success(f"✅ Good Retention: {dropout_rate}%")
                        st.markdown("**Insight:** Most children are staying in school. Good job.")

            # 2. Radar Chart
            with tab2:
                st.caption("💡 **How to read this:** The Blue Shape is your economy. The Dotted Line is the G20 Average. You want your Blue Shape to be bigger than the Dotted Line.")
                
                categories = ['Primary Ed', 'Secondary Ed', 'Workforce Quality', 'GDP Growth', 'Innovation']
                input_values = [
                    st.session_state.p_enroll / 10, 
                    st.session_state.s_enroll / 10, 
                    st.session_state.hci_score * 10, 
                    max(0, predicted_growth + 5),
                    (st.session_state.hci_score * st.session_state.s_enroll) / 10
                ]
                avg_values = [9.5, 8.5, 7.0, 5.5, 6.0] 
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=input_values, theta=categories, fill='toself', name=f'{selected_country}'))
                fig_radar.add_trace(go.Scatterpolar(r=avg_values, theta=categories, fill='toself', name='G20 Average', line=dict(dash='dot')))
                
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True, height=350)
                st.plotly_chart(fig_radar, use_container_width=True)

            # 3. Scatter Plot
            with tab3:
                st.caption("💡 **What this means:** This compares your 'Workforce Quality' (X-axis) against 'GDP Growth' (Y-axis). Being higher up is better.")
                np.random.seed(42)
                x_vals = np.random.uniform(0.3, 0.9, 30)
                y_vals = (x_vals * 8) + np.random.normal(0, 1, 30)
                
                fig_scatter = px.scatter(x=x_vals, y=y_vals, labels={'x': 'Workforce Quality (HCI)', 'y': 'GDP Growth (%)'},
                                         title="Global Comparison", template="plotly_white")
                
                fig_scatter.add_trace(go.Scatter(x=[st.session_state.hci_score], y=[predicted_growth], 
                                                 mode='markers+text', marker=dict(color='red', size=15, symbol='star'),
                                                 name='You', text=['YOU'], textposition="top center"))
                
                st.plotly_chart(fig_scatter, use_container_width=True)


            # --- D. DOWNLOAD REPORT ---
            st.markdown("### 📄 Export Results")
            report_pdf = utils.create_pdf_report(
                selected_country, latest_growth, predicted_growth,
                st.session_state.p_enroll, st.session_state.s_enroll, st.session_state.hci_score
            )
            
            st.download_button(
                label="📥 Download Strategy Report (PDF)",
                data=report_pdf,
                file_name=f"{selected_country}_Strategy_Report.pdf",
                mime="application/pdf",
                help="Click to generate a professional PDF summary of this simulation."
            )

    else:
        st.error("Data unavailable for this region.")

# --- 4. CHATBOT SECTION ---
with col_chat:
    st.markdown("### 🤖 Assistant")
    st.markdown("---")
    
    with st.container():
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
    if prompt := st.chat_input("Ask about the simulation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        if st.session_state.simulation_run:
            response = f"I've analyzed the data for {selected_country}. Your 'Workforce Quality' score is {st.session_state.hci_score}. Increasing this score usually has the biggest impact on long-term GDP."
        else:
            response = "I'm ready! Please adjust the sliders on the left and click 'Run Simulation' first."
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# --- 5. FOOTER ---
st.markdown("""
<div class="footer-container">
    <div class="footer-author">DEVELOPED BY ALI SHER KHAN TAREEN</div>
    <div class="footer-note">Simple AI Model • Educational Purpose Only</div>
</div>
""", unsafe_allow_html=True)