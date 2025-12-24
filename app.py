import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Global Growth AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. SESSION STATE (For Chatbot History) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f8f9fa; }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 24px; font-weight: 600; color: #2e7d32; }
    .metric-label { font-size: 14px; color: #6c757d; }
    </style>
""", unsafe_allow_html=True)

# --- 4. HELPER FUNCTIONS ---
@st.cache_data
def get_world_bank_data(country_code, indicator):
    # GDP Growth: NY.GDP.MKTP.KD.ZG
    # Primary Enrollment: SE.PRM.ENRR
    # Secondary Enrollment: SE.SEC.ENRR
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&date=2000:2023"
    response = requests.get(url)
    data = response.json()
    
    if len(data) > 1:
        df = pd.DataFrame(data[1])
        df['value'] = pd.to_numeric(df['value'])
        df['date'] = pd.to_numeric(df['date'])
        df = df.sort_values('date')
        return df
    return pd.DataFrame()

# --- 5. SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Configuration")

# Country Selector
country_options = {"United States": "US", "China": "CN", "India": "IN", "Pakistan": "PK", "Germany": "DE"}
selected_country = st.sidebar.selectbox("Select Country", list(country_options.keys()), index=0)
country_code = country_options[selected_country]

st.sidebar.markdown("---")
st.sidebar.header("🎓 Education Indicators")
st.sidebar.info("Adjust these to simulate impact on GDP:")

# The Sliders you asked for
primary_enroll = st.sidebar.slider("Primary School Enrollment (%)", 0, 100, 85)
secondary_enroll = st.sidebar.slider("Secondary School Enrollment (%)", 0, 100, 60)
human_capital_index = st.sidebar.slider("Human Capital Index (0-1)", 0.0, 1.0, 0.6)

# --- 6. MAIN DASHBOARD ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 🌍")
with col_title:
    st.markdown(f"# Global Growth AI\n### Analysis: **{selected_country}**")

# Tabs for Dashboard vs Chatbot
tab1, tab2 = st.tabs(["📊 Dashboard & Simulation", "🤖 AI Chatbot"])

# --- TAB 1: DASHBOARD ---
with tab1:
    # Fetch GDP Data
    df_gdp = get_world_bank_data(country_code, "NY.GDP.MKTP.KD.ZG")

    if not df_gdp.empty:
        latest_growth = df_gdp['value'].iloc[-1]
        
        # --- SIMULATION LOGIC ---
        # A simple formula to show how sliders affect the "Predicted" growth
        # (This replaces the AI model logic for the simulation part)
        simulated_boost = (primary_enroll * 0.02) + (secondary_enroll * 0.03) + (human_capital_index * 1.5)
        predicted_growth = latest_growth + (simulated_boost / 10) # Scaling it down to be realistic

        # Metrics Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Actual GDP Growth</div>
                <div class="metric-value">{latest_growth:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AI Predicted Growth</div>
                <div class="metric-value" style="color: #1976d2;">{predicted_growth:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Impact Factor</div>
                <div class="metric-value">+{simulated_boost/10:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        # Charts
        st.markdown("### 📈 Economic Trends")
        
        # Plotly Chart
        fig = px.area(df_gdp, x='date', y='value', title="Historical GDP Growth",
                      labels={'value': 'Growth (%)'}, color_discrete_sequence=['#2e7d32'])
        fig.add_hline(y=predicted_growth, line_dash="dot", annotation_text="Simulated Forecast", annotation_position="top left", line_color="blue")
        
        fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Could not fetch data.")

# --- TAB 2: AI CHATBOT ---
with tab2:
    st.subheader("💬 Chat with Global Growth Assistant")
    st.caption("Ask about economic trends, definitions, or the current data.")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask something about the economy..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. AI Response (Simulated Rule-Based Logic)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simple Logic for responses
            if "gdp" in prompt.lower():
                response_text = f"The current GDP growth for {selected_country} is {latest_growth:.2f}%. Based on your education sliders, we predict it could rise to {predicted_growth:.2f}%."
            elif "enrollment" in prompt.lower():
                response_text = f"You have set Primary Enrollment to {primary_enroll}% and Secondary to {secondary_enroll}%. Higher enrollment typically correlates with long-term economic stability."
            elif "hello" in prompt.lower() or "hi" in prompt.lower():
                response_text = "Hello! I am your Economic Assistant. Ask me about GDP, enrollment trends, or forecasting."
            else:
                response_text = "That's an interesting economic question. I'm currently analyzing the correlation between education indices and market volatility for that topic."

            # Typing effect
            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})