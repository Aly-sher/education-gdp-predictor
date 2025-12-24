import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Global Growth AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
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
    
    /* Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 26px; font-weight: 700; color: #2e7d32; }
    .metric-label { font-size: 13px; font-weight: 600; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Chat Container */
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background-color: white;
        height: 500px;
        overflow-y: scroll;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. HELPER FUNCTIONS ---
@st.cache_data
def get_world_bank_data(country_code, indicator):
    # API Call to World Bank
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
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
st.sidebar.title("Global Growth AI")
st.sidebar.header("⚙️ Configuration")

# Country Selector
country_options = {"United States": "US", "China": "CN", "India": "IN", "Pakistan": "PK", "Germany": "DE", "Brazil": "BR"}
selected_country = st.sidebar.selectbox("Select Country", list(country_options.keys()), index=0)
country_code = country_options[selected_country]

st.sidebar.markdown("---")
st.sidebar.header("🎓 Education Simulation")
st.sidebar.info("Adjust indicators to forecast growth:")

# --- SLIDER LOGIC FIX ---
# 1. Primary Slider
primary_enroll = st.sidebar.slider("Primary School Enrollment (%)", 0, 100, 85)

# 2. Secondary Slider (Max value is constrained by Primary Enrollment)
secondary_enroll = st.sidebar.slider(
    "Secondary School Enrollment (%)", 
    0, 
    primary_enroll,  # MAX value is set to whatever Primary is
    min(60, primary_enroll) # Default value adjusts if primary drops below 60
)

# 3. Validation Message
if secondary_enroll == primary_enroll:
    st.sidebar.caption("⚠️ Secondary cannot exceed Primary.")

human_capital_index = st.sidebar.slider("Human Capital Index (0-1)", 0.0, 1.0, 0.6)

# --- 6. MAIN LAYOUT (Split Screen) ---

# Top Header
st.markdown(f"## 🌍 Economic Dashboard: **{selected_country}**")

# Create Two Columns: Left for Data (65%), Right for Chatbot (35%)
col_data, col_chat = st.columns([2, 1], gap="large")

# --- LEFT COLUMN: DATA & VISUALS ---
with col_data:
    with st.spinner('Analyzing economic data...'):
        df_gdp = get_world_bank_data(country_code, "NY.GDP.MKTP.KD.ZG")

    if not df_gdp.empty:
        latest_growth = df_gdp['value'].iloc[-1]
        
        # Simulation Logic
        simulated_boost = (primary_enroll * 0.015) + (secondary_enroll * 0.025) + (human_capital_index * 2.0)
        predicted_growth = latest_growth + (simulated_boost / 8)

        # --- ROW 1: METRIC CARDS ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Actual GDP Growth</div>
                <div class="metric-value">{latest_growth:.2f}%</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">AI Forecast</div>
                <div class="metric-value" style="color: #1976d2;">{predicted_growth:.2f}%</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Simulation Boost</div>
                <div class="metric-value">+{simulated_boost/8:.2f}%</div></div>""", unsafe_allow_html=True)

        # --- ROW 2: MAIN CHART (Area Plot) ---
        st.subheader("📈 GDP Growth Trends")
        fig_gdp = px.area(df_gdp, x='date', y='value', 
                          labels={'value': 'Growth Rate (%)', 'date': 'Year'},
                          color_discrete_sequence=['#4caf50'])
        
        # Add Forecast Line
        fig_gdp.add_hline(y=predicted_growth, line_dash="dot", line_color="#2196f3", 
                          annotation_text=f"Forecast: {predicted_growth:.1f}%")
        
        fig_gdp.update_layout(plot_bgcolor='white', margin=dict(t=10, b=10, l=10, r=10),
                              xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f5f5f5'))
        st.plotly_chart(fig_gdp, use_container_width=True)

        # --- ROW 3: NEW CHARTS (Bar & Scatter) ---
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("📊 Enrollment Rates")
            # Creating a small dataframe for the bar chart
            enroll_data = pd.DataFrame({
                "Level": ["Primary", "Secondary"],
                "Percentage": [primary_enroll, secondary_enroll],
                "Color": ["#ff9800", "#ff5722"]
            })
            fig_bar = px.bar(enroll_data, x="Level", y="Percentage", color="Level", 
                             text_auto=True, color_discrete_sequence=["#ff9800", "#ff5722"])
            fig_bar.update_layout(showlegend=False, plot_bgcolor='white', height=300)
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            st.subheader("🧩 Human Capital vs Growth")
            # Simulated data for visual appeal
            x_sim = np.linspace(0, 1, 20)
            y_sim = x_sim * 5 + np.random.normal(0, 0.5, 20) + latest_growth
            
            fig_scatter = px.scatter(x=x_sim, y=y_sim, labels={'x': 'Human Capital Index', 'y': 'Proj. Growth'},
                                     title="Correlation Model")
            # Highlight current user selection
            fig_scatter.add_trace(go.Scatter(x=[human_capital_index], y=[predicted_growth], 
                                             mode='markers', marker=dict(color='red', size=12), name='You'))
            fig_scatter.update_layout(plot_bgcolor='white', height=300)
            st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.error("⚠️ Data unavailable. API Limit might be reached.")

# --- RIGHT COLUMN: AI CHATBOT ---
with col_chat:
    st.markdown("### 🤖 AI Assistant")
    st.markdown("---")
    
    # Chat container styling
    with st.container():
        # Display history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about the forecast..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. AI Logic (Context Aware)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Context Variables for the AI
            context_growth = f"{latest_growth:.2f}%"
            context_pred = f"{predicted_growth:.2f}%"
            
            if "growth" in prompt.lower() or "gdp" in prompt.lower():
                response_text = f"Current GDP growth for {selected_country} is {context_growth}. With your simulation settings, we project it could hit {context_pred}."
            elif "education" in prompt.lower() or "school" in prompt.lower():
                response_text = f"You've set Primary Enrollment to {primary_enroll}% and Secondary to {secondary_enroll}%. This gap suggests {primary_enroll - secondary_enroll}% drop-out rate."
            elif "secondary" in prompt.lower():
                response_text = "Note that Secondary enrollment is constrained; it cannot mathematically exceed Primary enrollment in our model."
            else:
                response_text = "I am tracking the economic indicators on your dashboard. Try asking about 'growth forecast' or 'enrollment impact'."

            # Typing Animation
            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})