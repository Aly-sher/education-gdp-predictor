import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# --- 1. PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="Global Growth AI",
    page_icon="🌍",
    layout="wide",  # "wide" looks professional on desktop, scales down on mobile
    initial_sidebar_state="collapsed" # Collapsed on mobile to save space
)

# --- 2. CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Card Styling for Metrics */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #2e7d32;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }

    /* Clean up the top header padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def get_world_bank_data(country_code):
    # Fetching real GDP growth data from World Bank API
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/NY.GDP.MKTP.KD.ZG?format=json&date=2000:2023"
    response = requests.get(url)
    data = response.json()
    
    if len(data) > 1:
        df = pd.DataFrame(data[1])
        df['value'] = pd.to_numeric(df['value'])
        df['date'] = pd.to_numeric(df['date'])
        df = df.sort_values('date')
        return df
    return pd.DataFrame()

# --- 4. SIDEBAR (CONTROLS) ---
st.sidebar.header("⚙️ Configuration")
country_options = {
    "United States": "US",
    "China": "CN",
    "India": "IN",
    "Pakistan": "PK",
    "Germany": "DE",
    "United Kingdom": "GB"
}
selected_country = st.sidebar.selectbox("Select Country", list(country_options.keys()), index=0)
country_code = country_options[selected_country]

# --- 5. MAIN DASHBOARD UI ---

# Header Section
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 🌍")
with col_title:
    st.markdown(f"# Global Growth AI\n### Economic Forecast & Analysis: **{selected_country}**")

st.markdown("---")

# Fetch Data
with st.spinner('Fetching live World Bank data...'):
    df = get_world_bank_data(country_code)

if not df.empty:
    # --- METRICS SECTION (CARDS) ---
    latest_year = df['date'].iloc[-1]
    latest_growth = df['value'].iloc[-1]
    avg_growth = df['value'].mean()
    
    # Custom HTML Cards for responsive layout
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latest GDP Growth ({latest_year})</div>
            <div class="metric-value">{latest_growth:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Growth (2000-2023)</div>
            <div class="metric-value">{avg_growth:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        status_color = "#2e7d32" if latest_growth > 0 else "#c62828"
        status_text = "Expansion" if latest_growth > 0 else "Recession"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Economic Status</div>
            <div class="metric-value" style="color: {status_color};">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 GDP Growth Trends")
    
    # --- CHARTS SECTION ---
    # Tab layout looks very professional on mobile
    tab1, tab2 = st.tabs(["Historical Trend", "Forecast Analysis"])
    
    with tab1:
        # Plotly Area Chart
        fig = px.area(df, x='date', y='value', 
                      title=f"GDP Growth Rate Over Time ({selected_country})",
                      labels={'value': 'Growth Rate (%)', 'date': 'Year'},
                      color_discrete_sequence=['#0d47a1'])
        
        # Professional Chart Styling
        fig.update_layout(
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        # CRITICAL FIX FOR SAFARI: use_container_width=True
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("⚠️ This forecast uses a simple moving average for demonstration.")
        
        # Simple Forecast Logic
        df['SMA_3'] = df['value'].rolling(window=3).mean()
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines', name='Actual', line=dict(color='#cfd8dc', width=2)))
        fig_forecast.add_trace(go.Scatter(x=df['date'], y=df['SMA_3'], mode='lines', name='3-Year Trend', line=dict(color='#2e7d32', width=3)))
        
        fig_forecast.update_layout(
            title="Trend Forecast (Moving Average)",
            plot_bgcolor='white',
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

else:
    st.error("Could not load data. The World Bank API might be busy.")

# --- 6. FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #90a4ae; font-size: 12px;">
    Global Growth AI © 2025 | Powered by World Bank Data | Developed by Ali Sher Khan
</div>
""", unsafe_allow_html=True)