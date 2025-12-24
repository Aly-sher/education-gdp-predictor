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
    
    /* Metric Card Styling */
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
    
    /* AI Recommendation Card Styling */
    .rec-card {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        border-left: 5px solid #2196f3;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    }
    .rec-title { font-weight: 700; color: #1565c0; margin-bottom: 5px; font-size: 16px; }
    .rec-body { color: #455a64; font-size: 14px; line-height: 1.5; }
    
    /* Footer Styling */
    .footer-container {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        background-color: #f1f3f4;
        border-radius: 10px;
        color: #555;
    }
    .footer-author {
        font-weight: bold;
        font-size: 16px;
        color: #333;
        margin-bottom: 5px;
    }
    .footer-note {
        font-size: 12px;
        font-style: italic;
        color: #777;
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

# Check/Note about 6 countries
st.sidebar.caption(f"⚠️ Note: Data analysis is currently limited to these {len(country_options)} countries.")

st.sidebar.markdown("---")
st.sidebar.header("🎓 Education Simulation")
st.sidebar.info("Adjust indicators to forecast growth:")

# --- SLIDER LOGIC ---
primary_enroll = st.sidebar.slider("Primary School Enrollment (%)", 0, 100, 85)

secondary_enroll = st.sidebar.slider(
    "Secondary School Enrollment (%)", 
    0, 
    primary_enroll,  # MAX value constrained by Primary
    min(60, primary_enroll)
)

if secondary_enroll == primary_enroll:
    st.sidebar.caption("⚠️ Secondary capped at Primary level.")

human_capital_index = st.sidebar.slider("Human Capital Index (0-1)", 0.0, 1.0, 0.6)

# --- 6. MAIN LAYOUT ---

st.markdown(f"## 🌍 Economic Dashboard: **{selected_country}**")
col_data, col_chat = st.columns([2, 1], gap="large")

# --- LEFT COLUMN: DATA & VISUALS ---
with col_data:
    with st.spinner('Crunching numbers...'):
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
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Policy Impact</div>
                <div class="metric-value">+{simulated_boost/8:.2f}%</div></div>""", unsafe_allow_html=True)

        # --- ROW 2: MAIN CHART ---
        st.subheader("📈 GDP Growth Trajectory")
        fig_gdp = px.area(df_gdp, x='date', y='value', 
                          labels={'value': 'Growth Rate (%)', 'date': 'Year'},
                          color_discrete_sequence=['#4caf50'])
        fig_gdp.add_hline(y=predicted_growth, line_dash="dot", line_color="#2196f3", 
                          annotation_text=f"AI Target: {predicted_growth:.1f}%")
        fig_gdp.update_layout(plot_bgcolor='white', margin=dict(t=10, b=10, l=10, r=10),
                              xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f5f5f5'))
        st.plotly_chart(fig_gdp, use_container_width=True)

        # --- ROW 3: VISUALS ---
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("🎓 Enrollment Funnel")
            enroll_data = pd.DataFrame({
                "Level": ["Primary", "Secondary"],
                "Percentage": [primary_enroll, secondary_enroll]
            })
            fig_bar = px.bar(enroll_data, x="Level", y="Percentage", color="Level", text_auto=True, 
                             color_discrete_sequence=["#ff9800", "#ff5722"])
            fig_bar.update_layout(showlegend=False, plot_bgcolor='white', height=250, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            st.caption("🧩 Human Capital Impact")
            x_sim = np.linspace(0, 1, 20)
            y_sim = x_sim * 5 + np.random.normal(0, 0.5, 20) + latest_growth
            fig_scatter = px.scatter(x=x_sim, y=y_sim, labels={'x': 'HCI', 'y': 'Growth'})
            fig_scatter.add_trace(go.Scatter(x=[human_capital_index], y=[predicted_growth], 
                                             mode='markers', marker=dict(color='red', size=14), name='You'))
            fig_scatter.update_layout(plot_bgcolor='white', height=250, margin=dict(t=0,b=0,l=0,r=0), showlegend=False)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # --- ROW 4: AI STRATEGIC RECOMMENDATIONS ---
        st.markdown("### 🧠 AI Strategic Briefing")
        st.markdown("---")
        
        rec_col1, rec_col2 = st.columns(2)
        
        recs = []
        
        # Rec 1: Dropout Crisis
        if secondary_enroll < (primary_enroll - 20):
            recs.append({
                "title": "🚨 Plug the Talent Leak",
                "body": f"You are losing **{primary_enroll - secondary_enroll}%** of students before high school. Immediate vocational incentives are required to bridge this gap."
            })
        else:
            recs.append({
                "title": "✅ Strong Retention Pipeline",
                "body": "Your education pipeline is robust. Shift focus from 'Access' to 'Quality' of education to maximize returns."
            })

        # Rec 2: Human Capital
        if human_capital_index < 0.5:
            recs.append({
                "title": "⚡ Supercharge the Workforce",
                "body": "Human Capital is your bottleneck. Aggressive upskilling programs and healthcare investments will yield exponential GDP returns."
            })
        else:
            recs.append({
                "title": "🚀 Innovation Frontier",
                "body": "Your workforce is ready. Pivot policy towards R&D subsidies and Tech Infrastructure to unlock the next tier of growth."
            })

        # Rec 3: Growth Status
        if predicted_growth < 2.0:
            recs.append({
                "title": "📉 Stimulus Required",
                "body": "The economy is sluggish. Consider fiscal stimulus packages combined with the education reforms above to jumpstart momentum."
            })
        
        # Display Recommendations in Stylish Cards
        with rec_col1:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">{recs[0]['title']}</div>
                <div class="rec-body">{recs[0]['body']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with rec_col2:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">{recs[1]['title']}</div>
                <div class="rec-body">{recs[1]['body']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        if len(recs) > 2:
            st.markdown(f"""
            <div class="rec-card" style="border-left: 5px solid #ff5722;">
                <div class="rec-title" style="color: #bf360c;">{recs[2]['title']}</div>
                <div class="rec-body">{recs[2]['body']}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("⚠️ API connection failed. Please try again later.")

# --- RIGHT COLUMN: AI CHATBOT ---
with col_chat:
    st.markdown("### 🤖 Assistant")
    st.markdown("---")
    
    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if prompt := st.chat_input("Ask about the strategy..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if "recommend" in prompt.lower() or "improve" in prompt.lower():
                response_text = "Based on your sliders, I recommend focusing on 'Retention'. Creating vocational paths for students leaving primary school will instantly boost your Human Capital score."
            elif "growth" in prompt.lower():
                response_text = f"We are projecting a {predicted_growth:.2f}% growth rate. This is driven largely by your Human Capital Index setting of {human_capital_index}."
            else:
                response_text = "I am analyzing your policy settings. Try asking: 'How can I improve GDP?' or 'Why is growth low?'"

            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- 7. FOOTER ---
st.markdown("---")
st.markdown("""
<div class="footer-container">
    <div class="footer-author">DEVELOPED BY ALI SHER KHAN TAREEN</div>
    <div class="footer-note">Note: The prediction is based on the data that we have at the moment.</div>
</div>
""", unsafe_allow_html=True)