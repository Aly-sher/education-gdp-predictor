import streamlit as st
import pandas as pd
import numpy as np
import wbgapi as wb
import plotly.graph_objects as go
import groq
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------------------
# 1. CONFIGURATION & THEME
# -------------------------------------------
# UPDATED TITLE HERE
st.set_page_config(page_title="Global Growth AI", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #f8f9fa; color: #212529; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #2c3e50; color: white; }
    [data-testid="stSidebar"] * { color: #ecf0f1 !important; }
    
    /* Metrics */
    div[data-testid="metric-container"] { 
        background-color: white; 
        border: 1px solid #e9ecef; 
        padding: 15px; 
        border-radius: 8px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
    }
    
    /* Strategy Cards */
    .advice-card { 
        background-color: white; 
        border-left: 5px solid #ddd; 
        padding: 15px; 
        border-radius: 6px; 
        margin-bottom: 10px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
    }
    
    /* Chat Container */
    .chat-container { 
        border-top: 2px solid #e9ecef; 
        padding-top: 20px; 
        margin-top: 20px; 
    }
    
    /* Footer Signature */
    .footer { 
        position: fixed; 
        left: 0; 
        bottom: 0; 
        width: 100%; 
        background-color: transparent; 
        color: #95a5a6; 
        text-align: right; 
        padding-right: 20px; 
        padding-bottom: 10px; 
        font-size: 14px; 
        font-weight: 600; 
        z-index: 100; 
        pointer-events: none; 
    }
</style>
<div class="footer">Developed by ALI SHER KHAN TAREEN</div>
""", unsafe_allow_html=True)

def format_billions(value):
    if value >= 1e12: return f"${value/1e12:.2f}T"
    elif value >= 1e9: return f"${value/1e9:.2f}B"
    else: return f"${value/1e6:.2f}M"

# -------------------------------------------
# 2. LOGIC: AI POLICY CONSULTANT
# -------------------------------------------
def generate_policy_advice(country, prim, sec, exp, pred_lit, current_gdp, pred_gdp, hist_exp):
    advice_list = []
    
    # 1. FISCAL MOMENTUM
    delta_spend = exp - hist_exp
    if delta_spend < -0.5:
        advice_list.append({
            "color": "#e74c3c", 
            "icon": "✂️", 
            "title": "Austerity Risk", 
            "insight": f"Spending cut by **{abs(delta_spend):.1f}%** vs history.", 
            "action": "Restore funding."
        })
    
    # 2. STRUCTURAL EFFICIENCY
    if exp > 5.0 and pred_lit < 80.0:
        advice_list.append({
            "color": "#f39c12", 
            "icon": "📉", 
            "title": "Inefficiency", 
            "insight": f"High spend ({exp}%) but low literacy.", 
            "action": "Audit payrolls."
        })

    # 3. PIPELINE
    if (prim - sec) > 15:
        advice_list.append({
            "color": "#e74c3c", 
            "icon": "🛑", 
            "title": "Dropout Crisis", 
            "insight": f"**{prim-sec:.1f}%** drop-off rate.", 
            "action": "Subsidize secondary."
        })

    # 4. ROI
    growth = ((pred_gdp - current_gdp) / current_gdp) * 100
    advice_list.append({
        "color": "#3498db", 
        "icon": "📈", 
        "title": "ROI Analysis", 
        "insight": f"Predicted Growth: **{growth:.1f}%**.", 
        "action": "Maintain stability."
    })

    return advice_list

# -------------------------------------------
# 3. DATA ENGINE
# -------------------------------------------
@st.cache_data
def fetch_world_bank_data():
    indicators = {
        'NY.GDP.MKTP.CD': 'Total_GDP', 
        'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
        'SE.ADT.LITR.ZS': 'Literacy_rate', 
        'SE.PRM.ENRR': 'Primary_enrollment',
        'SE.SEC.ENRR': 'Secondary_enrollment', 
        'SE.XPD.TOTL.GD.ZS': 'Education_expenditure'
    }
    country_codes = ["USA", "PAK", "IND", "CHN", "GBR", "BRA", "CAN", "AUS", "DEU", "FRA"]
    try:
        raw_data = wb.data.fetch(indicators.keys(), economy=country_codes, time=range(2000, 2025))
        raw_df = pd.DataFrame(list(raw_data))
        if raw_df.empty: return pd.DataFrame()
        
        df = raw_df.pivot_table(index=['economy', 'time'], columns='series', values='value')
        df.reset_index(inplace=True)
        df.rename(columns=indicators, inplace=True)
        df.rename(columns={'economy': 'Country Code', 'time': 'Year'}, inplace=True)
        df['Year'] = df['Year'].astype(str).str.replace('YR', '', regex=False).astype(int)
        
        code_map = {'USA':'United States','PAK':'Pakistan','IND':'India','CHN':'China','GBR':'United Kingdom','BRA':'Brazil','CAN':'Canada','AUS':'Australia','DEU':'Germany','FRA':'France'}
        df['Country Name'] = df['Country Code'].map(code_map)
        
        numeric_cols = list(indicators.values())
        for col in numeric_cols:
            if col not in df.columns: df[col] = 0
            df[col] = df.groupby('Country Code')[col].transform(lambda x: x.interpolate(limit_direction='both').ffill().bfill())
            df[col] = df[col].fillna(df[col].mean())
            df[col] = df[col].fillna(0)
            
        return df
    except: return pd.DataFrame()

# -------------------------------------------
# 4. AI MODEL
# -------------------------------------------
@st.cache_resource
def train_model(df):
    if df.empty: return None, None, None
    df = df.dropna(subset=['Country Name']).fillna(0)
    
    le = LabelEncoder()
    df['Country_Encoded'] = le.fit_transform(df['Country Name'].astype(str))
    
    X = df[['Country_Encoded', 'Year', 'Primary_enrollment', 'Secondary_enrollment', 'Education_expenditure']].values
    y = df[['Total_GDP', 'GDP_Per_Capita', 'Literacy_rate']].values
    
    scaler = StandardScaler()
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    
    try:
        model.fit(scaler.fit_transform(X), y)
        return scaler, le, model
    except: return None, None, None

with st.spinner("Initializing System..."):
    df = fetch_world_bank_data()
    scaler, le, model = train_model(df) if not df.empty else (None, None, None)

# -------------------------------------------
# 5. DASHBOARD UI
# -------------------------------------------
if model:
    # --- SIDEBAR ---
    st.sidebar.header("🎛️ AI Economic Simulator")
    country = st.sidebar.selectbox("Economy", list(le.classes_), index=list(le.classes_).index('Pakistan') if 'Pakistan' in le.classes_ else 0)
    country_data = df[df['Country Name'] == country].sort_values('Year')
    latest = country_data.iloc[-1]
    
    st.sidebar.subheader("Target")
    year = st.sidebar.slider("", 2025, 2030, 2026)
    
    st.sidebar.subheader("Policies")
    prim = st.sidebar.slider("Primary (%)", 50.0, 120.0, float(latest['Primary_enrollment']))
    sec = st.sidebar.slider("Secondary (%)", 20.0, float(prim), min(float(prim), float(latest['Secondary_enrollment'])))
    exp = st.sidebar.slider("Edu Spend (%)", 1.0, 10.0, float(latest['Education_expenditure']))

    if st.sidebar.button("Run Simulation", type="primary", use_container_width=True):
        c_code = le.transform([country])[0]
        # Fixed formatting
        inputs_raw = [[c_code, year, prim, sec, exp]]
        pred = model.predict(scaler.transform(inputs_raw))
        
        st.session_state['simulation_ran'] = True
        st.session_state['pred_results'] = pred[0]
        st.session_state['inputs'] = {'country': country, 'year': year, 'exp': exp}

    # --- DISPLAY RESULTS ---
    if st.session_state.get('simulation_ran'):
        pred_total, pred_pcap, pred_lit = st.session_state['pred_results']
        
        # UPDATED MAIN TITLE HERE
        st.title(f"🚀 Global Growth AI: {country}")
        
        m1, m2, m3 = st.columns(3)
        gdp_delta = ((pred_total - latest['Total_GDP']) / latest['Total_GDP']) * 100 if latest['Total_GDP'] > 0 else 0
        m1.metric("GDP Forecast", format_billions(pred_total), delta=f"{gdp_delta:.1f}%")
        m2.metric("Per Capita", f"${pred_pcap:,.0f}")
        m3.metric("Literacy", f"{pred_lit:.1f}%")
        
        st.markdown("---")
        
        col_chart, col_strategy = st.columns([1.8, 1], gap="medium")
        
        with col_chart:
            st.subheader("📈 Economic Trajectory")
            chart_mode = st.radio("Metric:", ["Total GDP", "Per Capita"], horizontal=True, label_visibility="collapsed")
            target, val = ('Total_GDP', pred_total) if "Total" in chart_mode else ('GDP_Per_Capita', pred_pcap)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=country_data['Year'], y=country_data[target], mode='lines', name='History', line=dict(color='#95a5a6', width=3)))
            fig.add_trace(go.Scatter(x=[country_data['Year'].max(), year], y=[country_data.loc[country_data['Year'].max()==country_data['Year'], target].values[0], val], mode='lines+markers', name='Forecast', line=dict(color='#2980b9', width=3, dash='dashdot')))
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='white', plot_bgcolor='white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_strategy:
            st.subheader("🤖 Strategic Analysis")
            recs = generate_policy_advice(country, prim, sec, exp, pred_lit, latest['Total_GDP'], pred_total, latest['Education_expenditure'])
            for r in recs:
                st.markdown(f"<div class='advice-card' style='border-left-color:{r['color']}'><b>{r['icon']} {r['title']}</b><br><small>{r['insight']}</small><br><span style='color:{r['color']}'><b>{r['action']}</b></span></div>", unsafe_allow_html=True)

        # Chatbot
        st.markdown("<div class='chat-container'></div>", unsafe_allow_html=True)
        st.subheader("💬 AI Analyst Chat")
        
        if "messages" not in st.session_state: st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])

        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
            client = groq.Groq(api_key=api_key)
            
            if prompt := st.chat_input(f"Ask about the {year} forecast..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                
                context = f"Country: {country}. Year: {year}. GDP: {format_billions(pred_total)}. Spend: {exp}%. Literacy: {pred_lit}%. Q: {prompt}"
                
                with st.chat_message("assistant"):
                    try:
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "You are a senior economist."}, 
                                {"role": "user", "content": context}
                            ], 
                            temperature=0.7
                        )
                        response = completion.choices[0].message.content
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")
        else: st.info("💡 Add Groq API Key to secrets.")

else: st.error("Data Connection Failed")