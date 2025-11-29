import streamlit as st
import pandas as pd
import numpy as np
import wbgapi as wb
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------------------
# 1. CONFIGURATION & PROFESSIONAL THEME
# -------------------------------------------
st.set_page_config(page_title="Global Growth Outlook", page_icon="🌐", layout="wide")

# Professional Fintech Theme CSS + YOUR SIGNATURE
st.markdown("""
<style>
    /* Main Area Aesthetic */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Sidebar Aesthetic */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #ecf0f1 !important;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e9ecef;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* AI Advice Cards */
    .advice-card {
        background-color: white;
        border-left: 6px solid #ddd;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .advice-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .advice-title { margin: 0; font-weight: 600; color: #34495e; }
    .advice-insight { color: #666; font-size: 0.95rem; margin: 8px 0; }
    .advice-action { font-weight: 700; font-size: 0.9rem; }
    
    /* Chart Container */
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 100%;
    }
    
    /* SIGNATURE FOOTER */
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
<div class="footer">
    Developed by ALI SHER KHAN TAREEN
</div>
""", unsafe_allow_html=True)

def format_billions(value):
    if value >= 1e12: return f"${value/1e12:.2f} Trillion"
    elif value >= 1e9: return f"${value/1e9:.2f} Billion"
    else: return f"${value/1e6:.2f} Million"

# -------------------------------------------
# 2. ADVANCED LOGIC: AI STRATEGIC CONSULTANT
# -------------------------------------------
def generate_policy_advice(country, prim, sec, exp, pred_lit, current_gdp, pred_gdp, hist_exp):
    advice_list = []
    
    # 1. FISCAL MOMENTUM CHECK
    delta_spend = exp - hist_exp
    if delta_spend < -0.5:
        advice_list.append({
            "type": "critical", "color": "#e74c3c", "icon": "✂️",
            "title": "Austerity Risk Detected",
            "insight": f"You have cut education spending by **{abs(delta_spend):.1f}%** compared to historical levels. This often leads to immediate teacher strikes and quality degradation.",
            "action": "Recommendation: Restore funding to at least historical levels to maintain stability."
        })
    elif delta_spend > 1.5:
        advice_list.append({
            "type": "info", "color": "#3498db", "icon": "🚀",
            "title": "Aggressive Fiscal Expansion",
            "insight": f"You are increasing the budget by **{delta_spend:.1f}%**. Ensure your Ministry of Education has the absorptive capacity to spend this efficiently without corruption.",
            "action": "Strategy: Earmark 30% of new funds specifically for digital infrastructure."
        })

    # 2. STRUCTURAL EFFICIENCY CHECK
    if exp > 5.0 and pred_lit < 80.0:
        advice_list.append({
            "type": "warning", "color": "#f39c12", "icon": "📉",
            "title": "Structural Inefficiency",
            "insight": f"You are spending a First-World budget (**{exp}%**) but getting Third-World results (Literacy **{pred_lit:.1f}%**). The money is not reaching the classroom.",
            "action": "Audit: Stop increasing budget. Launch a forensic audit of teacher 'ghost worker' payrolls."
        })

    # 3. THE "MIDDLE INCOME TRAP"
    drop_off = prim - sec
    if drop_off > 15:
        advice_list.append({
            "type": "critical", "color": "#e74c3c", "icon": "🛑",
            "title": "The 'Middle Income' Trap",
            "insight": f"Your economy cannot modernize because **{drop_off:.1f}%** of students quit before High School. You will be stuck with low-wage labor.",
            "action": "Policy: Make Secondary Education compulsory and free by law."
        })
    elif sec > 90 and exp < 3.0:
        advice_list.append({
            "type": "warning", "color": "#f39c12", "icon": "🏗️",
            "title": "Infrastructure Strain",
            "insight": "Your enrollment is high, but funding is too low. This implies overcrowded classrooms (60+ students) and poor quality.",
            "action": "Focus: Build physical schools immediately to reduce class sizes."
        })

    # 4. ECONOMIC ROI CHECK
    gdp_growth = ((pred_gdp - current_gdp) / current_gdp) * 100
    if gdp_growth < 5.0 and exp > 4.0:
        advice_list.append({
            "type": "info", "color": "#9b59b6", "icon": "🐢",
            "title": "Sluggish ROI",
            "insight": "Despite healthy spending, GDP growth is slow. Education takes 10+ years to impact GDP. Do not cut funding due to lack of short-term results.",
            "action": "Patience: Maintain policy consistency for at least one decade."
        })

    if len(advice_list) == 0:
        advice_list.append({
            "type": "success", "color": "#27ae60", "icon": "✨",
            "title": "Optimized Policy Mix",
            "insight": "Your inputs strike a perfect balance between fiscal responsibility and human capital development.",
            "action": "Next Step: Focus on gender parity and STEM curriculum updates."
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
        
        code_map = {
            'USA': 'United States', 'PAK': 'Pakistan', 'IND': 'India', 'CHN': 'China',
            'GBR': 'United Kingdom', 'BRA': 'Brazil', 'CAN': 'Canada', 'AUS': 'Australia',
            'DEU': 'Germany', 'FRA': 'France'
        }
        df['Country Name'] = df['Country Code'].map(code_map)
        
        for col in list(indicators.values()):
            if col in df.columns:
                df[col] = df.groupby('Country Code')[col].transform(lambda x: x.interpolate(limit_direction='both'))
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = 0
        return df
    except Exception: return pd.DataFrame()

# -------------------------------------------
# 4. AI MODEL
# -------------------------------------------
@st.cache_resource
def train_model(df):
    if df.empty: return None, None, None
    df = df.dropna(subset=['Country Name'])
    
    le = LabelEncoder()
    df['Country_Encoded'] = le.fit_transform(df['Country Name'].astype(str))
    
    X = df[['Country_Encoded', 'Year', 'Primary_enrollment', 'Secondary_enrollment', 'Education_expenditure']].values
    y = df[['Total_GDP', 'GDP_Per_Capita', 'Literacy_rate']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_scaled, y)
    return scaler, le, model

with st.spinner("Initializing Global Economic Data Engine..."):
    df = fetch_world_bank_data()
    if not df.empty:
        scaler, le, model = train_model(df)
        success = (model is not None)
    else:
        success = False

# -------------------------------------------
# 5. PROFESSIONAL DASHBOARD UI
# -------------------------------------------
if success:
    st.sidebar.header("🎛️ Economic Simulator")
    
    country_list = list(le.classes_)
    default_ix = country_list.index('Pakistan') if 'Pakistan' in country_list else 0
    country = st.sidebar.selectbox("Select Economy", country_list, index=default_ix)

    country_mask = df['Country Name'] == country
    country_data = df[country_mask].sort_values('Year')
    latest = country_data.iloc[-1]
    
    st.sidebar.subheader("Projection Target")
    year = st.sidebar.slider("", 2025, 2030, 2026, label_visibility="collapsed")
    
    st.sidebar.subheader("🏛️ Policy Levers")
    prim = st.sidebar.slider("Primary Enrollment (%)", 50.0, 120.0, float(latest['Primary_enrollment']))
    sec_default = min(float(prim), float(latest['Secondary_enrollment']))
    sec = st.sidebar.slider("Secondary Enrollment (%)", 20.0, float(prim), sec_default)
    exp = st.sidebar.slider("Govt Edu Spend (% GDP)", 1.0, 10.0, float(latest['Education_expenditure']))

    run_simulation = st.sidebar.button("Run Forecast Model", type="primary", use_container_width=True)

    st.title(f"🌐 Global Growth Outlook: {country}")
    st.markdown(f"Economic forecast scenario for year **{year}**.")

    if run_simulation:
        c_code = le.transform([country])[0]
        inputs = scaler.transform([[c_code, year, prim, sec, exp]])
        pred = model.predict(inputs)
        pred_total, pred_pcap, pred_lit = pred[0]
        
        # HERO METRICS
        gdp_growth = ((pred_total - latest['Total_GDP']) / latest['Total_GDP']) * 100
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Forecasted Total GDP", format_billions(pred_total), delta=f"{gdp_growth:.1f}% Growth")
        m2.metric("Forecasted Per Capita", f"${pred_pcap:,.0f}", delta="Standard of Living")
        m3.metric("Forecasted Literacy", f"{pred_lit:.1f}%", delta="Human Capital Index")

        st.markdown("---")

        col_chart, col_advice = st.columns([3, 2], gap="medium")

        # CHART
        with col_chart:
            st.subheader("📈 Economic Trajectory")
            chart_mode = st.radio("View Metric:", ["Total GDP (National Power)", "GDP Per Capita (Wealth)"], horizontal=True, label_visibility="collapsed")
            
            if "Total" in chart_mode:
                target_col, pred_val, line_color = 'Total_GDP', pred_total, '#2980b9'
            else:
                target_col, pred_val, line_color = 'GDP_Per_Capita', pred_pcap, '#8e44ad'

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=country_data['Year'], y=country_data[target_col], mode='lines', name='Historical', line=dict(color='#95a5a6', width=3)))
            
            last_year = country_data['Year'].max()
            last_val = country_data.loc[country_data['Year'] == last_year, target_col].values[0]
            
            fig.add_trace(go.Scatter(x=[last_year, year], y=[last_val, pred_val], mode='lines+markers', name='Forecast', line=dict(color=line_color, width=3, dash='dashdot'), marker=dict(size=8)))

            fig.update_layout(
                height=450,
                paper_bgcolor='rgba(255,255,255,1)',
                plot_bgcolor='rgba(255,255,255,1)',
                font=dict(color='#2c3e50'),
                xaxis=dict(showgrid=False, title="Year"),
                yaxis=dict(showgrid=True, gridcolor='#ecf0f1', title=chart_mode.split(" (")[0]),
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ADVICE
        with col_advice:
            st.subheader("🤖 Strategic Analysis")
            recommendations = generate_policy_advice(
                country, prim, sec, exp, pred_lit, 
                latest['Total_GDP'], pred_total, 
                latest['Education_expenditure']
            )
            for rec in recommendations:
                st.markdown(f"""
                <div class="advice-card" style="border-left-color: {rec['color']};">
                    <h4 class="advice-title">{rec['icon']} {rec['title']}</h4>
                    <p class="advice-insight">{rec['insight']}</p>
                    <p class="advice-action" style="color: {rec['color']};">{rec['action']}</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("Unable to establish connection to World Bank economic data services.")