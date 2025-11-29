import streamlit as st
import pandas as pd
import numpy as np
import wbgapi as wb
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------
st.set_page_config(page_title="Global Growth AI", page_icon="🌍", layout="wide")

# -------------------------------------------
# 2. REAL DATA ENGINE (Bulletproof Version)
# -------------------------------------------
@st.cache_data
def fetch_world_bank_data():
    """
    Fetches raw data and manually pivots it to avoid Index errors.
    """
    # Define Indicators
    indicators = {
        'NY.GDP.PCAP.CD': 'GDP_per_capita',
        'SE.ADT.LITR.ZS': 'Literacy_rate',
        'SE.PRM.ENRR': 'Primary_enrollment',
        'SE.SEC.ENRR': 'Secondary_enrollment',
        'SE.XPD.TOTL.GD.ZS': 'Education_expenditure'
    }
    
    country_codes = ["USA", "PAK", "IND", "CHN", "GBR", "BRA", "CAN", "AUS", "DEU", "FRA"]
    
    print("⏳ Connecting to World Bank API...")
    
    try:
        # 1. Fetch RAW data (List of dictionaries) - 100% safe format
        # We assume time range 2000-2024
        raw_data = wb.data.fetch(indicators.keys(), economy=country_codes, time=range(2000, 2025))
        
        # 2. Convert to DataFrame
        raw_df = pd.DataFrame(list(raw_data))
        
        if raw_df.empty:
            st.error("API returned no data.")
            return pd.DataFrame()

        # 3. Manual Pivot (This fixes the 'NY.GDP' error)
        # We turn the 'series' column into actual columns (GDP, Literacy, etc.)
        df = raw_df.pivot_table(index=['economy', 'time'], columns='series', values='value')
        
        # 4. Clean Up
        df.reset_index(inplace=True)
        
        # Rename columns using our dictionary
        df.rename(columns=indicators, inplace=True)
        df.rename(columns={'economy': 'Country Code', 'time': 'Year'}, inplace=True)
        
        # Clean Year (remove 'YR')
        df['Year'] = df['Year'].astype(str).str.replace('YR', '', regex=False).astype(int)
        
        # Map Names
        code_map = {
            'USA': 'United States', 'PAK': 'Pakistan', 'IND': 'India', 'CHN': 'China',
            'GBR': 'United Kingdom', 'BRA': 'Brazil', 'CAN': 'Canada', 'AUS': 'Australia',
            'DEU': 'Germany', 'FRA': 'France'
        }
        df['Country Name'] = df['Country Code'].map(code_map)
        
        # Handling Missing Values
        numeric_cols = list(indicators.values())
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df.groupby('Country Code')[col].transform(lambda x: x.interpolate(limit_direction='both'))
                df[col] = df[col].fillna(df[col].mean())
            else:
                # If a column is missing entirely (e.g. Literacy), fill with 0
                df[col] = 0
            
        return df

    except Exception as e:
        st.error(f"Data Processing Error: {e}")
        return pd.DataFrame()

# -------------------------------------------
# 3. AI MODEL TRAINING
# -------------------------------------------
@st.cache_resource
def train_model(df):
    if df.empty: return None, None, None

    # Filter out rows where Country Name might be NaN (if API returned unexpected codes)
    df = df.dropna(subset=['Country Name'])

    le = LabelEncoder()
    df['Country_Encoded'] = le.fit_transform(df['Country Name'].astype(str))
    
    # Ensure correct column order
    feature_cols = ['Country_Encoded', 'Year', 'Primary_enrollment', 'Secondary_enrollment', 'Education_expenditure']
    target_cols = ['GDP_per_capita', 'Literacy_rate']
    
    # Check if all columns exist
    if not all(col in df.columns for col in feature_cols + target_cols):
        st.error("Missing required columns in API data.")
        return None, None, None

    X = df[feature_cols].values
    y = df[target_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_scaled, y)
    
    return scaler, le, model

# Load Data
with st.spinner("Downloading Real World Data..."):
    df = fetch_world_bank_data()
    if not df.empty:
        scaler, le, model = train_model(df)
        success = (model is not None)
    else:
        success = False

# -------------------------------------------
# 4. DASHBOARD UI
# -------------------------------------------
st.title("🌍 Real-World Economic Forecaster")

if success:
    st.markdown("Powered by **World Bank Data (wbgapi)** & Random Forest Regression.")
    
    # Sidebar
    st.sidebar.header("🔮 Settings")
    country_list = list(le.classes_)
    default_ix = country_list.index('Pakistan') if 'Pakistan' in country_list else 0
    country = st.sidebar.selectbox("Select Country", country_list, index=default_ix)

    # Defaults
    country_data = df[df['Country Name'] == country].sort_values('Year')
    if not country_data.empty:
        latest = country_data.iloc[-1]
        def_prim = float(latest['Primary_enrollment'])
        def_sec = float(latest['Secondary_enrollment'])
        def_exp = float(latest['Education_expenditure'])
    else:
        def_prim, def_sec, def_exp = 90.0, 70.0, 4.0

    year = st.sidebar.slider("Target Year", 2025, 2030, 2026)
    st.sidebar.markdown("---")
    st.sidebar.write("### Policy Inputs")
    prim = st.sidebar.slider("Primary Enrollment (%)", 50.0, 120.0, def_prim)
    sec = st.sidebar.slider("Secondary Enrollment (%)", 20.0, 100.0, def_sec)
    exp = st.sidebar.slider("Govt Edu Spend (% GDP)", 1.0, 10.0, def_exp)

    if st.sidebar.button("Run AI Prediction", type="primary"):
        c_code = le.transform([country])[0]
        inputs = scaler.transform([[c_code, year, prim, sec, exp]])
        pred = model.predict(inputs)
        pred_gdp, pred_lit = pred[0]
        
        c1, c2 = st.columns(2)
        c1.metric("Predicted GDP per Capita", f"${pred_gdp:,.2f}")
        c2.metric("Predicted Literacy Rate", f"{pred_lit:.2f}%")
        
        st.subheader(f"GDP Trajectory: {country}")
        hist_chart = country_data[['Year', 'GDP_per_capita']].copy()
        hist_chart['Type'] = 'Historical'
        pred_chart = pd.DataFrame({'Year': [year], 'GDP_per_capita': [pred_gdp], 'Type': ['Prediction']})
        final_chart = pd.concat([hist_chart, pred_chart])
        st.line_chart(final_chart, x='Year', y='GDP_per_capita', color='Type')

else:
    st.warning("Application could not load data. Please check logs.")