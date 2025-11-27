import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------------------
# 1. APP CONFIGURATION
# -------------------------------------------
st.set_page_config(
    page_title="Global Growth Predictor",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------------------
# 2. TRAIN MODEL ON THE FLY (The Self-Healing Part)
# -------------------------------------------
@st.cache_resource
def get_model():
    """
    This function trains the model instantly if it's not loaded.
    Since the dataset is synthetic/small, this takes <1 second.
    """
    # A. Generate Data
    countries = ['USA','China','India','Germany','UK','France','Brazil','Japan','Canada','Australia']
    years = list(range(1995, 2030))
    rows = []
    for c in countries:
        base_gdp = np.random.uniform(2000, 60000)
        for y in years:
            rows.append({
                'Country Name': c,
                'Year': y,
                'GDP_per_capita': base_gdp * (1 + 0.03*(y-years[0])) * np.random.normal(1, 0.02),
                'Literacy_rate': min(99.9, 80 + 0.1*(y-years[0])),
                'Primary_enrollment': min(100, 80 + 0.3*(y-years[0])),
                'Secondary_enrollment': min(100, 50 + 0.5*(y-years[0])),
                'Education_expenditure': max(1.5, np.random.uniform(2, 6))
            })
    df = pd.DataFrame(rows)

    # B. Preprocess
    le = LabelEncoder()
    df['Country_Encoded'] = le.fit_transform(df['Country Name'])
    
    X = df[['Country_Encoded', 'Year', 'Primary_enrollment', 'Secondary_enrollment', 'Education_expenditure']].values
    y = df[['GDP_per_capita', 'Literacy_rate']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # C. Train
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_scaled, y)
    
    return scaler, le, model

# Load the model (Builds it once, then remembers it)
scaler, le, model = get_model()

# -------------------------------------------
# 3. SIDEBAR UI
# -------------------------------------------
st.sidebar.header("🔮 Prediction Settings")

if le:
    country_list = list(le.classes_)
    default_ix = country_list.index('USA') if 'USA' in country_list else 0
    country = st.sidebar.selectbox("Select Country", country_list, index=default_ix)
    country_code = le.transform([country])[0]
else:
    country = st.sidebar.text_input("Country", "USA")
    country_code = 0

year = st.sidebar.slider("Target Year", 2025, 2050, 2030)
st.sidebar.markdown("---")
st.sidebar.subheader("Education Parameters")

prim_enroll = st.sidebar.slider("Primary Enrollment (%)", 50.0, 100.0, 95.0)
sec_enroll = st.sidebar.slider("Secondary Enrollment (%)", 30.0, 100.0, 85.0)
edu_exp = st.sidebar.slider("Govt Expenditure (% of GDP)", 1.0, 10.0, 4.5)

predict_btn = st.sidebar.button("Run Prediction", type="primary")

# -------------------------------------------
# 4. MAIN DASHBOARD
# -------------------------------------------
st.title("🌍 AI Economic Forecaster")
st.caption("Powered by Random Forest Regression")

if predict_btn:
    try:
        # Prepare Input
        input_data = np.array([[country_code, year, prim_enroll, sec_enroll, edu_exp]])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        pred_gdp, pred_lit = prediction[0]
        
        # Display Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="💰 Predicted GDP per Capita", value=f"${pred_gdp:,.2f}")
        with col2:
            st.metric(label="📚 Predicted Literacy Rate", value=f"{pred_lit:.2f}%")
            
        st.success("Prediction generated successfully!")
        
        # Visuals
        st.subheader("Visual Projection")
        st.bar_chart(pd.DataFrame({
            'Value': [pred_gdp, pred_lit*100] # scaling literacy for visibility
        }, index=['GDP ($)', 'Literacy (x100)']))
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

elif not predict_btn:
    st.info("👈 Adjust the sliders in the sidebar and click 'Run Prediction' to start.")