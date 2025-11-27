import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import os

# NOTE: TensorFlow removed for compatibility with Python 3.13

# -------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------
st.set_page_config(
    page_title="Global Growth Predictor",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------------------
# 2. LOAD ASSETS
# -------------------------------------------
@st.cache_resource
def load_assets():
    # Load Preprocessors
    if not os.path.exists('preprocessors.pkl'):
        st.error("⚠️ Files missing! Please run 'setup_safe_model.py' first.")
        return None, None, None
        
    with open('preprocessors.pkl', 'rb') as f:
        assets = pickle.load(f)
        scaler = assets['scaler']
        le = assets['label_encoder']
    
    # Load Model (Strictly Pickle/RandomForest)
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        st.error("⚠️ Model missing! Please run 'setup_safe_model.py' first.")
        return None, None, None
        
    return scaler, le, model

scaler, le, model = load_assets()

# -------------------------------------------
# 3. SIDEBAR
# -------------------------------------------
st.sidebar.header("🔮 Prediction Settings")

if le:
    country_list = list(le.classes_)
    default_ix = country_list.index('USA') if 'USA' in country_list else 0
    country = st.sidebar.selectbox("Select Country", country_list, index=default_ix)
else:
    country = st.sidebar.text_input("Country", "USA")

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
st.caption("Powered by Random Forest Regression (No TensorFlow)")

if predict_btn and model and le:
    try:
        # Prepare Data
        country_code = le.transform([country])[0]
        input_data = np.array([[country_code, year, prim_enroll, sec_enroll, edu_exp]])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        pred_gdp, pred_lit = prediction[0]
        
        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="💰 Predicted GDP per Capita", value=f"${pred_gdp:,.2f}")
        with col2:
            st.metric(label="📚 Predicted Literacy Rate", value=f"{pred_lit:.2f}%")
            
        st.success("Prediction generated successfully!")
        
        # Simple Plot
        st.subheader("Visual Projection")
        chart_data = pd.DataFrame({
            'Metric': ['GDP ($)', 'Literacy (%)'],
            'Value': [pred_gdp, pred_lit]
        })
        st.bar_chart(chart_data.set_index('Metric'))
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

elif not predict_btn:
    st.info("👈 Adjust sliders and click 'Run Prediction'")