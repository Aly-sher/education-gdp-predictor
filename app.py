import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import plotly.express as px
import os

# -------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------
st.set_page_config(
    page_title="Global Growth Predictor",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------------------
# 2. LOAD ASSETS (Cached for speed)
# -------------------------------------------
@st.cache_resource
def load_assets():
    # Load Preprocessors
    if not os.path.exists('preprocessors.pkl'):
        st.error("Error: 'preprocessors.pkl' not found. Run main.py first.")
        return None, None, None
        
    with open('preprocessors.pkl', 'rb') as f:
        assets = pickle.load(f)
        scaler = assets['scaler']
        le = assets['label_encoder']
    
    # Load Model (Handle Keras or Pickle)
    model = None
    if os.path.exists('best_model.keras'):
        model = tf.keras.models.load_model('best_model.keras')
    elif os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        st.error("Error: No model file found. Run main.py first.")
        
    return scaler, le, model

# Load Data for Visualization
@st.cache_data
def load_data():
    if os.path.exists('cleaned_education_gdp_dataset.csv'):
        return pd.read_csv('cleaned_education_gdp_dataset.csv')
    return pd.DataFrame() # Return empty if missing

scaler, le, model = load_assets()
df = load_data()

# -------------------------------------------
# 3. SIDEBAR - USER INPUTS
# -------------------------------------------
st.sidebar.header("🔮 Prediction Settings")

# Dynamic Country List
if le:
    country_list = list(le.classes_)
    default_ix = country_list.index('USA') if 'USA' in country_list else 0
    country = st.sidebar.selectbox("Select Country", country_list, index=default_ix)
else:
    country = st.sidebar.text_input("Country", "USA")

year = st.sidebar.slider("Target Year", 2025, 2050, 2030)

st.sidebar.markdown("---")
st.sidebar.subheader("Education Parameters")

# Defaults based on selected country history (if available)
def_prim, def_sec, def_exp = 95.0, 85.0, 4.5
if not df.empty and country in df['Country Name'].values:
    last_row = df[df['Country Name'] == country].sort_values('Year').iloc[-1]
    def_prim = float(last_row['Primary_enrollment'])
    def_sec = float(last_row['Secondary_enrollment'])
    def_exp = float(last_row['Education_expenditure'])

prim_enroll = st.sidebar.slider("Primary Enrollment (%)", 0.0, 100.0, def_prim)
sec_enroll = st.sidebar.slider("Secondary Enrollment (%)", 0.0, 100.0, def_sec)
edu_exp = st.sidebar.slider("Govt Expenditure (% of GDP)", 1.0, 10.0, def_exp)

predict_btn = st.sidebar.button("Run Prediction", type="primary")

# -------------------------------------------
# 4. MAIN DASHBOARD
# -------------------------------------------
st.title("🌍 AI Economic Forecaster")
st.markdown(f"Predicting GDP and Literacy for **{country}** in **{year}** based on educational investment.")

if predict_btn and model and le:
    # A. PREPROCESS INPUT
    try:
        country_code = le.transform([country])[0]
        input_data = np.array([[country_code, year, prim_enroll, sec_enroll, edu_exp]])
        input_scaled = scaler.transform(input_data)
        
        # B. PREDICT
        if isinstance(model, tf.keras.Model):
            prediction = model.predict(input_scaled, verbose=0)
        else:
            prediction = model.predict(input_scaled)
            
        pred_gdp, pred_lit = prediction[0]
        
        # C. DISPLAY METRICS
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="💰 Predicted GDP per Capita", value=f"${pred_gdp:,.2f}")
        with col2:
            st.metric(label="📚 Predicted Literacy Rate", value=f"{pred_lit:.2f}%")
            
        st.success("Prediction generated successfully!")
        
        # D. VISUALIZATION (Plotly)
        if not df.empty and country in df['Country Name'].values:
            st.subheader("📈 Historical Trends vs. Prediction")
            
            country_data = df[df['Country Name'] == country].sort_values('Year')
            
            # Create a combined dataframe for plotting
            future_data = pd.DataFrame({
                'Year': [year],
                'GDP_per_capita': [pred_gdp],
                'Literacy_rate': [pred_lit],
                'Type': ['Prediction']
            })
            country_data['Type'] = 'Historical'
            
            # Combine history and prediction (just for plotting the points)
            plot_df = pd.concat([country_data[['Year','GDP_per_capita','Literacy_rate','Type']], future_data])
            
            # GDP Chart
            fig_gdp = px.line(country_data, x='Year', y='GDP_per_capita', title=f"{country}: GDP Trajectory")
            fig_gdp.add_scatter(x=[year], y=[pred_gdp], mode='markers', marker=dict(size=15, color='red'), name='Prediction')
            st.plotly_chart(fig_gdp, use_container_width=True)
            
            # Literacy Chart
            fig_lit = px.line(country_data, x='Year', y='Literacy_rate', title=f"{country}: Literacy Trajectory")
            fig_lit.add_scatter(x=[year], y=[pred_lit], mode='markers', marker=dict(size=15, color='red'), name='Prediction')
            st.plotly_chart(fig_lit, use_container_width=True)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        
elif not predict_btn:
    st.info("👈 Adjust the settings in the sidebar and click 'Run Prediction' to start.")