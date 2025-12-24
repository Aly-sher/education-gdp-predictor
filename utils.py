# utils.py
import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from fpdf import FPDF
import base64

# --- 1. EXPANDED DATA FETCHING ---
# G20 + Major Economies
COUNTRY_MAP = {
    "United States": "US", "China": "CN", "India": "IN", "Germany": "DE", 
    "Japan": "JP", "United Kingdom": "GB", "France": "FR", "Brazil": "BR", 
    "Italy": "IT", "Canada": "CA", "Russia": "RU", "South Korea": "KR", 
    "Australia": "AU", "Mexico": "MX", "Indonesia": "ID", "Saudi Arabia": "SA", 
    "Turkey": "TR", "Argentina": "AR", "South Africa": "ZA", "Pakistan": "PK"
}

@st.cache_data
def get_world_bank_data(country_code, indicator):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&date=2000:2023"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if len(data) > 1 and isinstance(data[1], list):
            df = pd.DataFrame(data[1])
            df['value'] = pd.to_numeric(df['value'])
            df['date'] = pd.to_numeric(df['date'])
            df = df.sort_values('date')
            return df.dropna(subset=['value'])
    except Exception:
        pass
    return pd.DataFrame()

# --- 2. REPORT GENERATOR (New Feature) ---
def create_pdf_report(country, current_gdp, projected_gdp, p_enroll, s_enroll, hci):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Title
    pdf.cell(200, 10, txt=f"Economic Strategy Report: {country}", ln=True, align='C')
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Executive Summary", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=f"This report outlines the projected economic impact of educational policy shifts in {country}. "
                               f"Based on the AI simulation, the economy is projected to grow from {current_gdp:.2f}% to {projected_gdp:.2f}%.")
    pdf.ln(5)
    
    # Parameters
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Simulation Parameters", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"- Primary Enrollment Target: {p_enroll}%", ln=True)
    pdf.cell(200, 10, txt=f"- Secondary Enrollment Target: {s_enroll}%", ln=True)
    pdf.cell(200, 10, txt=f"- Human Capital Index: {hci}", ln=True)
    pdf.ln(5)
    
    # Strategic Advice
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="3. AI Recommendations", ln=True)
    pdf.set_font("Arial", size=11)
    
    if s_enroll < p_enroll - 10:
        pdf.multi_cell(0, 10, txt="- CRITICAL: Reduce drop-out rates between primary and secondary levels.")
    if hci < 0.5:
        pdf.multi_cell(0, 10, txt="- PRIORITY: Increase healthcare spending to boost HCI score.")
    else:
        pdf.multi_cell(0, 10, txt="- STRATEGY: Focus on R&D and Technology infrastructure.")
        
    return pdf.output(dest='S').encode('latin-1')

# --- 3. AI ENGINE ---
class EconomicAI:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def train(self):
        np.random.seed(42)
        X_train = []
        y_train = []
        for _ in range(500):
            p = np.random.uniform(50, 100)
            s = np.random.uniform(30, 100)
            h = np.random.uniform(0.3, 0.9)
            penalty = -5.0 if s > p else 0
            growth_impact = (p * 0.02) + (s * 0.04) + (h * 3.5) + penalty + np.random.normal(0, 0.2)
            X_train.append([p, s, h])
            y_train.append(growth_impact)
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict_impact(self, primary, secondary, hci):
        if not self.is_trained:
            self.train()
        return self.model.predict(np.array([[primary, secondary, hci]]))[0]

ai_engine = EconomicAI()