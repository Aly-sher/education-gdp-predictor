# utils.py
import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from fpdf import FPDF
from groq import Groq

# --- 1. DATA FETCHING ---
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
        response = requests.get(url, timeout=3)
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

# --- 2. REAL AI CHATBOT (GROQ) ---
def consult_groq_ai(api_key, user_query, country, current_gdp, projected_gdp, p_enroll, s_enroll, hci, year):
    """
    Sends the simulation context to Groq (Llama 3.1) for a professional economic analysis.
    """
    try:
        client = Groq(api_key=api_key)
        
        system_context = f"""
        You are a Senior Economic Strategist for the World Bank.
        Current Analysis for: {country}
        - Baseline Growth: {current_gdp:.2f}%
        - AI Projected Growth ({year}): {projected_gdp:.2f}%
        - Policy Settings: Primary Edu {p_enroll}%, Secondary Edu {s_enroll}%, Human Capital Index {hci}.
        
        User Query: {user_query}
        
        Instructions:
        1. Answer strictly based on economic principles.
        2. Reference the specific numbers above to prove you analyzed the data.
        3. Keep answers concise (under 3 sentences) but insightful.
        4. If the projection is high, mention HCI as a driver. If low, mention dropout rates.
        """

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", # <--- UPDATED MODEL NAME
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=150,
            top_p=1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq AI: {str(e)}. Please check your API Key."

# --- 3. REPORT GENERATOR ---
def create_pdf_report(country, current_gdp, projected_gdp, p_enroll, s_enroll, hci):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Strategy Report: {country}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt=f"Executive Summary:\nThe AI simulation projects an economic shift from {current_gdp:.2f}% to {projected_gdp:.2f}% based on your policy adjustments.\n\nParameters:\n- Primary Enrollment: {p_enroll}%\n- Secondary Enrollment: {s_enroll}%\n- Workforce Quality (HCI): {hci}")
    return pdf.output(dest='S').encode('latin-1')

# --- 4. PREDICTION ENGINE ---
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