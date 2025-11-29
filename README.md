# 🌐 Global Growth Outlook AI

A professional-grade economic forecasting engine that uses Machine Learning to predict GDP growth, Literacy rates, and Human Capital development based on government policy inputs.

🔗 **Live Application:** [https://alysher-gdp-predictor.streamlit.app](https://alysher-gdp-predictor.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B) ![Status](https://img.shields.io/badge/Status-Live-success)

## 🚀 Key Features

### 1. Real-World Data Engine
Integrated with the **World Bank API (`wbgapi`)** to fetch 25+ years of live historical economic data for major economies including USA, Pakistan, China, India, and Germany.

### 2. AI Strategic Consultant
Includes a logic-based AI advisor that analyzes inputs against historical baselines to detect:
* **Austerity Risks:** If budget cuts exceed historical norms.
* **Structural Inefficiency:** High spending with low output.
* **The Middle-Income Trap:** High dropout rates between primary and secondary education.

### 3. Professional Visualization
* **Interactive Forecasts:** Powered by **Plotly**, featuring historical trend lines and dashed projection vectors.
* **ROI Analysis:** Calculates real-time Return on Investment for education spending.
* **Glassmorphism UI:** Custom CSS implementation for a modern, fintech-style aesthetic.

## 🛠️ Tech Stack
* **Core:** Python 3.10+
* **Data Source:** World Bank API (wbgapi)
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)
* **Visualization:** Plotly Graph Objects
* **Frontend:** Streamlit

## 💻 Installation & Local Run

```bash
# Clone the repository
git clone [https://github.com/Aly-sher/education-gdp-predictor.git](https://github.com/Aly-sher/education-gdp-predictor.git)

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py