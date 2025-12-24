# 🌍 Global Growth AI (v2.7 Enterprise)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-ff4b4b.svg)
![AI Model](https://img.shields.io/badge/AI-Llama%203.1%20(Groq)-purple)
![Status](https://img.shields.io/badge/Status-Production-success)

> **Developed by Ali Sher Khan Tareen** > *AI-Powered Economic Forecasting & Strategic Simulation Engine*

---

## 📖 Overview
**Global Growth AI** is a production-grade economic simulator designed for policy analysts and strategic planners. It combines a **Random Forest Regression Model** for quantitative forecasting with **Llama 3.1 (via Groq)** for qualitative strategic analysis.

**Version 2.7** introduces **Automatic Authentication (Auto-Auth)**, enhanced security protocols, and logic-locked policy sliders to ensure realistic simulations.

---

## 🚀 Key Features

### 🧠 Dual-Core AI System
* **Quantitative:** Random Forest model predicts GDP growth based on educational inputs (Primary/Secondary Enrollment and Human Capital Index).
* **Qualitative:** Integrated **Llama 3.1** (via Groq API) acts as a Senior Economic Analyst, answering complex questions about the simulation data.

### 🛡️ Enterprise Security
* **Auto-Auth:** Automatically connects to AI services using Cloud Secrets (no manual key entry required).
* **Secure Fallback:** Provides a secure password field for local testing with pre-filled developer keys.

### 📊 Advanced Simulation
* **Time-Series Forecasting:** Dynamic projection slider (2025–2030).
* **Logic Constraints:** Secondary enrollment sliders are physically locked to never exceed Primary enrollment, ensuring data integrity.
* **G20 Support:** Real-time World Bank data for 20+ major economies.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Responsive, component-based UI |
| **LLM Engine** | Groq API | Llama-3.1-8b-Instant (Ultra-low latency) |
| **Predictive AI** | Scikit-Learn | Random Forest Regressor |
| **Data Source** | World Bank API | Real-time macro-economic indicators |
| **Reporting** | FPDF | Dynamic PDF Strategy Brief generation |
| **Visualization** | Plotly | Interactive financial charting |

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/Aly-sher/education-gdp-predictor.git](https://github.com/Aly-sher/education-gdp-predictor.git)
   cd education-gdp-predictor

   ```
   ## 📂 Project Structure

   Global-Growth-AI/
├── app.py           # Main Dashboard UI & Controller
├── utils.py         # AI Engines (RF + Groq), Data Fetching
├── styles.py        # CSS Styling & Enterprise Theme
├── requirements.txt # Project Dependencies
└── README.md        # Documentation

## 📜 Disclaimer
**Predictions are based on a synthetic training model calibrated on World Bank historical data.** 
**For educational and demonstration purposes only.**
   
