# 🚀 Global Growth AI

**A Full-Stack Economic Forecasting Engine powered by Machine Learning & Generative AI.**

This application predicts national GDP growth and Literacy rates based on education policy inputs. It combines **Predictive Modeling** (Random Forest) with **Generative Analysis** (Llama 3) to provide real-time strategic economic advice.

🔗 **Live Application:** [https://alysher-gdp-predictor.streamlit.app](https://alysher-gdp-predictor.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B) ![GenAI](https://img.shields.io/badge/AI-Llama_3-violet) ![API](https://img.shields.io/badge/Data-World_Bank-green)

---

## ✨ Key Features

### 🧠 1. Generative AI Analyst (New!)
Integrated **Llama 3.3 (via Groq API)** to act as an autonomous economic consultant.
* Users can **chat with the data** in real-time.
* The AI reads the specific simulation results and explains the "Why" behind the numbers.

### 📊 2. Real-Time Data Engine
Connected to the **World Bank API (`wbgapi`)** to fetch live historical data (2000–Present) for major economies including USA, Pakistan, China, India, and Germany. No static CSVs used.

### 📈 3. Predictive Analytics
* **Model:** Multi-Output Random Forest Regressor.
* **Function:** Predicts **Total GDP**, **GDP Per Capita**, and **Literacy Rate** for target years (2025–2030).
* **Self-Healing Pipeline:** Automatically handles missing data points via intelligent interpolation.

### 🎨 4. Professional UI/UX
* **Glassmorphism Design:** Custom CSS implementation for a modern Fintech aesthetic.
* **Interactive Charts:** Powered by **Plotly** for zooming, hovering, and comparing historical trends vs. future projections.
* **Smart Logic:** Includes logic guardrails (e.g., detecting impossible enrollment stats) and strategic sanity checks.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Generative AI** | Groq API (Llama 3.3-70b) |
| **Machine Learning** | Scikit-Learn (Random Forest) |
| **Data Source** | World Bank API (`wbgapi`) |
| **Visualization** | Plotly Graph Objects |
| **Frontend** | Streamlit (Python) |
| **Data Processing** | Pandas, NumPy |

---

## 💻 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Aly-sher/education-gdp-predictor.git](https://github.com/Aly-sher/education-gdp-predictor.git)
   cd education-gdp-predictor