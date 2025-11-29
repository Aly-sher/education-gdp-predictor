# 🌍 Real-Time AI Economic Forecaster

A live Machine Learning dashboard that forecasts **GDP per Capita** and **Literacy Rates** based on government educational investments.

🔗 **Live App:** [https://alysher-gdp-predictor.streamlit.app](https://alysher-gdp-predictor.streamlit.app)

## 🚀 Key Features
- **Real-World Data Engine:** Connects to the **World Bank API** to fetch 25+ years of historical data for 10 major economies (USA, Pakistan, China, India, etc.).
- **Self-Healing Pipeline:** Automatically handles missing values using interpolation and trains the model on-the-fly.
- **Scenario Simulation:** allows users to adjust policy levers (Primary Enrollment, Govt Spend) to see the predicted economic impact in 2025-2030.

## 🛠️ Tech Stack
- **Data Source:** World Bank API (`wbgapi`)
- **Machine Learning:** Scikit-Learn (Random Forest Regressor)
- **App Framework:** Streamlit
- **Data Processing:** Pandas & NumPy

## 📉 How It Works
1. **ETL Layer:** Fetches raw data for indicators like `NY.GDP.PCAP.CD` (GDP) and `SE.XPD.TOTL.GD.ZS` (Edu Spend).
2. **Preprocessing:** Pivots the data, cleans text artifacts, and fills missing time-series data via linear interpolation.
3. **Modeling:** Trains a Multi-Output Random Forest Regressor to map education inputs to economic outputs.
4. **Inference:** Generates predictions based on user-defined inputs for future years.

## 💻 Run Locally
```bash
git clone [https://github.com/Aly-sher/education-gdp-predictor.git](https://github.com/Aly-sher/education-gdp-predictor.git)
pip install -r requirements.txt
streamlit run app.py
