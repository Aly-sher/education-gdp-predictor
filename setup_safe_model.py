# setup_safe_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("⏳ Generating safe Random Forest model...")

# 1. Generate Synthetic Data (Same as before)
countries = ['USA','China','India','Germany','UK','France','Brazil','Japan','Canada','Australia']
years = list(range(1995, 2030))
rows = []
for c in countries:
    base_gdp = np.random.uniform(2000, 60000)
    base_lit = np.random.uniform(70, 99)
    for y in years:
        rows.append({
            'Country Name': c,
            'Year': y,
            'GDP_per_capita': base_gdp * (1 + 0.03*(y-years[0])) * np.random.normal(1, 0.02),
            'Literacy_rate': min(99.9, base_lit + 0.1*(y-years[0])),
            'Primary_enrollment': min(100, 80 + 0.3*(y-years[0])),
            'Secondary_enrollment': min(100, 50 + 0.5*(y-years[0])),
            'Education_expenditure': max(1.5, np.random.uniform(2, 6))
        })
df = pd.DataFrame(rows)

# 2. Preprocessing
le = LabelEncoder()
df['Country_Encoded'] = le.fit_transform(df['Country Name'])

X = df[['Country_Encoded', 'Year', 'Primary_enrollment', 'Secondary_enrollment', 'Education_expenditure']].values
y = df[['GDP_per_capita', 'Literacy_rate']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train Random Forest (No TensorFlow!)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_scaled, y)

# 4. Save
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('preprocessors.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler, 'label_encoder': le}, f)

print("✅ Success! Created 'best_model.pkl' (TensorFlow-free).")