
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Colab file upload helper
try:
    from google.colab import files
except Exception:
    files = None

# Widgets for interactive UI
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except Exception:
    WIDGETS_AVAILABLE = False

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
np.random.seed(42)
tf.random.set_seed(42)

print("Libraries loaded. Environment ready.\n")

# -------------------------
# Utilities
# -------------------------
def format_currency(value):
    try:
        v = float(value)
    except: return str(value)
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.2f}M"
    if v >= 1e3: return f"${v/1e3:.2f}K"
    return f"${v:,.2f}"

def format_percent(value):
    try:
        return f"{float(value):.2f}%"
    except: return str(value)

# -------------------------
# STEP 1: Load & Generate Data
# -------------------------
print("STEP 1 — Load dataset\n")

# Fallback synthetic generator
def generate_synthetic_data():
    countries = ['USA','China','India','Germany','UK','France','Brazil','Japan','Canada','Australia']
    years = list(range(1995, 2026))
    rows=[]
    for c in countries:
        # Create different baselines for different countries
        base_gdp = np.random.uniform(2000, 60000)
        base_lit = np.random.uniform(70, 99)
        
        for y in years:
            rows.append({
                'Country Name': c,
                'Country Code': c[:3].upper(),
                'Year': y,
                'GDP_per_capita': base_gdp * (1 + 0.03*(y-years[0])) * np.random.normal(1, 0.02),
                'Literacy_rate': min(99.9, base_lit + 0.1*(y-years[0]) + np.random.normal(0, 0.5)),
                'Primary_enrollment': min(100, 80 + 0.3*(y-years[0]) + np.random.normal(0, 2)),
                'Secondary_enrollment': min(100, 50 + 0.5*(y-years[0]) + np.random.normal(0, 3)),
                'Education_expenditure': max(1.5, np.random.uniform(2, 6))
            })
    return pd.DataFrame(rows)

df = None
# Logic: If running locally and file exists, load it. If Colab, ask upload. Else generate.
if files:
    print("Upload 'cleaned_education_gdp_dataset.csv' (Cancel to generate synthetic data).")
    try:
        uploaded = files.upload()
        if uploaded:
            filename = list(uploaded.keys())[0]
            df = pd.read_csv(filename)
            print(f"Loaded: {filename}")
    except: pass

if df is None:
    # Check local
    if os.path.exists('cleaned_education_gdp_dataset.csv'):
        df = pd.read_csv('cleaned_education_gdp_dataset.csv')
        print("Loaded local CSV.")
    else:
        df = generate_synthetic_data()
        print("Generated synthetic dataset.")

# Normalize columns
expected_cols = ['Country Name','Year','GDP_per_capita','Literacy_rate',
                 'Primary_enrollment','Secondary_enrollment','Education_expenditure']
for c in expected_cols:
    if c not in df.columns and c.lower() in df.columns:
        df[c] = df[c.lower()]

# -------------------------
# STEP 2: Advanced Cleaning (Groupwise Imputation)
# -------------------------
print("\nSTEP 2 — Cleaning & Feature Engineering")

# 1. Sort for interpolation
df = df.sort_values(['Country Name', 'Year'])

# 2. Groupwise Interpolation (Better than global mean)
# We linear interpolate missing values within a country's timeline
numeric_cols = ['GDP_per_capita','Literacy_rate','Primary_enrollment',
                'Secondary_enrollment','Education_expenditure']

for col in numeric_cols:
    df[col] = df.groupby('Country Name')[col].transform(lambda x: x.interpolate(limit_direction='both'))

# 3. Fill remaining NaNs (if a country has NO data for a column) with global mean
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

df = df.dropna(subset=['GDP_per_capita']) # Target cannot be NaN

# 4. Feature Engineering: Label Encode Country
# This allows the model to learn "USA usually has high GDP" vs "India usually has medium GDP"
le = LabelEncoder()
df['Country_Encoded'] = le.fit_transform(df['Country Name'])

print("Data Cleaned. Shape:", df.shape)

# -------------------------
# STEP 3: Setup Features & Split
# -------------------------
# Included 'Country_Encoded' so models know WHICH country they are predicting
feature_columns = ['Country_Encoded', 'Year', 'Primary_enrollment', 
                   'Secondary_enrollment', 'Education_expenditure']
target_columns = ['GDP_per_capita', 'Literacy_rate']

X = df[feature_columns].values
y = df[target_columns].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (Fit on Train, Transform on Test)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Optional: Scale Y for Neural Networks to converge faster (inverse transform later)
# For simplicity here, we keep Y unscaled to interpret RMSE easily.

# -------------------------
# STEP 4: Modeling
# -------------------------
print("\nSTEP 4 — Training Models...")
models = {}
results = {}

def compute_metrics(y_true, y_pred, model_name):
    print(f"--- {model_name} ---")
    metrics = {}
    for i, col in enumerate(target_columns):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        metrics[col] = {'RMSE': rmse, 'R2': r2}
        print(f"{col}: RMSE={rmse:.2f} | R2={r2:.4f}")
    results[model_name] = metrics
    print("")

# 1. Random Forest (Tree models handle encoded categories well)
rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
rf.fit(X_train_scaled, y_train)
compute_metrics(y_test, rf.predict(X_test_scaled), 'RandomForest')
models['RandomForest'] = rf

# 2. Gradient Boosting
gb = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
gb.fit(X_train_scaled, y_train)
compute_metrics(y_test, gb.predict(X_test_scaled), 'GradientBoosting')
models['GradientBoosting'] = gb

# 3. Neural Network (Improved Architecture)
def build_nn(input_dim, output_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return model

nn = build_nn(X_train_scaled.shape[1], y_train.shape[1])
history = nn.fit(
    X_train_scaled, y_train, 
    validation_split=0.2, 
    epochs=150, batch_size=32, verbose=0,
    callbacks=[callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
)
compute_metrics(y_test, nn.predict(X_test_scaled, verbose=0), 'NeuralNetwork')
models['NeuralNetwork'] = nn

# -------------------------
# STEP 5: Selection & Interactive Dashboard
# -------------------------
# Select Best Model based on Avg R2
best_name = max(results, key=lambda x: np.mean([results[x][t]['R2'] for t in target_columns]))
best_model = models[best_name]
print(f"BEST MODEL SELECTED: {best_name}")

if WIDGETS_AVAILABLE:
    print("\n--- INTERACTIVE DASHBOARD ---\n")
    
    # Widget Definitions
    w_country = widgets.Dropdown(options=sorted(df['Country Name'].unique()), description='Country:')
    w_year = widgets.IntText(value=2030, description='Year:')
    w_prim = widgets.FloatText(value=95.0, description='Primary %:')
    w_sec = widgets.FloatText(value=85.0, description='Secondary %:')
    w_exp = widgets.FloatText(value=4.5, description='Edu Exp %:')
    btn = widgets.Button(description='Predict', button_style='primary')
    out = widgets.Output()

    def on_click(b):
        with out:
            clear_output()
            # Prepare Input
            try:
                # Encode country using the fitted LabelEncoder
                c_code = le.transform([w_country.value])[0]
                
                # Create input array (matches feature_columns order)
                input_data = np.array([[c_code, w_year.value, w_prim.value, w_sec.value, w_exp.value]])
                
                # Scale
                input_scaled = scaler_X.transform(input_data)
                
                # Predict
                if isinstance(best_model, keras.Model):
                    pred = best_model.predict(input_scaled, verbose=0)
                else:
                    pred = best_model.predict(input_scaled)
                
                gdp_p, lit_p = pred[0]
                
                print(f"Prediction for {w_country.value} in {w_year.value}:")
                print(f"-"*30)
                print(f"GDP per Capita: {format_currency(gdp_p)}")
                print(f"Literacy Rate:  {format_percent(lit_p)}")
                print(f"-"*30)
                
                # Visual Context
                hist = df[df['Country Name'] == w_country.value]
                plt.figure(figsize=(10, 4))
                
                plt.subplot(1, 2, 1)
                sns.lineplot(x=hist['Year'], y=hist['GDP_per_capita'], label='History')
                plt.scatter([w_year.value], [gdp_p], c='red', s=100, label='Prediction')
                plt.title('GDP Trend')
                
                plt.subplot(1, 2, 2)
                sns.lineplot(x=hist['Year'], y=hist['Literacy_rate'], label='History', color='green')
                plt.scatter([w_year.value], [lit_p], c='red', s=100, label='Prediction')
                plt.title('Literacy Trend')
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Error: {str(e)}")

    btn.on_click(on_click)
    display(widgets.VBox([w_country, w_year, w_prim, w_sec, w_exp, btn]), out)

# -------------------------
# STEP 6: Robust Saving
# -------------------------
print("\nSTEP 6 — Saving Artifacts")

# Save Preprocessors (Safe to pickle)
with open('preprocessors.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler_X, 'label_encoder': le}, f)
print("Saved preprocessors.pkl")

# Save Model (Conditional logic)
if isinstance(best_model, keras.Model):
    best_model.save('best_model.keras')
    print(f"Saved {best_name} as best_model.keras")
else:
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Saved {best_name} as best_model.pkl")