# ml_models/market_prediction.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("🌾 Creating Market Price Prediction Model...")

# Sample data (real data venum na Agmarknet API use pannalam)
data = {
    'crop': ['rice', 'rice', 'rice', 'wheat', 'wheat', 'maize', 'maize', 'tomato', 'tomato', 'onion'],
    'season': ['kharif', 'rabi', 'kharif', 'rabi', 'kharif', 'kharif', 'rabi', 'kharif', 'rabi', 'rabi'],
    'state': ['Tamil Nadu', 'Punjab', 'Karnataka', 'Punjab', 'Haryana', 'Karnataka', 'Maharashtra', 'Maharashtra', 'Karnataka', 'Gujarat'],
    'demand': [85, 90, 88, 75, 78, 82, 80, 95, 92, 88],  # 0-100 scale
    'production': [92, 88, 95, 85, 82, 78, 75, 65, 70, 72],  # 0-100 scale
    'price_per_quintal': [2100, 2200, 2050, 2400, 2350, 1900, 1950, 1800, 1750, 2600]
}

df = pd.DataFrame(data)

# Encode categorical variables
le_crop = LabelEncoder()
le_season = LabelEncoder()
le_state = LabelEncoder()

df['crop_encoded'] = le_crop.fit_transform(df['crop'])
df['season_encoded'] = le_season.fit_transform(df['season'])
df['state_encoded'] = le_state.fit_transform(df['state'])

# Features & Target
X = df[['crop_encoded', 'season_encoded', 'state_encoded', 'demand', 'production']]
y = df['price_per_quintal']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save models
os.makedirs("ml_models", exist_ok=True)
joblib.dump(model, "ml_models/market_model.pkl")
joblib.dump(le_crop, "ml_models/crop_encoder.pkl")
joblib.dump(le_season, "ml_models/season_encoder.pkl")
joblib.dump(le_state, "ml_models/state_encoder.pkl")

print("✅ Market prediction model saved!")
print(f"✅ Training accuracy: {model.score(X, y):.2%}")

# Save sample data for demo
df.to_csv("ml_models/market_sample_data.csv", index=False)
print("✅ Sample data saved!")
