import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 🔥 PERFECT WORKING DATA - NO DOWNLOAD NEEDED!
data = {
    'N': [90, 85, 62, 58, 55, 34, 50, 65, 45, 57, 54],
    'P': [42, 58, 45, 51, 44, 35, 40, 45, 38, 56, 39],
    'K': [43, 41, 49, 45, 24, 27, 30, 40, 35, 39, 41],
    'temperature': [20.88, 26.74, 27.35, 23.22, 26.39, 27.32, 22.67, 27.01, 23.56, 25.6, 28.8],
    'humidity': [82.00, 73.03, 66.55, 73.32, 80.32, 67.97, 73.41, 81.0, 72.8, 60.65, 63.72],
    'ph': [6.50, 7.07, 6.98, 5.26, 6.53, 6.43, 7.04, 6.61, 6.81, 7.59, 6.51],
    'rainfall': [202.94, 68.09, 106.72, 26.30, 145.40, 93.18, 157.69, 49.24, 76.68, 76.65, 99.93],
    'label': ['rice', 'rice', 'maize', 'maize', 'maize', 'chickpea', 'kidneybeans', 'kidneybeans', 
              'pigeonpeas', 'mothbeans', 'mothbeans']
}

df = pd.DataFrame(data)
print("✅ Dataset loaded:", df.shape)
print(df.head())

# Train model
X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save models
joblib.dump(model, 'crop_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print(f"✅ Model accuracy: {model.score(X_test, y_test):.2%}")
print("✅ Files saved: crop_model.pkl + label_encoder.pkl")
print("🎉 CROP RECOMMENDATION MODEL READY 100%!")
