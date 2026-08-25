import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('data/crop_recommendation.csv')  # N, P, K, temperature, humidity, ph, rainfall, label
X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, 'ml_models/crop_model.pkl')
joblib.dump(le, 'ml_models/label_encoder.pkl')
print("Model accuracy:", model.score(X_test, y_test))  # ~99% [web:21]
