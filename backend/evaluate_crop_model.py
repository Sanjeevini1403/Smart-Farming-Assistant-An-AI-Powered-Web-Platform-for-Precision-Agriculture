"""
Run this to get REAL accuracy / precision / recall / F1 numbers for the crop
Random Forest model, for the IEEE paper's Section VII-B and Section IX.

IMPORTANT: backend/data/crop_recommendation.csv in this repo is currently just
a kagglehub download stub (a few lines of Python), not the actual dataset.
Download the real file first:

    pip install kagglehub
    python -c "import kagglehub; print(kagglehub.dataset_download('atharvaingle/crop-recommendation-dataset'))"

...then copy the resulting Crop_recommendation.csv to backend/data/crop_recommendation.csv
before running this script. This will also let you retrain crop_model.pkl on
the full 22-crop dataset (see ml_models/train_crop.py) instead of the reduced
6-crop local subset currently shipped.

Also note: ml_models/train_crop.py currently calls train_test_split() WITHOUT a
random_state, so the split (and therefore accuracy) is different every time you
retrain. This script fixes random_state=42 for a reproducible split — apply the
same fix in train_crop.py before retraining the shipped model.

Usage:
    cd backend
    python evaluate_crop_model.py

This script now also runs stratified k-fold cross-validation and compares the
Random Forest against KNN and SVM baselines, as referenced in the IEEE paper's
Section VII-C / IX (Table IV) — required before any cross-validated accuracy
or baseline-comparison figure can be reported for the crop-recommendation model.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

df = pd.read_csv('data/crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Fixed random_state so this split is reproducible for the paper.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

print("=== HOLD-OUT RESULTS (80/20 split, random_state=42) ===")
print(f"Total samples: {len(df)}  |  Classes: {len(le.classes_)}  |  Train/test: {len(X_train)}/{len(X_test)}")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision (macro): {precision*100:.2f}%")
print(f"Recall (macro):    {recall*100:.2f}%")
print(f"F1-score (macro):  {f1*100:.2f}%")
print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Stratified k-fold cross-validation (Section VII-C / IX, Table IV) ---
print("\n=== STRATIFIED 5-FOLD CROSS-VALIDATION ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y_encoded, cv=skf)
print(f"Per-fold accuracy: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# --- Baseline comparison: KNN and SVM, as referenced against [8] ---
print("\n=== BASELINE COMPARISON (same 80/20 split) ===")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for name, clf, use_scaled in [
    ("Random Forest (proposed)", RandomForestClassifier(n_estimators=100, random_state=42), False),
    ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5), True),
    ("SVM (RBF kernel)", SVC(kernel='rbf', random_state=42), True),
]:
    Xtr, Xte = (X_train_s, X_test_s) if use_scaled else (X_train, X_test)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    a = accuracy_score(y_test, pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average='macro')
    print(f"{name:28s} | Accuracy: {a*100:6.2f}% | Precision: {p*100:6.2f}% | Recall: {r*100:6.2f}% | F1: {f*100:6.2f}%")

print("\nCopy the numbers above into the paper's Section IX / Table IV once you")
print("have retrained on the full dataset — do not hand-type estimated values.")
