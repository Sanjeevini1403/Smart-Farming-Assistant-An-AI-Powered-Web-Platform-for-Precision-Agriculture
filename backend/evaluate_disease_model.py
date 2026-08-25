"""
Run this on a machine with TensorFlow installed (per requirements.txt) to get
REAL accuracy / precision / recall / F1 / confusion-matrix numbers for the
IEEE paper's Section VII-B and Section IX. Do not hand-type numbers into the
paper without running this first.

Usage:
    cd backend
    python evaluate_disease_model.py
"""
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

IMG_SIZE = 224
VAL_DIR = "dataset/val"

model = load_model("ml_models/disease_mobilenet.h5")
with open("ml_models/class_labels.json") as f:
    class_labels = json.load(f)

val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=16,
    class_mode='categorical', shuffle=False
)

# Sanity check: class order from the generator must match class_labels.json
print("Generator class order:", val_gen.class_indices)
print("Saved class_labels.json:", class_labels)

y_true = val_gen.classes
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='macro')

print("\n=== RESULTS TO PUT IN THE PAPER (Section VII-B / IX) ===")
print(f"Validation samples: {len(y_true)}")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision (macro): {precision*100:.2f}%")
print(f"Recall (macro):    {recall*100:.2f}%")
print(f"F1-score (macro):  {f1*100:.2f}%")
print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))
print("\nConfusion matrix (rows=true, cols=predicted):")
print(list(val_gen.class_indices.keys()))
print(confusion_matrix(y_true, y_pred))
