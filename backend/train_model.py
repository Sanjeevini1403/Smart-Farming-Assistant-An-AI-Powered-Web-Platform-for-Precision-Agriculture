import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os
import json

# 🔥 CONFIGURATION
IMG_SIZE = 224
BATCH_SIZE = 8

train_dir = "dataset/train"
val_dir = "dataset/val"

# 🔥 DATA AUGMENTATION (Train only)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 🔥 LOAD TRAINING DATA
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# 🔥 LOAD VALIDATION DATA
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 🔥 GET NUMBER OF CLASSES DYNAMICALLY
num_classes = train_generator.num_classes
print(f"📊 Number of classes detected: {num_classes}")
print(f"📋 Class labels: {train_generator.class_indices}")

# 🔥 SAVE CLASS LABELS FOR LATER USE
class_labels = list(train_generator.class_indices.keys())
os.makedirs("ml_models", exist_ok=True)
with open("ml_models/class_labels.json", "w") as f:
    json.dump(class_labels, f)
print(f"✅ Class labels saved to ml_models/class_labels.json")

# 🔥 LOAD BASE MODEL (MobileNetV2)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Freeze base model

# 🔥 CREATE CUSTOM HEAD
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Add dropout to prevent overfitting
    layers.Dense(num_classes, activation='softmax')  # ✅ DYNAMIC CLASS COUNT
])

# 🔥 COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🔥 TRAIN MODEL
print("🌱 Starting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # Increased to 10 for better accuracy
    verbose=1
)

# 🔥 SAVE MODEL
model.save("ml_models/disease_mobilenet.h5")
print("🔥 Model Trained & Saved Successfully to ml_models/disease_mobilenet.h5!")

# 🔥 MODEL SUMMARY
model.summary()
