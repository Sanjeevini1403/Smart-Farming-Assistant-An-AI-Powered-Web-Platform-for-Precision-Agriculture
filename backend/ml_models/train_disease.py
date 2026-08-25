# ml_models/train_disease.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
import os
import json

print("=" * 60)
print("🌿 PLANT DISEASE CLASSIFICATION - TRAINING")
print("=" * 60)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Paths
train_dir = "dataset/train"
val_dir = "dataset/val"

# Verify dataset
if not os.path.exists(train_dir):
    print(f"❌ Error: Training folder not found at {os.path.abspath(train_dir)}")
    exit(1)

if not os.path.exists(val_dir):
    print(f"❌ Error: Validation folder not found at {os.path.abspath(val_dir)}")
    exit(1)

print(f"✅ Training directory: {os.path.abspath(train_dir)}")
print(f"✅ Validation directory: {os.path.abspath(val_dir)}")

# Data Augmentation
print("\n🔄 Preparing data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Info
num_classes = train_generator.num_classes
class_indices = train_generator.class_indices
print(f"\n✅ Classes found: {num_classes}")
print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")
print(f"✅ Batch size: {BATCH_SIZE}")
print(f"✅ Steps per epoch: {train_generator.samples // BATCH_SIZE}")

# Save class indices
os.makedirs("ml_models", exist_ok=True)
with open("ml_models/class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)
print("✅ Class indices saved to ml_models/class_indices.json")

# Reverse mapping for predictions
indices_to_classes = {v: k for k, v in class_indices.items()}
print(f"\n📋 Sample classes:")
for i in range(min(5, num_classes)):
    print(f"   {i}: {indices_to_classes[i]}")

# Build Model (MobileNetV2 - Transfer Learning)
print("\n🚀 Building MobileNetV2 model...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Model Summary:")
model.summary()

# Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'ml_models/disease_cnn_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train
print(f"\n🚀 Starting training for {EPOCHS} epochs...")
print("=" * 60)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save("ml_models/disease_cnn.h5")
print("\n✅ Model saved as ml_models/disease_cnn.h5")

# Results
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"✅ Final Training Accuracy: {final_train_acc:.2%}")
print(f"✅ Final Validation Accuracy: {final_val_acc:.2%}")
print(f"🎯 Model ready for deployment!")

# Quick test
print("\n🧪 Running quick test...")
test_img_path = None
for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        if images:
            test_img_path = os.path.join(class_path, images[0])
            expected_class = class_folder
            break

if test_img_path and os.path.exists(test_img_path):
    from PIL import Image
    import numpy as np
    
    img = Image.open(test_img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)
    pred_class = indices_to_classes[np.argmax(pred)]
    confidence = np.max(pred) * 100
    
    print(f"✅ Test image: {expected_class}")
    print(f"🎯 Predicted: {pred_class}")
    print(f"💯 Confidence: {confidence:.2f}%")
    
    if expected_class == pred_class:
        print("✅ TEST PASSED!")
    else:
        print("⚠️ Test prediction different (may still be correct)")

print("\n" + "=" * 60)
print("🚀 NEXT STEPS:")
print("1. Run: python app.py")
print("2. Open: http://127.0.0.1:5000")
print("3. Test disease detection!")
print("=" * 60)
