import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Assume data/plantvillage/ with train/val folders
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2)
train_gen = train_datagen.flow_from_directory('data/plantvillage/train', target_size=(224,224), batch_size=32)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(38, activation='softmax')  # 38 classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=10)

model.save('ml_models/disease_cnn.h5')
