# VGG16 + LSTM

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import build_model

# Define dataset paths
import os
train_dir = os.path.join("Dataset", "Train")
val_dir = os.path.join("Dataset", "Validation")


# Verify data format
print(f"TensorFlow Data Format: {tf.keras.backend.image_data_format()}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_ds = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)
val_ds = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)

# Build model (without TimeDistributed)
model = build_model()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=[early_stopping, reduce_lr])

# Save trained model
model.save("real_fake_classifier_new.h5")

print("Model trained and saved successfully!")
