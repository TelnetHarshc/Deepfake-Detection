import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define dataset paths
train_dir = "Train"
val_dir = "Validate"
test_dir = "Test"

# Load datasets
img_size = (224, 224)
batch_size = 32

train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=img_size, batch_size=batch_size
)
test_ds = keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=img_size, batch_size=batch_size
)

# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))



#build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,          # Stop after 3 epochs of no improvement
    restore_best_weights=True  # Restore best weights
)

# Train the model with early stopping
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,  # Increase epochs since early stopping will stop if needed
    callbacks=[early_stopping]
)
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
model.save("real_fake_classifier.h5")

