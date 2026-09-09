# from dataset import load_datasets
# from model import build_model
# import tensorflow as tf

# # Step 1: Load data
# train_ds, val_ds, test_ds = load_datasets()

# # Step 2: Build model
# model = build_model()

# # Step 3: Train model
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=20
# )

# # Step 4: Evaluate on test data
# test_loss, test_acc = model.evaluate(test_ds)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")

# # Step 5: Save the model
# model.save("real_fake_classifier_new.h5")
# print("✅ Model saved as real_fake_classifier_new.h5")
from dataset import load_datasets
from model import build_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Step 1: Load data
train_ds, val_ds, test_ds = load_datasets()

# Step 2: Build model
model = build_model()

# Step 3: Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Step 4: Train model with callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# Step 5: Evaluate on test data
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n🎯 Test Accuracy: {test_acc * 100:.2f}%")

# Step 6: Save the final model (last trained version, optional)
model.save("real_fake_classifier_new.h5")
print("✅ Final model saved as real_fake_classifier_new.h5")
