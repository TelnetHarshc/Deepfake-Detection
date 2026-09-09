import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

# Load model
model = keras.models.load_model("real_fake_classifier_new.h5")

# Load test dataset
from dataset import load_datasets
_, _, test_ds = load_datasets()

# Extract true labels and predictions
y_true = []
y_pred_prob = []

for images, labels in test_ds:
    y_true.extend(labels.numpy())  # Extract true labels
    y_pred_prob.extend(model.predict(images).flatten())  # Get predicted probabilities

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = (np.array(y_pred_prob) > 0.5).astype("int")  # Convert probabilities to binary

# Get classification report as dictionary
report = classification_report(y_true, y_pred, output_dict=True)

# Convert report to DataFrame for tabular format
df = pd.DataFrame(report).transpose()

# Save to CSV for reference
df.to_csv("classification_metrics.csv")

# Print the table
print(df)

