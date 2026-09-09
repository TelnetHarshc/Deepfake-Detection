# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow import keras
# from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc

# # Load model
# model = keras.models.load_model("best_model.h5")

# # Load test dataset
# from dataset import load_datasets
# _, _, test_ds = load_datasets()

# # Extract true labels and predictions
# y_true = []
# y_pred_prob = []

# for images, labels in test_ds:
#     y_true.extend(labels) # Extract true labels
#     y_pred_prob.extend(model.predict(images).flatten())  # Get predicted probabilities

# # Convert lists to numpy arrays
# y_true = np.array(y_true)
# y_pred_prob = np.array(y_pred_prob)
# y_pred = (y_pred_prob > 0.5).astype("int")  # Convert probabilities to binary predictions

# # Compute confusion matrix
# cm = confusion_matrix(y_true, y_pred)

# # Plot confusion matrix
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# # Classification Report
# from sklearn.metrics import precision_score, recall_score, f1_score

# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)

# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print("\nClassification Report:\n", classification_report(y_true, y_pred))

# # Plot Precision-Recall Curve
# precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)
# plt.figure(figsize=(6, 5))
# plt.plot(recall_vals, precision_vals, marker='.', label="Precision-Recall Curve")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# plt.grid()
# plt.show()

# # Plot ROC Curve
# fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.4f})")
# plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.grid()
# plt.show()



# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow import keras
# from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score

# # Load model
# model = keras.models.load_model("best_model.h5")

# # Load test dataset
# from dataset import load_datasets
# _, _, test_ds = load_datasets()

# # ✅ Optional: Speed up for debugging
# # test_ds = test_ds.take(10)

# # Extract true labels and predictions
# y_true = []
# y_pred_prob = []

# print("[INFO] Starting evaluation...")
# for i, (images, labels) in enumerate(test_ds):
#     print(f"Processing batch {i + 1}...")
#     # Convert to numpy if EagerTensor
#     if tf.is_tensor(labels):
#         labels = labels.numpy()
#     y_true.extend(labels)
#     y_pred_prob.extend(model.predict(images, verbose=0).flatten())

# # Convert to arrays
# y_true = np.array(y_true)
# y_pred_prob = np.array(y_pred_prob)
# y_pred = (y_pred_prob > 0.5).astype("int")

# # Confusion Matrix
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()

# # Precision, Recall, F1 Score
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)

# print(f"\nPrecision: {precision:.4f}")
# print(f"Recall:    {recall:.4f}")
# print(f"F1 Score:  {f1:.4f}")
# print("\nClassification Report:\n", classification_report(y_true, y_pred))

# # Precision-Recall Curve
# precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)
# plt.figure(figsize=(6, 5))
# plt.plot(recall_vals, precision_vals, marker='.', label="Precision-Recall")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# # ROC Curve
# fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color='blue')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()




# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from dataset import load_datasets

# # Step 1: Load datasets (same paths you used before)
# train_ds, val_ds, test_ds = load_datasets()

# # Step 2: Load your best trained model
# model = load_model("best_model.h5")

# # Step 3: Evaluate on test data
# test_loss, test_acc = model.evaluate(test_ds)
# print(f"\n🎯 Test Accuracy: {test_acc * 100:.2f}%")


from dataset import load_datasets
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load data
_, _, test_ds = load_datasets()

# Step 2: Load model
model = load_model("best_model.h5")

# Step 3: Predict all test data
y_prob = model.predict(test_ds, verbose=1)
y_pred = (y_prob > 0.5).astype(int).flatten()

# Step 4: Get true labels
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])

# Step 5: Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

# Step 6: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# Step 7: ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Step 8: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# Step 9: F1 Score
f1 = f1_score(y_true, y_pred)
print(f"\n🎯 F1 Score: {f1:.4f}")
