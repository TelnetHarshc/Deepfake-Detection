import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("real_fake_classifier_new.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    prediction = model.predict(img_array)[0][0]
    return "Real" if prediction > 0.5 else "Fake"

# Test on an image
img_path = "Test/Fake Images/fake_22.jpg"
print(f"Prediction: {predict_image(img_path)}")
