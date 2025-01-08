import tensorflow as tf

model = tf.keras.models.load_model("fire_and_smoke_model.keras")

import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# Load and preprocess the image
img_path = "smoke-background.webp"  # Change this to your image path
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))  # Resize to match model input size
img = img / 255.0  # Normalize pixel values
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make a prediction
prediction = model.predict(img)

# Print result
print("Prediction:", prediction)

if prediction[0][0] > 0.5:
    print("🔥 Fire Detected!")
else:
    print("✅ No Fire Detected!")

class_names = ["No Fire", "Smoke", "Fire"]  # Update based on your classes
predicted_class = np.argmax(prediction)
print(f"Predicted Class: {class_names[predicted_class]}")
