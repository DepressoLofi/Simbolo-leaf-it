import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the model and class names
model = tf.keras.models.load_model("plantdisease_model.h5")

with open("class_names.json", "r") as file:
    class_names = json.load(file)

def model_predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    final_image = np.expand_dims(img_array, axis=0)

    predictions = model.predict(final_image)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    predicted_class = class_names[predicted_class_idx]
    return predicted_class, confidence
