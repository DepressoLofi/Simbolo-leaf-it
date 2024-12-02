import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
from PIL import Image  

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
)


model = tf.keras.models.load_model("plantdisease_model.h5")

# TODO:: to add Json file path
class_dictionary = {0: 'Banana', 1: 'Coconut', 2: 'Corn', 3: 'Watermelon'}  

# the function
def predict(img):
    # Preprocess the image
    img_array = image.img_to_array(img)  
    img_array = img_array / 255.0  
    final_image = np.expand_dims(img_array, axis=0)  

    # Predict with the ResNet50 model
    predictions = model.predict(final_image)
    predicted_class_idx = np.argmax(predictions[0]) 
    confidence = np.max(predictions[0])  

    # Map the predicted index to the class label
    predicted_class = class_dictionary[predicted_class_idx]
    return predicted_class, confidence


# Streamlit App UI
st.title("Plant Disease Detection 🌿")
uploaded_file = st.file_uploader("Upload your plant image here", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize and predict
    img = img.resize((224, 224))  # ResNet50 expects 224x224 input
    class_name, confidence = predict(img)

    # Display results
    st.write(f"**Predicted Class:** {class_name}")
    st.write(f"**Confidence:** {confidence:.2f}")



