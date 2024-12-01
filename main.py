import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils

model = tf.keras.applications.mobilenet.MobileNet()


# the function
def predict(img):
    #preprocessing the image
    resized_img = image.img_to_array(img)
    final_image = np.expand_dims(resized_img, axis=0)
    final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)
    #predict the image
    predictions = model.predict(final_image)
    results = imagenet_utils.decode_predictions(predictions)
    #getting answer
    results = results[0][0][1]
    return results


st.title("Plant Disease Detection")
uploaded_file = st.file_uploader("Upload your pic here", type=['png', 'jpg'])




if uploaded_file is not None:
    st.image(uploaded_file)
    img = image.load_img(uploaded_file , target_size = (224, 224))
    prediction = predict(img)
    # prediction = "Apple"
    st.text(prediction)

    

    