import streamlit as st
import numpy as np
import os
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import pathlib
from PIL import Image # type: ignore
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils # type: ignore
import json
from dotenv import load_dotenv
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

#
# Image Detection Codes
#

model = tf.keras.models.load_model("plantdisease_model.h5")

classes_file = 'class_names.json'
with open(classes_file, 'r') as data:
    class_names = json.load(data)


# the function
def predict(img):
    # Preprocess the image
    img = img.resize((224, 224)) 
    img_array = image.img_to_array(img)  
    img_array = img_array / 255.0  
    final_image = np.expand_dims(img_array, axis=0)  

    # Predict with the ResNet50 model
    predictions = model.predict(final_image)
    predicted_class_idx = np.argmax(predictions[0]) 
    confidence = np.max(predictions[0])  

    # Map the predicted index to the class label
    predicted_class = class_names[predicted_class_idx]
    return predicted_class, confidence



#
# Chatbot codes
#


groq_api_key = os.environ['GROQ_API_KEY']
# model = 'Mixtral-8x7b-32768'
model = 'llama3-8b-8192'
# model = 'whisper-large-v3-turbo'

plant_prompt_template = PromptTemplate(
    input_variables=["input", "history"],
    template=(
        "You are a chatbot specialized in plants. "
        "You answer questions only about plants, including their care, biology, and uses. "
        "If asked about unrelated topics, respond with: 'I only discuss plants.'\n\n"
        "Conversation history:\n{history}\n\n"
        "User: {input}\nChatbot:"
    )
)

conversational_memory_length = 10
memory = ConversationBufferMemory(k=conversational_memory_length)


groq_chat = ChatGroq(
        groq_api_key = groq_api_key,
        model_name = model,
    )

conversation = ConversationChain(
        llm = groq_chat,
        memory = memory,
        prompt=plant_prompt_template
    )








# Set the page layout
st.set_page_config(layout="wide")

# Function to load external CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the external CSS
css_path = pathlib.Path("style/styles.css")
if css_path.exists():
    load_css(css_path)

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
chat_animation = load_lottie("https://lottie.host/e66fbbc2-b561-41cd-a6dd-f544a29bec7b/A4tqdvyArS.json")
contact = load_lottie("https://lottie.host/e77ad952-c631-4b87-a7e5-14358de78320/U0AfD37neX.json")
plant_animation = load_lottie("https://lottie.host/60ce0978-f932-491c-8e0f-4ddaee3951f6/NwZNzCET2n.json")
image_detection_animation = load_lottie("https://lottie.host/e77ad952-c631-4b87-a7e5-14358de78320/U0AfD37neX.json")

# Header
with st.container():
    col1, col2 = st.columns(2)
    with col1:
      st.title("Welcome to Leaf-It")
      with col2:
        st_lottie(plant_animation, height=200)
st.write("---")

# Option menu for navigation
selected = option_menu(
    menu_title= None,
    options=['Home', 'Chat', 'Image Detection'],
    icons=['house', 'chat', 'camera'],
    orientation= 'vertical',
    styles={"columns": {"background-color": "#F0F2F6"}},
)

if selected == "Home":
    st.title("Welcome to Our Project")
    st.write("""
        This project demonstrates a multi-functional web application using **Streamlit**.
        - **Chat**: Engage in real-time text input functionality.
        - **Image Detection**: Upload an image and see our AI-powered detection in action.
    """)
    st.write("""
        Feel free to explore and interact with the features.
        Our goal is to make complex technologies accessible and user-friendly.
    """)
    st.write("---")




# Chat UI start here ------------------->
if selected == 'Chat':
    with st.container():
        st_lottie(chat_animation, height=200)
        st.write("""
        ### What can I help with?
        """)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_placeholder = st.container()

        with chat_placeholder:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        prompt = st.chat_input("What is up?")
        
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with chat_placeholder:
                with st.chat_message("user"):
                    st.markdown(prompt)
        
                response = conversation.run(prompt)
            
                with st.chat_message("assistant"):
                    st.markdown(response)
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})






# image detection start here ------------->
if selected == 'Image Detection':
    with st.container():
        st_lottie(contact, height=200)
        st.header("Image Detection")
        st.write("Detect objects in your images using our AI-powered detection tool.")

        # Create a form for image upload and detection
        with st.form(key='image_detection_form'):
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
            submit_button = st.form_submit_button(label='Detect Objects')

        if submit_button:
            if uploaded_file is not None:
                # Display the uploaded image
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image', use_column_width=True)

                # Perform object detection
                with st.spinner('Detecting objects...'):
                   class_name, confidence = predict(img)
                   st.write(f"**Predicted Class:** {class_name}")
                #    st.write(f"**Confidence:** {confidence:.2f}")

            else:
                st.warning("Please upload an image file.")

       


# Optional: Footer
st.markdown("<hr><p style='text-align: center; color: '>© 2024 Leaf-It by Simbolo's Farmers.</p>", unsafe_allow_html=True)




