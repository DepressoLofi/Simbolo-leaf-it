import streamlit as st
from chat import initialize_chatbot
from image_detection import model_predict
from utils import load_css, plant_animation, image_detection_animation, chat_animation
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from PIL import Image
from chat import conversation



# Load external CSS
load_css("style/styles.css")

# Hide streamlit style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            fotter {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

chatbot = initialize_chatbot()


# Option menu for navigation
if "selected_button" in st.session_state:
    st.session_state.selected_menu = st.session_state.selected_button
    del st.session_state.selected_button

if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "Home"

selected = option_menu(
        menu_title= None,
        options=['Home', 'Chat', 'Image Detection'],
        icons=['house', 'chat', 'camera'],
        key='selected_menu',
        default_index=["Home", "Chat", "Image Detection"].index(st.session_state['selected_menu']),
        orientation= 'horizontal',
)

if selected == "Home":
    if plant_animation:
        st_lottie(plant_animation, height=200)
    else:
        st.error("Failed to load Lottie animation.")

    st.title("Welcome to Leaf-It")
    st.write("Explore the features of this app.")

# Chat bot

elif selected == 'Chat':
    with st.container():
        if chat_animation:
            st_lottie(chat_animation, height=200)
        else:
            st.error("Failed to load Lottie animation.")

        st.title("Chat with the Plant Expert")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_placeholder = st.container()

        with chat_placeholder:
            for message in st.session_state.messages:
                avatar_emoji = "🌿" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar_emoji):
                    st.markdown(message["content"])

        prompt = st.chat_input("How can I help with plants?")

        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with chat_placeholder:
                with st.chat_message("user", avatar="🌿"):
                    st.markdown(prompt)

                response = conversation.run(prompt)

                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response,})

# Image Detection
elif selected == 'Image Detection':
    with st.container():
        if image_detection_animation:
            st_lottie(image_detection_animation, height=200)
        else:
            st.error("Failed to load Lottie animation.")

        st.header("Image Detection")
        if st.button("Go to Chat"):
            st.session_state.selected_button = "Chat"
            st.rerun()
        class_name = None
        # Create a form for image upload and detection
        with st.form(key='image_detection_form'):
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "jfif"])
            submit_button = st.form_submit_button(label='Detect Objects')

        if submit_button:
            if uploaded_file is not None:
                # Display the uploaded image
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image')

                # Perform object detection
                with st.spinner('Detecting objects...'):
                    class_name, confidence = model_predict(img)
                    st.header(f"**Predicted Class:** {class_name}")
                   
                    #    st.write(f"**Confidence:** {confidence:.2f}")
               
            else:
                st.warning("Please upload an image file. Click the Browse files button!")

        if (class_name):
            if st.button("Go to Chat Pls"):
                st.session_state.selected_button = "Chat"
                st.rerun()