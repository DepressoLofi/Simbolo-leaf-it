import streamlit as st
from chat import initialize_chatbot
from image_detection import model_predict
from utils import load_css, plant_animation, image_detection_animation, chat_animation
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from PIL import Image
from chat import conversation

st.set_page_config(
    page_title="Leaf-it",  
    page_icon="🌱",           
)

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

# for chat
chatbot = initialize_chatbot()

if 'class_name' not in st.session_state:
    st.session_state.class_name = None

if 'topic_ask' not in st.session_state:
    st.session_state.topic_ask = None


# Option menu for navigation
if st.session_state.get('switch_button', False):
    st.session_state.topic_ask = st.session_state.class_name
    st.session_state['menu_option'] = 1
    manual_select = st.session_state['menu_option']
else:
    manual_select = None
    
selected = option_menu(None, ["Home", "LeafChat", "Disease Detection"], 
    icons=['house', 'chat', "camera"], 
    orientation="horizontal", manual_select=manual_select, key='menu_4')


if selected == "Home":
    if plant_animation:
        st_lottie(plant_animation, height=200)
    else:
        st.error("Failed to load Lottie animation.")

    st.title("Welcome to Leaf-It")
    st.markdown('''
    :green[Leaf-it] is a web application that provides information about plants, their care, biology, and uses.
    It also allows users to detect plant diseases and provides information about them.
''')
    st.write('''This project initially aimmed at providing aid and knolwedge for the farmers to understand about the plants and their diseases in agriculture field.
             But we also believe that this project can also help people who grow plants as a hobby and want to give their lovely plants a better life.''')
    

# Chat bot

elif selected == 'LeafChat':
    
    with st.container():
        if chat_animation:
            st_lottie(chat_animation, height=200)
        else:
            st.error("Failed to load Lottie animation.")

        st.title("Chat with the Plant Expert")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_placeholder = st.container()

        prompt = st.chat_input("How can I help with plants?")

        with chat_placeholder:

            if st.session_state.topic_ask:
                st.session_state.messages = []
                prompt = f"How to cure {st.session_state.topic_ask}?"
                st.session_state.topic_ask = None

            for message in st.session_state.messages:
                avatar_emoji = "🌿" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar_emoji):
                    st.markdown(message["content"])

        

        

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
elif selected == 'Disease Detection':
    
    with st.container():
        

        st.header("Disease Detection")
        
        
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
                   st.header(f"{class_name}")

                #    st.write(f"**Confidence:** {confidence:.2f}")

            else:
                st.warning("Please upload an image file. Click the Browse files button!")

    if class_name:
        st.button("Ask Leaf-Chat on How to Cure", key='switch_button')
        st.session_state.class_name = class_name
