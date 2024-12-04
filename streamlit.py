import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import pathlib
from PIL import Image # type: ignore

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

# Chat UI
if selected == 'Chat':
    with st.container():
        st_lottie(chat_animation, height=200)
        st.write("""
        ### What can I help with?
        """)

        # Initialize the session state to hold messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display the chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

        # Input for new user message
        user_input = st.chat_input("Type your message...")
        if user_input:
            # Append user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Generate a simple bot response
            bot_response = f"Echo: {user_input}"  # Replace this with your AI model's response
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

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
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Perform object detection
                with st.spinner('Detecting objects...'):
                    # Load YOLOv5 model (ensure model is downloaded; alternatively, use a smaller model or a different one)
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

                    # Perform inference
                    results = model(image)

                    # Render results on the image
                    annotated_image = results.render()[0]
                    annotated_image = Image.fromarray(annotated_image)

                    # Display annotated image
                    st.image(annotated_image, caption='Detected Objects', use_column_width=True)

                    # Display detection results
                    st.write("### Detection Results")
                    detection_df = results.pandas().xyxy[0]
                    st.dataframe(detection_df)
            else:
                st.warning("Please upload an image file.")

        st.write("---")
        st.markdown("**Note**: Object detection is performed using a pre-trained YOLOv5 model. For custom models or advanced configurations, further development is required.")

# Optional: Footer
st.markdown("<hr><p style='text-align: center; color: '>© 2024 Leaf-It by Simbolo's Farmers.</p>", unsafe_allow_html=True)