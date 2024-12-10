import requests
import streamlit as st

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# def load_lottie(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

def fetch_lottie_animation(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

plant_animation = fetch_lottie_animation("https://lottie.host/60ce0978-f932-491c-8e0f-4ddaee3951f6/NwZNzCET2n.json")
image_detection_animation = fetch_lottie_animation("https://lottie.host/e77ad952-c631-4b87-a7e5-14358de78320/U0AfD37neX.json")
chat_animation = fetch_lottie_animation("https://lottie.host/e66fbbc2-b561-41cd-a6dd-f544a29bec7b/A4tqdvyArS.json")