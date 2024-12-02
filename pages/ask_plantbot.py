import streamlit as st
import os
from groq import Groq 
import random
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


st.title("simple chat")

load_dotenv()

# information of groq chat
groq_api_key = os.environ['GROQ_API_KEY']
model = 'Mixtral-8x7b-32768'
# model = 'llama2-70b-4096'


conversational_memory_length = 10
memory = ConversationBufferMemory(k=conversational_memory_length)


groq_chat = ChatGroq(
        groq_api_key = groq_api_key,
        model_name = model,
    )

conversation = ConversationChain(
        llm = groq_chat,
        memory = memory
    )



if "messages" not in st.session_state:
    st.session_state.messages = []



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    response = conversation.run(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

