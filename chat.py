import os
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
# model_chatbot_name = 'Mixtral-8x7b-32768'
model_chatbot_name = 'llama3-8b-8192'
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
        model_name = model_chatbot_name,
    )

conversation = ConversationChain(
        llm = groq_chat,
        memory = memory,
        prompt=plant_prompt_template
    )

# Chatbot configuration
def initialize_chatbot():
    groq_api_key = os.environ['GROQ_API_KEY']
    model_chatbot_name = 'llama3-8b-8192'

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

    memory = ConversationBufferMemory(k=10)
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_chatbot_name,
    )

    return ConversationChain(llm=groq_chat, memory=memory, prompt=plant_prompt_template)


conversational_memory_length = 10
memory = ConversationBufferMemory(k=conversational_memory_length)
