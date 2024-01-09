import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from dotenv import load_dotenv
import os

"""
# Welcome to Alpha Test Coaching Bot
This GPT-powered chatbot can retrieve test questions and converse with the students to review their mistakes in the test.
Demo Version: Currently only support Alpha ELA STAAR G5.2017 test
"""
if "openai_api_key" not in st.session_state.keys(): # Prompt user to enter their key
    st.session_state.openai_api_key = st.text_input("Please enter your OpenAI API Key:")

if not openai_api_key:
        st.warning("Please add your OpenAI API key to continue.")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Let's begin! Which question do you want to discuss?"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the data – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, 
                                                                  system_prompt=
                                                                  "You are an expert in providing coaching for standardized test review. Your job is to converse with the student and identify if their mistakes are due to knowledge gaps/holes or test-taking anti-patterns."
                                                                  "You have access to all passages, questions, and answer keys from the documents in the data folder."
                                                                  "You are to provide useful coaching according to the recommended best practice for each anti-pattern according to the 'Test-Taking Anti-Pattern' document."
                                                                  "Ignore and redirect responses that are not related to the standardized test coaching. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Question Number:"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

