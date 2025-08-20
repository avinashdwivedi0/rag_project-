# backend/rag_chain.py
import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def create_chat_chain(vectorstore, memory):
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        api_key=GROQ_API_KEY  # Pass API key here
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return qa_chain
