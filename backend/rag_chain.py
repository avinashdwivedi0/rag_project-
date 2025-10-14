# backend/rag_chain.py

import os
import logging
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def create_chat_chain(vectorstore, memory):
    try:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
            api_key=GROQ_API_KEY  # Pass API key here
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=vectorstore.as_retriever(), memory=memory
        )
        logger.info("Chat chain successfully created.")
        return qa_chain
    except Exception as e:
        logger.error(f"Failed to create chat chain: {e}")
        raise
