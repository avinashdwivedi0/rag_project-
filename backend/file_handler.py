# backend/file_handler.py

import os
import hashlib
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

from backend.config import UPLOAD_DIR, VECTOR_DIR


def get_file_hash(file_list: List) -> str:
    """Generate a hash for the uploaded files."""
    hash_md5 = hashlib.md5()
    for file in file_list:
        content = file.read()
        file.seek(0)
        hash_md5.update(content)
    return hash_md5.hexdigest()


def save_uploaded_files(file_list: List, file_hash: str) -> List[str]:
    """Save uploaded files and return saved paths."""
    save_dir = os.path.join(UPLOAD_DIR, file_hash)
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = []
    for file in file_list:
        file_path = os.path.join(save_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        file.seek(0)
        saved_paths.append(file_path)
    return saved_paths


def load_documents_from_paths(paths: List[str]):
    """Load documents from file paths."""
    documents = []
    for path in paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)
        documents.extend(loader.load())
    return documents


def split_documents(documents):
    """Split long documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def embed_documents(splits, save_path: str):
    """Embed document splits and save the FAISS vectorstore."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore


def load_or_create_vectorstore(file_list: List):
    """Load FAISS vectorstore if exists, else create and save it."""
    file_hash = get_file_hash(file_list)
    index_path = os.path.join(VECTOR_DIR, file_hash)

    if os.path.exists(index_path):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        saved_paths = save_uploaded_files(file_list, file_hash)
        raw_docs = load_documents_from_paths(saved_paths)
        splits = split_documents(raw_docs)
        vectorstore = embed_documents(splits, index_path)

    # Fix: specify output_key="answer" to avoid ValueError when multiple outputs are returned
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return vectorstore, memory
