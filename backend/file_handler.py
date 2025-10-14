import os
import hashlib
import logging
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

from backend.config import UPLOAD_DIR, VECTOR_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_file_hash(file_list: List) -> str:
    hash_md5 = hashlib.md5()
    try:
        for file in file_list:
            content = file.read()
            file.seek(0)
            hash_md5.update(content)
    except Exception as e:
        logger.error(f"Error hashing files: {e}")
        raise
    return hash_md5.hexdigest()


def save_uploaded_files(file_list: List, file_hash: str) -> List[str]:
    save_dir = os.path.join(UPLOAD_DIR, file_hash)
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    try:
        for file in file_list:
            file_path = os.path.join(save_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            file.seek(0)
            saved_paths.append(file_path)
        logger.info(f"Saved {len(saved_paths)} files to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving files: {e}")
        raise
    return saved_paths


def load_documents_from_paths(paths: List[str]):
    documents = []
    for path in paths:
        try:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            elif path.endswith(".csv"):
                loader = CSVLoader(path)
            else:
                loader = TextLoader(path)
            documents.extend(loader.load())
            logger.info(f"Loaded documents from {path}")
        except Exception as e:
            logger.error(f"Error loading document {path}: {e}")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    try:
        return splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise


def embed_documents(splits, save_path: str):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(save_path)
        logger.info(f"Saved FAISS index at {save_path}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise


def load_or_create_vectorstore(file_list: List) -> Tuple[FAISS, ConversationBufferMemory]:
    try:
        file_hash = get_file_hash(file_list)
        index_path = os.path.join(VECTOR_DIR, file_hash)

        if os.path.exists(index_path):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded existing FAISS index for hash {file_hash}")
        else:
            saved_paths = save_uploaded_files(file_list, file_hash)
            raw_docs = load_documents_from_paths(saved_paths)
            splits = split_documents(raw_docs)
            vectorstore = embed_documents(splits, index_path)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        return vectorstore, memory
    except Exception as e:
        logger.error(f"Failed to load or create vectorstore: {e}")
        raise
