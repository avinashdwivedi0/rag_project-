# frontend/ui.py
import os
import streamlit as st
from backend.file_handler import load_or_create_vectorstore
from backend.rag_chain import create_chat_chain
from backend.config import ensure_dirs

def run_app():
    ensure_dirs()
    st.set_page_config(page_title="🧠 RAG Assistant", layout="wide")
    st.title("📂 Persistent RAG Assistant")
    st.markdown("Upload documents once — reuse them later with memory.")

    if "memory" not in st.session_state:
        st.session_state.memory = None  # Set in backend when vectorstore is created

    uploaded_files = st.file_uploader(
        "📤 Upload .txt or .pdf files", 
        type=["txt", "pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        vectorstore, memory = load_or_create_vectorstore(uploaded_files)
        st.session_state.memory = memory
        qa_chain = create_chat_chain(vectorstore, memory)

        query = st.chat_input("💬 Ask something about your documents...")
        if query:
            with st.spinner("Thinking..."):
                result = qa_chain({"question": query})
                st.chat_message("user").write(query)
                st.chat_message("assistant").write(result["answer"])

                # ✅ Safely handle missing 'source_documents'
                if "source_documents" in result:
                    with st.expander("📄 Source documents"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.code(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else "..."))
                else:
                    st.info("ℹ️ No source documents returned for this response.")
    else:
        st.info("⬆️ Upload documents to get started.")
