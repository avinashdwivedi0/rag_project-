import os
import shutil
import logging
import streamlit as st
from backend.file_handler import load_or_create_vectorstore
from backend.rag_chain import create_chat_chain
from backend.config import ensure_dirs, UPLOAD_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_app():
    ensure_dirs()
    st.set_page_config(page_title="🧠 RAG Assistant", layout="wide")
    st.title("📂 Persistent RAG Assistant")
    st.markdown("Upload documents once — reuse them later with memory.")

    if "memory" not in st.session_state:
        st.session_state.memory = None  # Set in backend when vectorstore is created

    uploaded_files = st.file_uploader(
        "📤 Upload .txt, .pdf, .docx, .csv files",
        type=["txt", "pdf", "docx", "csv"],
        accept_multiple_files=True,
    )

    # Document management section 
    st.subheader("Manage Documents")
    if st.button("Clear Uploaded Documents and Indexes"):
        st.info("Clearing uploaded documents and indexes...")
        try:
            if st.session_state.memory:
                mem_key = st.session_state.memory.memory_key
                dir_to_clear = os.path.join(UPLOAD_DIR, mem_key)
                if os.path.exists(dir_to_clear):
                    shutil.rmtree(dir_to_clear)
                    st.success("Cleared uploaded documents and indexes.")
                st.session_state.memory = None
            else:
                st.info("No documents to clear.")
        except Exception as e:
            st.error(f"Failed to clear documents: {e}")
            logger.error(f"Error clearing documents: {e}")

    if uploaded_files:
        try:
            st.info("Processing uploaded files. This may take a moment...")
            vectorstore, memory = load_or_create_vectorstore(uploaded_files)
            st.session_state.memory = memory
            qa_chain = create_chat_chain(vectorstore, memory)
        except Exception as e:
            logger.error(f"Error during vectorstore creation or chain setup: {e}")
            st.error(f"Failed to process files, please try again. Error: {str(e)}")
            return

        # Sidebar query options
        with st.sidebar:
            summarize = st.checkbox("Summarize responses")

        query = st.chat_input("💬 Ask something about your documents...")
        if query:
            try:
                with st.spinner("Thinking..."):
                    result = qa_chain({"question": query})
                st.chat_message("user").write(query)
                answer = result.get("answer", "No answer returned")
                if summarize:
                    # Example placeholder for summary, implement actual summarization logic if desired
                    answer = answer + "\n\n_(This answer is summarized)_"
                st.chat_message("assistant").write(answer)

                if "source_documents" in result:
                    with st.expander("📄 Source documents"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Source {i+1}:**")
                            content = doc.page_content
                            st.code(content[:500] + ("..." if len(content) > 500 else "..."))
                else:
                    st.info("ℹ️ No source documents returned for this response.")
            except Exception as e:
                logger.error(f"Chat query execution error: {e}")
                st.error(f"Failed to get response, please try again. Error: {str(e)}")

        # Export chat history
        if st.session_state.memory and hasattr(st.session_state.memory, 'chat_memory'):
            chat_messages = st.session_state.memory.chat_memory.messages
            history = "\n".join(
                [f"{msg.type.capitalize()}: {msg.content}" for msg in chat_messages]
            )
            st.download_button(
                label="📥 Download Chat History",
                data=history,
                file_name="chat_history.txt",
                mime="text/plain",
            )
        else:
            st.info("No chat history available.")

        # Feedback collection
        feedback = st.radio("Was this answer helpful?", ("Yes", "No"))
        if st.button("Submit Feedback"):
            st.success("Thanks for your feedback!")
            logger.info(f"User feedback: {feedback}")
    else:
        st.info("⬆️ Upload documents to get started.")
