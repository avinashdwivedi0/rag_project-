# Persistent RAG Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) assistant that lets you upload documents once and query them persistently with conversational memory. Powered by LangChain, Groq LLM, and FAISS vector store.

---

## Features

- Upload `.txt` or `.pdf` documents once and reuse them later
- Vector embeddings and similarity search with FAISS
- Conversational memory to keep context across queries
- Integrated with Groq LLM (`llama3-8b-8192`) for responses
- Streamlit UI with chat interface and source document references

---

## Getting Started

### Prerequisites

- Python 3.8+
- A Groq API key (sign up at [Groq](https://www.groq.com))

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rag-assistant.git
   cd rag-assistant

Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows


venv\Scripts\activate      # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Create a .env file in the project root with your Groq API key:

env
Copy code
GROQ_API_KEY=your_actual_groq_api_key_here
Running the App
bash
Copy code
streamlit run main.py
Open your browser at http://localhost:8501 to use the app.

Basic Usage Example
Upload your .txt or .pdf files via the UI, then ask questions about their content in the chat box:

python
Copy code
query = "What is the main point in the uploaded document?"
result = qa_chain({"question": query})
print("Answer:", result["answer"])

# Optional: Access source documents
for i, doc in enumerate(result["source_documents"]):
    print(f"Source {i+1}: {doc.page_content[:500]}...")
Project Structure
bash
Copy code
rag_project/
├── backend/
│   ├── rag_chain.py          # Defines LLM + retriever chain
│   ├── file_handler.py       # Handles file upload, vectorstore creation/loading
│   ├── config.py             # Directory paths and config
├── frontend/
│   └── ui.py                 # Streamlit UI code and interaction logic
├── main.py                   # Entry point to launch Streamlit app
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables (API keys, not committed)
