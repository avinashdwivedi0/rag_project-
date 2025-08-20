# backend/rag_chain.py
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

def create_chat_chain(vectorstore, memory):
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # ✅ This fixes the ValueError
    )
    return qa_chain
