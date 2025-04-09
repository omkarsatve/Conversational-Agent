import streamlit as st
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain.schema import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Load API Keys
google_api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Load LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=google_api_key,
    temperature=0.3,
    top_k=3
)

# Load Embeddings
embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_api_key,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Vectorstore
vectorstore_path = r"C:\Users\omkar\Downloads\Omkar_project_files\Omkar_project_files\vectorstores\dxfactor"
vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
output_parser = StrOutputParser()

# System Prompt
system_prompt = SystemMessage(
    content=(
        "You are a helpful AI assistant for the DXFactor website. "
        "Answer questions based on the content scraped from the site. "
        "You are having a multi-turn conversation, so remember the previous questions and answers. "
        "Keep your tone professional and helpful."
    )
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "first_turn" not in st.session_state:
    st.session_state.first_turn = True

# Streamlit UI
st.set_page_config(page_title="DXFactor Chatbot", page_icon="🤖")
st.title("🤖 DXFactor Chatbot")
st.markdown("Ask me anything about the DXFactor!")

# Chat UI
user_input = st.chat_input("Type your message here...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant docs
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join(doc.page_content for doc in docs[:5]) if docs else "No relevant documents found."

    # Prepare message sequence
    user_message = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {user_input}")
    messages = [system_prompt] + st.session_state.chat_history + [user_message]

    try:
        response = llm.invoke(messages)
        answer = output_parser.invoke(response).strip()
        if not answer:
            answer = "I'm sorry, I couldn't find a relevant answer based on the website data."
    except Exception as e:
        answer = f"An error occurred: {str(e)}"

    # Greet on first turn
    if st.session_state.first_turn:
        answer = "Hi! Welcome to DXFactor — how can I help you today?\n\n" + answer
        st.session_state.first_turn = False

    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Update history
    st.session_state.chat_history.extend([user_message, response])

# Optional: Button to clear chat
if st.button("🔄 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.first_turn = True
    st.rerun()
