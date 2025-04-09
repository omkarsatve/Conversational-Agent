from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langgraph.graph import StateGraph
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage
from langchain_core.messages import HumanMessage

from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
import os

google_api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",  # or gemini-pro if you want
    google_api_key=google_api_key,
    temperature=0.3,
    top_k=3
)

embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_api_key,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vectorstore_path = os.path.join(os.getcwd(), "vectorstores", "dxfactor")
vectorstore_path = r"C:\Users\omkar\Downloads\Omkar_project_files\Omkar_project_files\vectorstores\dxfactor"
# vectorstore_path = os.path.join(r"C:\\Users\\omkar\\Downloads\\Omkar_project_files\\Omkar_project_files", "vectorstores")

vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)

# Define the structure of the state
class ChatState(TypedDict):
    question: str
    answer: Optional[str]
    docs: Optional[List[Document]]
    first_turn: bool
    chat_history: List  # To hold past conversation turns

retriever = vectorstore.as_retriever()
output_parser = StrOutputParser()

# System prompt
system_prompt = SystemMessage(
    content=(
        "You are a helpful AI assistant for the DXFactor website. "
        "Answer questions based on the content scraped from the site. "
        "You are having a multi-turn conversation, so remember the previous questions and answers. "
        "If the user asks about something from earlier, use the chat history to respond appropriately. "
        "Keep your tone professional and helpful."
    )
)

# LangGraph
graph = StateGraph(ChatState)

# Node: Get user input
def get_user_input(state):
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Exiting chatbot. Have a great day!")
        raise SystemExit  # This prevents recursion loop
    state["question"] = user_input
    return state

# Node: Retrieve documents
def retrieve_docs(state):
    state["docs"] = retriever.get_relevant_documents(state["question"])
    return state

def generate_answer(state):
    question = state["question"]
    docs = state.get("docs", [])

    # Build the context from documents
    context = "\n\n".join(doc.page_content for doc in docs[:5]) if docs else "No relevant documents found."

    # Greeting logic
    if state["first_turn"]:
        greeting = "Hi! Welcome to DXFactor — how can I help you today?\n\n"
        state["first_turn"] = False
    else:
        greeting = ""

    # Prepare the message for this turn
    user_message = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")

    # Initialize chat_history if not present
    if "chat_history" not in state or state["chat_history"] is None:
        state["chat_history"] = []

    # Combine full message sequence
    messages = [system_prompt] + state["chat_history"] + [user_message]

    try:
        response = llm.invoke(messages)
        answer = output_parser.invoke(response).strip()
        if not answer:
            answer = "I'm sorry, I couldn't find a relevant answer based on the website data."
    except Exception as e:
        answer = f"An error occurred: {str(e)}"

    # Full response to display
    full_response = greeting + answer
    print("\n🤖 Gemini:")
    print(full_response)

    # Update chat history — must be done *before* next round
    state["chat_history"].extend([user_message, response])
    state["answer"] = answer
    return state

# Register nodes and flow
graph.add_node("Input", get_user_input)
graph.add_node("Retrieve", retrieve_docs)
graph.add_node("Answer", generate_answer)

graph.set_entry_point("Input")
graph.add_edge("Input", "Retrieve")
graph.add_edge("Retrieve", "Answer")
graph.add_edge("Answer", "Input")

# Compile chatbot
chatbot = graph.compile()

state = {"question": "", "answer": None, "docs": None, "first_turn": True, "chat_history": []}

while True:
    try:
        chatbot.invoke(state)
    except SystemExit:
        break

