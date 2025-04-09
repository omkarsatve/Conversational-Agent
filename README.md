# Conversational-Agent/Chatbot for DXFactor  🤖

A multi-turn conversational AI chatbot designed to answer questions based on content scraped from the DXFactor website. This project leverages `LangChain`, `LangGraph`, `FAISS`, and `Gemini (Google Generative AI)` to create an intelligent assistant capable of understanding and retrieving relevant responses from site content.

---

## 🧩 Project Structure

### 1. `01_scrape.py`  
> **Purpose:**  
Scrapes textual data from the DXFactor website and saves the content locally. This includes headings, paragraphs, and structured text from different web pages.

---

### 2. `02_create_embeddings.py`  
> **Purpose:**  
Processes the scraped data and generates vector embeddings using `HuggingFace Inference API` (`all-MiniLM-L6-v2` model).  
The resulting embeddings are stored locally in a FAISS vectorstore for efficient semantic search.

---

### 3. `chatbot.py`  
> **Purpose:**  
Creates an interactive chatbot using:
- **LangGraph** for managing multi-turn conversation flow
- **LangChain** for embedding-based retrieval (`FAISS`)
- **Google's Gemini** for LLM-powered answer generation

> **Features:**
- Remembers previous turns and uses chat history
- Retrieves relevant content chunks for each query
- Generates context-aware and professional responses
- Easy to extend for web interfaces using Streamlit or Gradio

---

## ⚙️ Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt


You’ll also need to set up your .env file with the following keys:
GOOGLE_API_KEY=your_google_genai_api_key
HUGGINGFACE_API_KEY=your_huggingface_inference_api_key

For Running the Chatbot
python chatbot.py

Directory Structure
Omkar_project_files/
├── 01_scrape.py
├── 02_create_embeddings.py
├── chatbot.py
├── vectorstores/
│   └── dxfactor/
│       ├── index.faiss
│       └── index.pkl
├── .env
└── README.md
