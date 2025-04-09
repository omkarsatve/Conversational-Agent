from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()

# Access your variables
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

# One folder up from the current directory
SAVE_DIR = os.path.abspath(os.path.join(os.getcwd(), "../vectorstores"))
os.makedirs(SAVE_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(SAVE_DIR, "dxfactor")

# Load your file
with open("../data/dxfactor_full_scrape.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

chunks = splitter.split_text(full_text)
print(f"Total chunks: {len(chunks)}")

embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_api_key,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed your chunks
embeddings = embedding_model.embed_documents(chunks)

print(f"Embeddings shape: {len(embeddings)} vectors of length {len(embeddings[0])}")

# Create vector store
vectorstore = FAISS.from_texts(chunks, embedding_model)
# Save to local disk
vectorstore.save_local(OUTPUT_FILE)
