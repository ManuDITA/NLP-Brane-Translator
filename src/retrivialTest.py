import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Must match the path used in knowledgeBase.py
CHROMA_PATH = "../brane_knowledge_db"

# Initialize the same embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# --- THE TEST ---
query = "Define a function or a workflow in Branescript"
print(f"🔍 Searching for: {query}\n")

# k=3 means "find the top 3 most relevant chunks"
docs = db.similarity_search(query, k=5)

for i, doc in enumerate(docs):
    print(f"--- Result {i+1} (Source: {doc.metadata.get('source', 'Unknown')}) ---")
    print(doc.page_content)
    print("-" * 50)