import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

REPOS = ["../submodules/manual", "../submodules/specification"]
CHROMA_PATH = "../brane_knowledge_db"

def build_knowledge_base():
    # 1. Load Documents using TextLoader (cleaner for raw MD than Unstructured)
    print("📂 Loading documents...")
    all_docs = []
    for repo_path in REPOS:
        # Use TextLoader for raw markdown to preserve syntax characters
        loader = DirectoryLoader(repo_path, glob="**/*.md", loader_cls=TextLoader)
        all_docs.extend(loader.load())

    print(f"✅ Loaded {len(all_docs)} files.")

    # 2. Use a specialized Markdown Splitter
    # This prevents code blocks from being chopped up randomly
    text_splitter = MarkdownTextSplitter(
        chunk_size=600, 
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(all_docs)
    print(f"✂️  Split into {len(chunks)} chunks.")

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Store (Wipe old DB first to avoid duplicates)
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"✨ Knowledge base created at '{CHROMA_PATH}'")

if __name__ == "__main__":
    build_knowledge_base()