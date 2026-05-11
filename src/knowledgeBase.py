"""
knowledgeBase.py

Builds TWO separate ChromaDB databases:
  1. brane_lang_db   — language spec + documentation (manual, specification)
  2. brane_pkg_db    — packages and datasets (context-relevant)

Run this once, or whenever your docs change:
    python knowledgeBase.py
"""

import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANUAL_PATH      = "../submodules/manual"
SPEC_PATH        = "../submodules/specification"
PACKAGES_PATH    = "../submodules/packages" 
DATASETS_PATH    = "../submodules/datasets"  

LANG_DB_PATH     = "../brane_lang_db"   
PKG_DB_PATH      = "../brane_pkg_db"    

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Syntax signal filter
# Only keep chunks that contain actual BraneScript syntax or meaningful prose.
# This drops changelogs, license files, pure nav pages, etc.
# ---------------------------------------------------------------------------
SYNTAX_SIGNALS = [
    "func ", "let ", ":=", "import", "workflow", "->", "package",
    "//", "return", "if ", "else", "while", "for ", "class ", "new ",
    "println", "commit_result", "data ", "unit",
]

def is_useful_chunk(chunk: Document) -> bool:
    text = chunk.page_content
    if len(text) < 80:          # too short to be meaningful
        return False
    if len(text.splitlines()) == 1 and len(text) < 120:
        return False            # single-line headings, TOC entries, etc.
    return any(sig in text for sig in SYNTAX_SIGNALS) or len(text) > 250


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_md_files(paths: list[str]) -> list[Document]:
    docs = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  ⚠️  Path not found, skipping: {path}")
            continue
        loader = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8"},
                                 silent_errors=True)
        loaded = loader.load()
        print(f"  📁 {path}: {len(loaded)} files")
        docs.extend(loaded)
    return docs


def chunk_and_filter(docs: list[Document],
                     chunk_size: int = 400,
                     chunk_overlap: int = 50,
                     apply_filter: bool = True) -> list[Document]:
    splitter = MarkdownTextSplitter(chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    if apply_filter:
        before = len(chunks)
        chunks = [c for c in chunks if is_useful_chunk(c)]
        print(f"  ✂️  {before} chunks → {len(chunks)} after filtering")
    else:
        print(f"  ✂️  {len(chunks)} chunks")
    return chunks


def build_db(chunks: list[Document], path: str,
             embeddings: HuggingFaceEmbeddings) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    Chroma.from_documents(documents=chunks, embedding=embeddings,
                          persist_directory=path)
    print(f"  ✨ Saved to '{path}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # --- 1. Language spec DB (manual + specification) ---
    print("\n📚 Building language specification DB...")
    lang_docs = load_md_files([MANUAL_PATH, SPEC_PATH])
    print(f"  ✅ Loaded {len(lang_docs)} files total")
    lang_chunks = chunk_and_filter(lang_docs, chunk_size=400,
                                   chunk_overlap=50, apply_filter=True)
    build_db(lang_chunks, LANG_DB_PATH, embeddings)

    # --- 2. Package / dataset DB (context-relevant) ---
    print("\n📦 Building package/dataset DB...")
    pkg_docs = load_md_files([PACKAGES_PATH, DATASETS_PATH])
    if pkg_docs:
        print(f"  ✅ Loaded {len(pkg_docs)} files total")
        # Don't filter pkg docs — every line may matter (function signatures, types)
        pkg_chunks = chunk_and_filter(pkg_docs, chunk_size=300,
                                      chunk_overlap=30, apply_filter=False)
        build_db(pkg_chunks, PKG_DB_PATH, embeddings)
    else:
        print("  ℹ️  No package/dataset docs found — pkg DB skipped.")
        print("      Populate ../submodules/packages and ../submodules/datasets")
        print("      with .md files describing your packages and datasets,")
        print("      then re-run this script.")

    print("\n🎉 Done. DBs ready.")
    print(f"   Lang spec : {LANG_DB_PATH}")
    print(f"   Pkg/data  : {PKG_DB_PATH}")


if __name__ == "__main__":
    build_knowledge_base()
