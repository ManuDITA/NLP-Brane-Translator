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
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SELECTED_DOCS_PATH = BASE_DIR / "selectedDocs"
MANUAL_PATH        = BASE_DIR / "submodules/manual"
SPEC_PATH          = BASE_DIR / "submodules/specification"
PACKAGES_PATH      = BASE_DIR / "submodules/packages"
DATASETS_PATH      = BASE_DIR / "submodules/datasets"

LANG_DB_PATH       = BASE_DIR / "brane_lang_db"
PKG_DB_PATH        = BASE_DIR / "brane_pkg_db" 

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


def chunk_packages_by_section(docs: list[Document]) -> list[Document]:
    """
    Smart chunking strategy for package documentation.
    
    Packages contain:
      - Metadata (container.yml, action definitions) → keep whole
      - Function descriptions → chunk at 1000 chars with 200 overlap
      - BraneScript examples → keep examples whole (usually < 800 chars each)
      - Configuration guides → chunk at 800 chars with 150 overlap
    
    This preserves semantic units and maintains code example integrity.
    """
    chunks = []
    
    for doc in docs:
        text = doc.page_content
        source = doc.metadata.get("source", "")
        
        # Heuristic: detect document type from filename/content
        is_config = "configuration" in source.lower() or "config" in text[:200].lower()
        is_example = "example" in source.lower() or text.count("bscript") > 2
        is_reference = "reference" in source.lower() or "quick" in source.lower()
        
        # Strategy 1: Keep container/action definitions whole
        if "container.yml" in source or "container:" in text[:100]:
            chunks.append(doc)
            continue
        
        # Strategy 2: Keep BraneScript examples whole (usually compact)
        if "```bscript" in text:
            # Split on code blocks to keep examples intact
            examples = text.split("```bscript")
            for i, example in enumerate(examples):
                if i == 0:
                    # first part before code block
                    if example.strip():
                        sub_chunks = MarkdownTextSplitter(
                            chunk_size=600, chunk_overlap=100
                        ).split_documents([Document(page_content=example, 
                                                   metadata=doc.metadata)])
                        chunks.extend(sub_chunks)
                else:
                    # keep code block + surrounding text together
                    code_section = "```bscript" + example
                    if len(code_section) < 2000:  # manageable size
                        chunks.append(Document(page_content=code_section.split("```")[0:2][0] + "```",
                                             metadata=doc.metadata))
                    else:
                        # large example, chunk it with generous overlap
                        sub_chunks = MarkdownTextSplitter(
                            chunk_size=1200, chunk_overlap=200
                        ).split_documents([Document(page_content=code_section,
                                                   metadata=doc.metadata)])
                        chunks.extend(sub_chunks)
        # Strategy 3: Configuration guides need context (larger chunks)
        elif is_config:
            sub_chunks = MarkdownTextSplitter(
                chunk_size=800, chunk_overlap=150
            ).split_documents([doc])
            chunks.extend(sub_chunks)
        # Strategy 4: Reference docs (like QUICK_REFERENCE) - preserve sections
        elif is_reference:
            # Keep "## Sections" together with their content
            sections = text.split("\n## ")
            for section in sections:
                if section.strip():
                    formatted = "## " + section if not section.startswith("#") else section
                    if len(formatted) < 1500:  # manageable section
                        chunks.append(Document(page_content=formatted,
                                             metadata=doc.metadata))
                    else:
                        # Large section, chunk with high overlap for coherence
                        sub_chunks = MarkdownTextSplitter(
                            chunk_size=1000, chunk_overlap=200
                        ).split_documents([Document(page_content=formatted,
                                                   metadata=doc.metadata)])
                        chunks.extend(sub_chunks)
        # Strategy 5: Default - larger chunks for general docs
        else:
            sub_chunks = MarkdownTextSplitter(
                chunk_size=900, chunk_overlap=180
            ).split_documents([doc])
            chunks.extend(sub_chunks)
    
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
    lang_docs = load_md_files([SELECTED_DOCS_PATH])
    print(f"  ✅ Loaded {len(lang_docs)} files total")
    lang_chunks = chunk_and_filter(lang_docs, chunk_size=400,
                                   chunk_overlap=50, apply_filter=True)
    build_db(lang_chunks, LANG_DB_PATH, embeddings)

    # --- 2. Package / dataset DB (context-relevant) ---
    print("\n📦 Building package/dataset DB...")
    pkg_docs = load_md_files([PACKAGES_PATH, DATASETS_PATH])
    if pkg_docs:
        print(f"  ✅ Loaded {len(pkg_docs)} files total")
        # Use smart chunking: preserves code examples, action definitions, and sections
        # Chunk sizes: metadata whole, examples 1000-1200, reference 1000, config 800, default 900
        pkg_chunks = chunk_packages_by_section(pkg_docs)
        print(f"  ✨ Smart chunking applied: {len(pkg_chunks)} semantic chunks")
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
