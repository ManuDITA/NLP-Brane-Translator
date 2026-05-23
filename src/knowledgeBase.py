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
SELECTED_DOCS_PATH      = BASE_DIR / "selectedDocs"
LANGUAGE_SPEC_PATH      = BASE_DIR / "submodules/languageSpec"
LANGUAGE_SPEC_EXAMPLES  = LANGUAGE_SPEC_PATH / "branescript"
PACKAGES_PATH           = BASE_DIR / "submodules/packages"

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


def load_text_files(paths: list[str], extensions: tuple[str, ...]) -> list[Document]:
    docs = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  ⚠️  Path not found, skipping: {path}")
            continue
        loaded_files = 0
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(extensions):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            text = f.read()
                    except Exception as e:
                        print(f"    ⚠️  Could not read {filepath}: {e}")
                        continue
                    docs.append(Document(page_content=text,
                                         metadata={"source": filepath}))
                    loaded_files += 1
        print(f"  📁 {path}: {loaded_files} files")
    return docs


def load_language_spec_docs(language_spec_path: Path) -> list[Document]:
    docs = []
    if not language_spec_path.exists():
        print(f"  ⚠️  Language spec path not found, skipping: {language_spec_path}")
        return docs

    # Load Rust-based syntax and semantics definitions
    rs_docs = load_text_files([language_spec_path], (".rs",))
    docs.extend(rs_docs)

    # Load BraneScript examples from the dedicated examples folder
    example_docs = load_text_files([LANGUAGE_SPEC_EXAMPLES], (".bs",))
    docs.extend(example_docs)

    print(f"  📁 Loaded {len(docs)} languageSpec documents")
    return docs


def chunk_language_spec_docs(docs: list[Document]) -> list[Document]:
    rs_docs = [d for d in docs if d.metadata.get("source", "").lower().endswith(".rs")]
    bs_docs = [d for d in docs if d.metadata.get("source", "").lower().endswith(".bs")]

    # Keep BraneScript examples whole and chunk Rust spec source only.
    bs_chunks = bs_docs
    rs_chunks = chunk_and_filter(rs_docs, chunk_size=1200, chunk_overlap=200, apply_filter=False)
    return rs_chunks + bs_chunks


def load_package_docs(packages_path: Path) -> list[Document]:
    docs = []
    if not packages_path.exists():
        print(f"  ⚠️  Packages path not found, skipping: {packages_path}")
        return docs

    for package_dir in sorted(packages_path.iterdir()):
        if not package_dir.is_dir():
            continue
        package_name = package_dir.name
        print(f"  🔧 Loading package: {package_name}")

        # Load the main package reference documents first
        for filename in ("QUICK_REFERENCE.md", "quick_reference.md", "CONFIGURATION.md", "configuration.md"):
            path = package_dir / filename
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "package": package_name,
                            "doc_type": filename.lower().replace(".md", ""),
                        }
                    ))
                    print(f"    ✅ Loaded {filename}")
                except Exception as e:
                    print(f"    ⚠️  Could not read {path}: {e}")

        # Also ingest container / configuration YAML files if present
        for yaml_ext in ("*.yml", "*.yaml"):
            for path in sorted(package_dir.glob(yaml_ext)):
                if path.name.lower() in ("container.yml", "container.yaml", "configuration.yml", "config.yml"):
                    try:
                        text = path.read_text(encoding="utf-8")
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": str(path),
                                "package": package_name,
                                "doc_type": "package_yaml",
                            }
                        ))
                        print(f"    ✅ Loaded {path.name}")
                    except Exception as e:
                        print(f"    ⚠️  Could not read {path}: {e}")

        # If no quick reference or configuration docs exist, optionally include README.md as fallback
        if not any(d.metadata.get("package") == package_name for d in docs):
            readme_path = package_dir / "README.md"
            if readme_path.exists():
                try:
                    text = readme_path.read_text(encoding="utf-8")
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": str(readme_path),
                            "package": package_name,
                            "doc_type": "readme",
                        }
                    ))
                    print(f"    ✅ Loaded README.md as fallback")
                except Exception as e:
                    print(f"    ⚠️  Could not read {readme_path}: {e}")

    print(f"  📁 Loaded {len(docs)} package documents")
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
        filename = os.path.basename(source).lower()
        
        # Heuristic: detect document type from filename/content
        is_config = "configuration" in filename or "config" in text[:200].lower()
        is_example = "example" in filename or text.count("bscript") > 2
        is_reference = "quick_reference" in filename or "quick reference" in text[:200].lower()
        is_readme = filename == "readme.md" or filename == "readme"
        is_yaml = filename.endswith(('.yml', '.yaml'))
        
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
        elif is_yaml:
            if len(text) < 2000:
                chunks.append(doc)
            else:
                sub_chunks = MarkdownTextSplitter(
                    chunk_size=1200, chunk_overlap=200
                ).split_documents([doc])
                chunks.extend(sub_chunks)
        # Strategy 3: Configuration guides need context (larger chunks)
        elif is_config:
            sub_chunks = MarkdownTextSplitter(
                chunk_size=900, chunk_overlap=180
            ).split_documents([doc])
            chunks.extend(sub_chunks)
        elif is_reference:
            sections = text.split("\n## ")
            for section in sections:
                if section.strip():
                    formatted = "## " + section if not section.startswith("#") else section
                    if len(formatted) < 1400:
                        chunks.append(Document(page_content=formatted,
                                             metadata=doc.metadata))
                    else:
                        sub_chunks = MarkdownTextSplitter(
                            chunk_size=1100, chunk_overlap=220
                        ).split_documents([Document(page_content=formatted,
                                                   metadata=doc.metadata)])
                        chunks.extend(sub_chunks)
        elif is_readme:
            sub_chunks = MarkdownTextSplitter(
                chunk_size=800, chunk_overlap=150
            ).split_documents([doc])
            chunks.extend(sub_chunks)
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

    # --- 1. Language spec DB (selectedDocs first, supplemented by manual + specification) ---
    print("\n📚 Building language specification DB...")
    selected_docs = load_md_files([SELECTED_DOCS_PATH])
    language_spec_docs = load_language_spec_docs(LANGUAGE_SPEC_PATH)
    print(
        f"  ✅ Loaded {len(selected_docs)} selectedDocs, "
        f"{len(language_spec_docs)} languageSpec files"
    )

    selected_chunks = chunk_and_filter(selected_docs, chunk_size=400,
                                       chunk_overlap=50, apply_filter=False)
    language_spec_chunks = chunk_language_spec_docs(language_spec_docs)
    lang_chunks = selected_chunks + language_spec_chunks
    build_db(lang_chunks, LANG_DB_PATH, embeddings)

    # --- 2. Package DB (package quick reference + configuration only) ---
    print("\n📦 Building package DB...")
    pkg_docs = load_package_docs(PACKAGES_PATH)
    if pkg_docs:
        pkg_chunks = chunk_packages_by_section(pkg_docs)
        print(f"  ✅ Loaded {len(pkg_docs)} package docs total")
        print(f"  ✨ Smart chunking applied: {len(pkg_chunks)} semantic chunks")
        build_db(pkg_chunks, PKG_DB_PATH, embeddings)
    else:
        print("  ℹ️  No package docs found — package DB skipped.")
        print("      Populate ../submodules/packages with package docs, quick references, and configuration files")
        print("      then re-run this script.")

    print("\n🎉 Done. DBs ready.")
    print(f"   Lang spec : {LANG_DB_PATH}")
    print(f"   Pkg/data  : {PKG_DB_PATH}")


if __name__ == "__main__":
    build_knowledge_base()
