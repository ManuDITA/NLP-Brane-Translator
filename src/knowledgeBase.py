"""
knowledgeBase.py

Builds the package/dataset ChromaDB used at inference time:
  brane_pkg_db — packages and datasets (function signatures, schemas, Data usage)

BraneScript syntax context is injected directly from data/syntax_reference.md
at inference time, so no language-spec DB is needed here.

Run this once, or whenever your packages or datasets change:
    python src/knowledgeBase.py
"""

import os
import shutil
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_hf_token

load_hf_token()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

PACKAGES_PATH = BASE_DIR / "packages"
DATASETS_PATH = BASE_DIR / "datasets"

PKG_DB_PATH  = BASE_DIR / "brane_pkg_db"

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_package_docs(packages_path: Path) -> list:
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


def load_dataset_docs(datasets_path: Path) -> list:
    """
    Index registered Brane datasets so the LLM knows their names and how to
    reference them in BraneScript.

    For each dataset directory we:
      1. Parse data.yml to extract the registered name (e.g. "heal_pa_2").
      2. Emit a synthetic Document that states the BraneScript usage pattern.
      3. Load README.md if present for additional context.
    """
    docs = []
    if not datasets_path.exists():
        print(f"  ⚠️  Datasets path not found, skipping: {datasets_path}")
        return docs

    for dataset_dir in sorted(datasets_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_folder = dataset_dir.name
        data_yml = dataset_dir / "data.yml"
        if not data_yml.exists():
            continue

        try:
            raw = data_yml.read_text(encoding="utf-8")
        except Exception as e:
            print(f"    ⚠️  Could not read {data_yml}: {e}")
            continue

        # Extract the registered name from the YAML (simple parse — avoids pyyaml dep)
        registered_name = None
        for line in raw.splitlines():
            if line.strip().startswith("name:"):
                registered_name = line.split(":", 1)[1].strip()
                break

        if not registered_name:
            print(f"    ⚠️  Could not find 'name:' in {data_yml}, skipping")
            continue

        print(f"  📊 Dataset: {dataset_folder} → registered name: \"{registered_name}\"")

        # Synthetic document — most important piece: tells the LLM the exact name
        synthetic = (
            f"## Registered Dataset: {registered_name}\n\n"
            f"Folder: datasets/{dataset_folder}/\n\n"
            f"This dataset is registered in Brane under the name `\"{registered_name}\"`.\n"
            f"To reference it in BraneScript:\n"
            f"```\n"
            f"let ds := new Data{{ name := \"{registered_name}\" }};\n"
            f"```\n"
            f"Pass `ds` to any package function that accepts a `Data` argument.\n"
            f"Do NOT pass the name as a plain string — always use `new Data{{ name := \"...\" }}`.\n\n"
            f"Full data.yml:\n{raw}"
        )
        # dataset_folder matches the package name (data_masking, epidemics, etc.)
        package_name = dataset_folder
        docs.append(Document(
            page_content=synthetic,
            metadata={
                "source": str(data_yml),
                "dataset": registered_name,
                "package": package_name,
                "doc_type": "dataset_registry",
            }
        ))

        # README for richer context (data format, available columns, etc.)
        readme = dataset_dir / "README.md"
        if readme.exists():
            try:
                text = readme.read_text(encoding="utf-8")
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(readme),
                        "dataset": registered_name,
                        "package": package_name,
                        "doc_type": "dataset_readme",
                    }
                ))
                print(f"    ✅ Loaded README.md")
            except Exception as e:
                print(f"    ⚠️  Could not read {readme}: {e}")

    print(f"  📁 Loaded {len(docs)} dataset documents")
    return docs


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

    print("\n📦 Building package + dataset DB...")
    pkg_docs = load_package_docs(PACKAGES_PATH)
    dataset_docs = load_dataset_docs(DATASETS_PATH)
    all_pkg_docs = pkg_docs + dataset_docs
    if all_pkg_docs:
        # Dataset registry docs are short and critical — keep them whole (no chunking)
        registry_chunks = [d for d in dataset_docs if d.metadata.get("doc_type") == "dataset_registry"]
        # Everything else goes through smart chunking
        chunked_docs = [d for d in all_pkg_docs if d not in registry_chunks]
        pkg_chunks = chunk_packages_by_section(chunked_docs) + registry_chunks
        print(f"  ✅ {len(pkg_docs)} package docs + {len(dataset_docs)} dataset docs")
        print(f"  ✨ Smart chunking: {len(pkg_chunks)} total chunks")
        build_db(pkg_chunks, PKG_DB_PATH, embeddings)
    else:
        print("  ℹ️  No package or dataset docs found — pkg DB skipped.")

    print(f"\n🎉 Done. DB ready at: {PKG_DB_PATH}")


if __name__ == "__main__":
    build_knowledge_base()
