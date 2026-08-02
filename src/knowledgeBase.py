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

        # Load keywords.txt — stores package aliases so pkg_retriever can detect
        # this package from domain-specific vocabulary without hardcoding in Python.
        keywords_path = package_dir / "keywords.txt"
        if keywords_path.exists():
            try:
                raw = keywords_path.read_text(encoding="utf-8")
                keywords = [
                    line.strip() for line in raw.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                if keywords:
                    docs.append(Document(
                        page_content=(
                            f"## {package_name} — keyword aliases\n\n"
                            + "\n".join(f"- {kw}" for kw in keywords)
                        ),
                        metadata={
                            "source": str(keywords_path),
                            "package": package_name,
                            "doc_type": "package_keywords",
                            "keywords": ",".join(keywords),
                        }
                    ))
                    print(f"    ✅ Loaded keywords.txt ({len(keywords)} aliases)")
            except Exception as e:
                print(f"    ⚠️  Could not read {keywords_path}: {e}")

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


def _scan_dataset_names(datasets_path: Path) -> dict[str, str]:
    """
    Scan datasets/ directory and return {package_folder_name: registered_dataset_name}.
    Reads the 'name:' field from each data.yml.  Used to generate accurate
    BraneScript examples without hardcoding dataset names in this file.
    """
    mapping: dict[str, str] = {}
    if not datasets_path.exists():
        return mapping
    for dataset_dir in sorted(datasets_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        data_yml = dataset_dir / "data.yml"
        if not data_yml.exists():
            continue
        try:
            raw = data_yml.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in raw.splitlines():
            if line.strip().startswith("name:"):
                registered = line.split(":", 1)[1].strip()
                mapping[dataset_dir.name] = registered
                break
    return mapping


def _use_when(input_types: list[str]) -> str:
    """Return a 'Use when' hint based on a function's input types."""
    if "Data" in input_types:
        return "Use when: intent refers to a dataset or a batch of records."
    if "IntermediateResult" in input_types:
        return "Use when: chaining from the output of a previous function."
    named = [t for t in input_types
             if t not in ("string", "integer", "real", "bool", "Data", "IntermediateResult")]
    if named:
        return f"Use when: intent specifies a single {named[0]} with explicit inline values."
    return ""


# Default placeholder values for BraneScript primitive types
_PRIM_DEFAULT: dict[str, str] = {
    "string":  '"example"',
    "integer": "0",
    "int":     "0",
    "real":    "0.0",
    "float":   "0.0",
    "bool":    "true",
}


def _type_lines(var_name: str, type_name: str, type_defs: dict,
                _depth: int = 0) -> list[str]:
    """
    Return BraneScript lines that define var_name of the given class type.
    Nested class-type fields are defined as sub-variables first.
    """
    if _depth > 2 or type_name not in type_defs:
        return [f"# provide {var_name}: {type_name}"]

    props = type_defs[type_name].get("properties", [])
    pre_lines: list[str] = []
    field_parts: list[str] = []

    for p in props:
        fname, ftype = p["name"], p["type"]
        if ftype in _PRIM_DEFAULT:
            field_parts.append(f"    {fname} := {_PRIM_DEFAULT[ftype]},")
        elif ftype in type_defs:
            sub_var = f"{fname}_val"
            pre_lines.extend(_type_lines(sub_var, ftype, type_defs, _depth + 1))
            field_parts.append(f"    {fname} := {sub_var},")
        else:
            field_parts.append(f"    # {fname}: {ftype}")

    return pre_lines + [f"let {var_name} := new {type_name} {{"] + field_parts + ["};"]


def _branescript_example(pkg: str, fn_name: str,
                          inputs: list[dict], outputs: list[dict],
                          dataset_names: dict[str, str] | None = None,
                          type_defs: dict | None = None) -> str:
    """
    Generate a minimal BraneScript usage example for one function.
    All types are derived from type_defs (parsed from container.yml) — nothing is hardcoded.
    dataset_names maps package_folder → registered dataset name.
    """
    lines = [f"import {pkg};", ""]
    output_type = outputs[0]["type"] if outputs else "string"
    ds_map = dataset_names or {}
    types = type_defs or {}

    call_args = []
    for inp in inputs:
        iname = inp["name"]
        itype = inp["type"]
        if itype == "Data":
            ds = ds_map.get(pkg, f"{pkg}_dataset")
            lines.append(f'let {iname} := new Data{{ name := "{ds}" }};')
            call_args.append(iname)
        elif itype == "IntermediateResult":
            lines.append(f"# '{iname}' is the IntermediateResult from a previous step")
            call_args.append(iname)
        elif itype in _PRIM_DEFAULT:
            lines.append(f"let {iname} := {_PRIM_DEFAULT[itype]};")
            call_args.append(iname)
        else:
            # Named class type — generate from type definitions
            lines.extend(_type_lines(iname, itype, types))
            call_args.append(iname)

    lines += ["", f"let result := {fn_name}({', '.join(call_args)});"]

    if output_type == "IntermediateResult":
        lines.append('commit_result("result_name", result);')
    else:
        lines.append("println(result);")

    return "\n".join(lines)


def build_function_chunks(doc: Document,
                           dataset_names: dict[str, str] | None = None) -> list[Document]:
    """
    Parse a container.yml Document and return one chunk per function.

    Each chunk contains:
      - Function name and full signature
      - Input/output types
      - Description from container.yml
      - "Use when" hint (Data vs Patient vs string)
      - A minimal BraneScript usage example

    A separate chunk is created for all type/class definitions.

    Falls back to returning the original doc if YAML parsing fails.
    """
    try:
        import yaml
    except ImportError:
        return [doc]

    try:
        data = yaml.safe_load(doc.page_content)
    except Exception:
        return [doc]

    if not isinstance(data, dict):
        return [doc]

    pkg_name = doc.metadata.get("package", data.get("name", "unknown"))
    actions  = data.get("actions", {})
    types    = data.get("types", {})
    chunks   = []

    # ── Types/classes chunk ─────────────────────────────────────────────────
    if types:
        type_lines = [f"## {pkg_name} — Class Types\n"]
        type_lines.append(
            "These class types are exported by the package. "
            "Use them directly in BraneScript — do NOT redeclare them.\n"
        )
        for tname, tdef in types.items():
            props = tdef.get("properties", [])
            fields = ", ".join(f"{p['name']}: {p['type']}" for p in props)
            type_lines.append(f"### {tname}")
            type_lines.append(f"Fields: {fields}")
            type_lines.append("")
        chunks.append(Document(
            page_content="\n".join(type_lines),
            metadata={**doc.metadata, "doc_type": "package_types"},
        ))

    # ── One chunk per function ──────────────────────────────────────────────
    for fn_name, action in actions.items():
        inputs  = action.get("input", [])
        outputs = action.get("output", [])
        desc    = action.get("description", "")
        input_types  = [i["type"] for i in inputs]
        output_types = [o["type"] for o in outputs]

        sig = (
            f"{fn_name}("
            + ", ".join(f"{i['name']}: {i['type']}" for i in inputs)
            + f") -> {output_types[0] if output_types else 'void'}"
        )
        use_when = _use_when(input_types)
        example  = _branescript_example(pkg_name, fn_name, inputs, outputs,
                                        dataset_names, types)

        content = "\n".join(filter(None, [
            f"## {pkg_name}.{fn_name}",
            "",
            desc,
            "",
            f"Signature: `{sig}`",
            f"Use when: {use_when}" if use_when else "",
            "",
            "BraneScript example:",
            "```",
            example,
            "```",
        ]))

        chunks.append(Document(
            page_content=content,
            metadata={
                **doc.metadata,
                "doc_type": "function",
                "function": fn_name,
            },
        ))

    return chunks if chunks else [doc]


def chunk_packages_by_section(docs: list[Document],
                               dataset_names: dict[str, str] | None = None) -> list[Document]:
    """
    Chunking strategy for package documentation.

    container.yml → one chunk per function (via build_function_chunks)
    Everything else → markdown-aware splitting
    """
    chunks = []

    for doc in docs:
        source = doc.metadata.get("source", "")
        filename = os.path.basename(source).lower()

        # container.yml → explode into per-function chunks
        if "container.yml" in source or filename in ("container.yml", "container.yaml"):
            chunks.extend(build_function_chunks(doc, dataset_names))
            continue

        # Everything else: markdown-aware splitting
        is_readme = filename in ("readme.md", "readme")
        chunk_size = 800 if is_readme else 900
        overlap    = 150 if is_readme else 180
        chunks.extend(
            MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            .split_documents([doc])
        )

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

    # Scan datasets once so function-chunk examples use real dataset names
    dataset_names = _scan_dataset_names(DATASETS_PATH)
    print(f"  📊 Dataset name map: {dataset_names}")

    print("\n📦 Building package + dataset DB...")
    pkg_docs = load_package_docs(PACKAGES_PATH)
    dataset_docs = load_dataset_docs(DATASETS_PATH)
    all_pkg_docs = pkg_docs + dataset_docs
    if all_pkg_docs:
        # Dataset registry docs are short and critical — keep them whole (no chunking)
        registry_chunks = [d for d in dataset_docs if d.metadata.get("doc_type") == "dataset_registry"]
        # keywords docs are also short — keep them whole
        keywords_chunks = [d for d in pkg_docs if d.metadata.get("doc_type") == "package_keywords"]
        # Everything else goes through smart chunking
        skip = set(id(d) for d in registry_chunks + keywords_chunks)
        chunked_docs = [d for d in all_pkg_docs if id(d) not in skip]
        pkg_chunks = chunk_packages_by_section(chunked_docs, dataset_names) + registry_chunks + keywords_chunks
        print(f"  ✅ {len(pkg_docs)} package docs + {len(dataset_docs)} dataset docs")
        print(f"  ✨ Smart chunking: {len(pkg_chunks)} total chunks")
        build_db(pkg_chunks, PKG_DB_PATH, embeddings)
    else:
        print("  ℹ️  No package or dataset docs found — pkg DB skipped.")

    print(f"\n🎉 Done. DB ready at: {PKG_DB_PATH}")


if __name__ == "__main__":
    build_knowledge_base()
