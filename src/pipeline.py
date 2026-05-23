"""
pipeline.py

Main entry point. Implements the full architecture:

  User intent
    → IntentDecomposer  (task breakdown + lang spec retrieval)
    → PkgRetriever      (package/dataset retrieval)
    → Prompt construction
    → Ollama (BraneScript generation)
    → Syntax check      (on fail: retry with error, max 3 attempts)
    → Semantic check    (on fail: retry with error, max 3 attempts)
    → Execute workflow  (placeholder — hook in your Brane runner here)
    → Save to example store (on success)

Run:
    python pipeline.py
"""

import os
import re
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from intent_decomposer import IntentDecomposer
from pkg_retriever import PkgRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LANG_DB_PATH          = "../brane_lang_db"
PKG_DB_PATH           = "../brane_pkg_db"
EXAMPLE_DB_PATH       = "../brane_lang_db"    # successful scripts go back into lang DB
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
BRANESCRIPT_OUTPUT_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "generated_branescripts")
)
EMBEDDING_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"

MAX_RETRIES = 3    # max attempts for syntax and semantic loops

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Main generation prompt
GENERATION_TEMPLATE = """You are an expert in the Brane Framework and BraneScript.

The user's request has been broken into these sub-tasks. Address ALL of them:
{subtasks}

LANGUAGE SPECIFICATION (syntax reference):
{lang_context}

PACKAGE / DATASET CONTEXT:
{pkg_context}

USER REQUEST:
{question}

{error_section}Output ONLY valid BraneScript code.
Use inline comments (// ...) to mark any assumptions about parameter names or types.
Do not assume packages, functions or datasets that are not mentioned in the PACKAGE / DATASET CONTEXT.
Do not invent any new top-level functions or variables.
Define every variable before it is used.
Use exact BraneScript assignment syntax:
  let <name> := <expression>;
Do not use `=` for assignment.
If the context is incomplete, ask a single clarifying question instead of making assumptions.
If you ask a question, do not generate any code.
If you have enough information, generate valid BraneScript code only.

CLARIFICATION HISTORY:
{clarification_section}

BRANESCRIPT CODE:"""

# Error section injected on retry
SYNTAX_ERROR_SECTION = """⚠️  Your previous attempt had a SYNTAX ERROR:
{error}
Fix the syntax error and regenerate the complete corrected code.

"""

SEMANTIC_ERROR_SECTION = """⚠️  Your previous attempt had a SEMANTIC ERROR:
{error}
The package name, dataset name, or function signature may be wrong.
Check the PACKAGE / DATASET CONTEXT above carefully and fix the issue.

"""

# ---------------------------------------------------------------------------
# Syntax check
# ---------------------------------------------------------------------------
def check_syntax(code: str) -> tuple[bool, str]:
    """
    Check BraneScript syntax.

    Currently a heuristic check — replace this with a real Brane parser
    call when available:
        result = subprocess.run(["brane", "check", "--stdin"], input=code, ...)

    Returns (is_valid, error_message).
    """
    errors = []

    # Check balanced braces
    if code.count("{") != code.count("}"):
        errors.append("Unbalanced braces: { and } counts do not match.")

    # Check balanced parentheses
    if code.count("(") != code.count(")"):
        errors.append("Unbalanced parentheses: ( and ) counts do not match.")

    # let assignments must use :=
    let_lines = [l for l in code.splitlines() if re.match(r'\s*let\s+\w+\s*=', l)]
    for ll in let_lines:
        if ":=" not in ll:
            errors.append(f"Assignment should use ':=' not '=': {ll.strip()}")

    if errors:
        return False, "\n".join(errors)
    return True, ""


# ---------------------------------------------------------------------------
# Semantic check
# ---------------------------------------------------------------------------
def check_semantic(code: str, pkg_context: str) -> tuple[bool, str]:
    """
    Check that package and dataset names used in the code appear
    in the retrieved pkg_context.

    Returns (is_valid, error_message).
    """
    errors = []

    # Extract import/package references from the generated code
    import_names = re.findall(
        r'import\s+([A-Za-z][A-Za-z0-9_\-]*)', code, re.IGNORECASE
    )
    pkg_calls = re.findall(
        r'([A-Za-z][A-Za-z0-9_\-]+)\s*::', code
    )

    referenced = set(import_names + pkg_calls)

    for name in referenced:
        if name.lower() in ("std", "io", "math"):
            continue    # standard lib, always valid
        if name not in pkg_context:
            errors.append(
                f"'{name}' is referenced in the code but not found in the "
                f"package/dataset context. Check spelling or availability."
            )

    if errors:
        return False, "\n".join(errors)
    return True, ""


# ---------------------------------------------------------------------------
# Save successful script to example store
# ---------------------------------------------------------------------------
def save_to_example_store(code: str, user_query: str,
                           embeddings: HuggingFaceEmbeddings) -> None:
    """
    Appends a successful BraneScript to the language spec DB as a new example.
    This is the 'grow example storage' feedback loop in the architecture.
    """
    from langchain_core.documents import Document
    doc = Document(
        page_content=f"// Example: {user_query}\n\n{code}",
        metadata={"source": "generated_example", "query": user_query}
    )
    try:
        db = Chroma(persist_directory=EXAMPLE_DB_PATH, embedding_function=embeddings)
        db.add_documents([doc])
        print("\n💾 Saved successful script to example store.")
    except Exception as e:
        print(f"\n⚠️  Could not save example: {e}")


def make_safe_filename(text: str, max_length: int = 50) -> str:
    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    filename = re.sub(r"_+", "_", filename)
    return filename[:max_length].strip("_") or "branescript"


def save_branescript_to_folder(code: str, user_query: str, output_dir: str = BRANESCRIPT_OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = make_safe_filename(user_query)
    filename = f"branescript_{timestamp}_{safe_query}.brane"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"// Generated from user query: {user_query}\n")
        f.write("// Saved by pipeline.py\n\n")
        f.write(code)
    print(f"\n💾 Saved BraneScript to folder: {path}")
    return path


# ---------------------------------------------------------------------------
# Execute workflow (placeholder)
# ---------------------------------------------------------------------------
def is_clarification_request(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    first_line = stripped.splitlines()[0].lower()
    question_prefixes = (
        "what", "which", "when", "where", "why", "how",
        "do", "does", "did", "can", "could", "would", "should",
        "is", "are", "may", "might", "please",
    )

    if stripped.endswith("?"):
        return True
    if any(first_line.startswith(prefix) for prefix in question_prefixes):
        return True
    if "need more information" in stripped.lower() or "clarify" in stripped.lower():
        return True
    return False


def ask_for_clarification(question: str) -> str:
    print("\n❓ The model is asking for clarification:")
    print(question.strip())
    answer = input("Your answer: ")
    return answer.strip()


def execute_workflow(code: str) -> tuple[bool, str]:
    """
    Hook your actual Brane execution here.
    e.g.: result = subprocess.run(["brane", "run", "--stdin"], input=code, ...)

    Returns (success, result_or_error_message).
    """
    print("\n🚀 Execute workflow: (placeholder — hook Brane runner here)")
    return True, "Execution placeholder: would run via `brane run`"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(user_query: str,
                 decomposer: IntentDecomposer,
                 pkg_retriever: PkgRetriever,
                 llm: Ollama,
                 embeddings: HuggingFaceEmbeddings) -> str:

    # ── Step 1: Task breakdown + language spec retrieval ──────────────────
    lang_context, subtasks = decomposer.run(user_query)
    subtasks_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(subtasks))

    # ── Step 2: Package / dataset retrieval ───────────────────────────────
    pkg_context = pkg_retriever.run(subtasks, user_query)
    print("\n📦 Retrieved package/dataset context:")
    if pkg_context.strip():
        print(pkg_context)
    else:
        print("   - (No package/dataset context returned)")

    clarification_history: list[tuple[str, str]] = []
    clarification_section = ""

    # ── Step 3: Generation loop (clarification + validation) ─────────────
    prompt = ChatPromptTemplate.from_template(GENERATION_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    error_section = ""
    code = ""
    attempt = 1

    while attempt <= MAX_RETRIES:
        print(f"\n⚙️  Generation attempt {attempt}/{MAX_RETRIES}...")

        code = chain.invoke({
            "subtasks":            subtasks_str,
            "lang_context":       lang_context,
            "pkg_context":        pkg_context,
            "question":           user_query,
            "error_section":      error_section,
            "clarification_section": clarification_section,
        })

        if is_clarification_request(code):
            answer = ask_for_clarification(code)
            if not answer:
                print("   ⚠️ No clarification answer provided. Aborting.")
                return code
            clarification_history.append((code.strip(), answer))
            clarification_section = "\n\n".join(
                f"Q: {q}\nA: {a}" for q, a in clarification_history
            )
            error_section = ""
            continue

        # ── Syntax check ──────────────────────────────────────────────────
        syntax_ok, syntax_error = check_syntax(code)
        if not syntax_ok:
            print(f"   ❌ Syntax check failed: {syntax_error}")
            if attempt < MAX_RETRIES:
                error_section = SYNTAX_ERROR_SECTION.format(error=syntax_error)
                attempt += 1
                continue
            else:
                print(f"   ⛔ Max retries reached on syntax. Returning last attempt.")
                return code

        print("   ✅ Syntax check passed")

        # ── Semantic check ─────────────────────────────────────────────────
        semantic_ok, semantic_error = check_semantic(code, pkg_context)
        if not semantic_ok:
            print(f"   ❌ Semantic check failed: {semantic_error}")
            if attempt < MAX_RETRIES:
                error_section = SEMANTIC_ERROR_SECTION.format(error=semantic_error)
                attempt += 1
                continue
            else:
                print(f"   ⛔ Max retries reached on semantic. Returning last attempt.")
                return code

        print("   ✅ Semantic check passed")
        break   # both checks passed

    # ── Step 4: Execute workflow ───────────────────────────────────────────
    #exec_ok, exec_result = execute_workflow(code)
    #print(f"   {'✅' if exec_ok else '❌'} {exec_result}")
#
    ## ── Step 5: Save generated BraneScript to folder ─────────────────────────
    if code.strip():
        save_branescript_to_folder(code, user_query)

    return code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔧 Initialising models and databases...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    llm = Ollama(model="llama3", temperature=0)

    # Language spec DB (built by knowledgeBase.py)
    lang_db = Chroma(persist_directory=LANG_DB_PATH, embedding_function=embeddings)

    # Package/dataset DB (built by knowledgeBase.py)
    # If it doesn't exist yet, use lang_db as fallback so PkgRetriever still works without package context
    pkg_db = None
    if os.path.exists(PKG_DB_PATH):
        print(f"✅  Found package DB at {PKG_DB_PATH}. Loading...")
        pkg_db = Chroma(persist_directory=PKG_DB_PATH, embedding_function=embeddings)
    else:
        print(f"⚠️  Package DB not found at {PKG_DB_PATH}.")
        print("   Run knowledgeBase.py after adding package docs to ../submodules/packages")
        print("   Continuing without package context...\n")
        # Use lang_db as fallback so PkgRetriever still works
        pkg_db = lang_db

    decomposer    = IntentDecomposer(llm=llm, lang_db=lang_db, k_per_subtask=3)
    pkg_retriever = PkgRetriever(pkg_db=pkg_db, k=4)

    user_query = (
        "I want to run a private cardiovascular analysis on a patient of age 42, sex male, height 175 and weight 70. After that, generate a report summarising the results."
        "Use package \"Healthcare\". Make sure to use the correct BraneScript "
    )

    print(f"\n🧠 Intent: {user_query}")
    print("⏳ Running pipeline...\n" + "─" * 50)

    result = run_pipeline(
        user_query=user_query,
        decomposer=decomposer,
        pkg_retriever=pkg_retriever,
        llm=llm,
        embeddings=embeddings,
    )

    print("\n" + "=" * 50)
    print("FINAL BRANESCRIPT OUTPUT:")
    print("=" * 50)
    print(result)
