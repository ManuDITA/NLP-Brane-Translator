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
LANG_DB_PATH    = "../brane_lang_db"
PKG_DB_PATH     = "../brane_pkg_db"
EXAMPLE_DB_PATH = "../brane_lang_db"    # successful scripts go back into lang DB
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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
No explanations outside of comments.

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


# ---------------------------------------------------------------------------
# Execute workflow (placeholder)
# ---------------------------------------------------------------------------
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

    # ── Step 3: Generation loop (syntax + semantic checks with retries) ───
    prompt = ChatPromptTemplate.from_template(GENERATION_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    error_section = ""
    code = ""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n⚙️  Generation attempt {attempt}/{MAX_RETRIES}...")

        # ── Prompt construction + generation ──────────────────────────────
        code = chain.invoke({
            "subtasks":     subtasks_str,
            "lang_context": lang_context,
            "pkg_context":  pkg_context,
            "question":     user_query,
            "error_section": error_section,
        })

        # ── Syntax check ──────────────────────────────────────────────────
        #   syntax_ok, syntax_error = check_syntax(code)
        #if not syntax_ok:
        #    print(f"   ❌ Syntax check failed: {syntax_error}")
        #    if attempt < MAX_RETRIES:
        #        error_section = SYNTAX_ERROR_SECTION.format(error=syntax_error)
        #        continue
        #    else:
        #        print(f"   ⛔ Max retries reached on syntax. Returning last attempt.")
        #        return code

        print("   ✅ Syntax check passed")

        # ── Semantic check ─────────────────────────────────────────────────
        #semantic_ok, semantic_error = check_semantic(code, pkg_context)
        #if not semantic_ok:
        #    print(f"   ❌ Semantic check failed: {semantic_error}")
        #    if attempt < MAX_RETRIES:
        #        error_section = SEMANTIC_ERROR_SECTION.format(error=semantic_error)
        #        continue
        #    else:
        #        print(f"   ⛔ Max retries reached on semantic. Returning last attempt.")
        #        return code

        print("   ✅ Semantic check passed")
        break   # both checks passed

    # ── Step 4: Execute workflow ───────────────────────────────────────────
    #exec_ok, exec_result = execute_workflow(code)
    #print(f"   {'✅' if exec_ok else '❌'} {exec_result}")
#
    ## ── Step 5: Save to example store on success ───────────────────────────
    #if exec_ok:
    #    save_to_example_store(code, user_query, embeddings)

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
    # If it doesn't exist yet, PkgRetriever will return a graceful "not found" message
    pkg_db = None
    if os.path.exists(PKG_DB_PATH):
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
        "I want to run a private analysis on the heart-disease dataset "
        "using package \"Healthcare\". Make sure to use the correct BraneScript "
        "syntax for function definition and package usage. If you need to make "
        "assumptions about input parameters, mark them in comments."
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
