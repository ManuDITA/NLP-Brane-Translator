"""
pipeline.py

Main entry point. Implements the simplified architecture:

  User request
    → Load raw workspace files as LLM context
    → Prompt construction
    → Ollama (BraneScript generation)
    → Syntax check      (on fail: retry with error, max 3 attempts)
    → Save generated BraneScript to disk

Run:
    python pipeline.py
"""

import os
import re
from datetime import datetime
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from intent_decomposer import IntentDecomposer
from pkg_retriever import PkgRetriever

# ---------------------------------------------------------------------------
# BraneScript canonical few-shot example injected into every prompt
# ---------------------------------------------------------------------------
BRANESCRIPT_FEW_SHOT = """
## BRANESCRIPT SYNTAX REFERENCE (few-shot examples)

Example 1 – import a package and call a function:
```
import healthcare;

let patient := "{\\"patient_id\\": \\"PAT001\\", \\"age\\": 55, \\"gender\\": \\"M\\", \\"vital_signs\\": {\\"blood_pressure\\": 150, \\"heart_rate\\": 80}, \\"lab_results\\": {\\"total_cholesterol\\": 220}, \\"medical_history\\": [\\"hypertension\\"]}";
let result := healthcare::analyze_heart_disease(patient);
println(result);
```

Example 2 – variable assignment and if/else:
```
let x := 42;
let msg := "hello";
if (x > 10) {
    println(msg);
} else {
    println("small");
}
```

Example 3 – function definition and call:
```
func greet(name: string) -> string {
    return "Hello, " + name;
}
let greeting := greet("world");
println(greeting);
```

Example 4 – routing execution to a named node/site with #[on("name")]:
```
import healthcare;

let patient := "{\\"patient_id\\": \\"PAT001\\", \\"age\\": 45, \\"gender\\": \\"M\\", \\"vital_signs\\": {\\"blood_pressure\\": 100, \\"heart_rate\\": 75}, \\"lab_results\\": {\\"total_cholesterol\\": 200}, \\"medical_history\\": []}";

#[on("marco")]
let result := healthcare::analyze_heart_disease(patient);
println(result);
```
NOTE: whenever the user says "on node X", "on site X", "at location X", or "run on X",
place `#[on("X")]` immediately before the relevant function call or block.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
BRANESCRIPT_OUTPUT_DIR = (PROJECT_ROOT / "generated_branescripts").resolve()
LANG_DB_PATH = PROJECT_ROOT / "brane_lang_db"
PKG_DB_PATH = PROJECT_ROOT / "brane_pkg_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Main generation prompt
# {no_think_prefix} must be the very first token — Qwen3 only acts on /no_think
# if it appears before any other text.
GENERATION_TEMPLATE = """{no_think_prefix}You are an expert in the Brane Framework and BraneScript.
{few_shot}
## ABSOLUTE RULES — READ CAREFULLY
1. Output ONLY valid BraneScript code.
2. BraneScript is NOT Python, Java, Rust, or any other language. Do NOT output code in any other language.
3. Do NOT use `def`, `class`, `from X import Y`, `self.`, or any Python/Java syntax.
4. Do NOT wrap output in markdown code fences (no ```bscript, no ```python, no ``` of any kind).
5. Do NOT add prose, explanations, headers, or narrative — ONLY code.
6. Use exact BraneScript assignment syntax: `let <name> := <expression>;` — NEVER use `=` alone.
7. Do NOT invent packages or functions not present in the PACKAGE / DATASET CONTEXT below.
8. Define every variable with `let` before using it.
9. If the user mentions a node, site, or location name (e.g. "on node marco", "on site Amy"), place `#[on("name")]` immediately before the relevant function call or block.
10. If context is incomplete, ask ONE clarifying question — do not generate any code.

USER REQUEST:
{question}

SUBTASKS:
{subtasks}

LANGUAGE SPEC CONTEXT:
{lang_context}

PACKAGE / DATASET CONTEXT:
{pkg_context}

{error_section}BRANESCRIPT CODE (output raw BraneScript only, no fences, no prose):"""

# Error sections injected on retry
SYNTAX_ERROR_SECTION = """⚠️  PREVIOUS ATTEMPT HAD A SYNTAX ERROR:
{error}
Fix the syntax error. Output ONLY corrected BraneScript — no prose, no fences.

"""

NON_CODE_ERROR_SECTION = """⚠️  PREVIOUS ATTEMPT DID NOT RETURN VALID BRANESCRIPT.
Output ONLY valid BraneScript code. No Python. No explanations. No markdown fences.
If you cannot produce code, ask a single clarifying question.

"""

PYTHON_CODE_ERROR_SECTION = """⚠️  PREVIOUS ATTEMPT GENERATED PYTHON CODE — THAT IS WRONG.
You MUST output BraneScript, NOT Python.

CORRECT BraneScript example:
    import healthcare;
    let patient := "{..json..}";
    let result := healthcare::analyze_heart_disease(patient);
    println(result);

Do NOT use: def, class, import os, import sys, self., or ANY Python syntax.
Output ONLY BraneScript code. No prose. No fences.

"""

# ---------------------------------------------------------------------------
# Output cleanup helpers
# ---------------------------------------------------------------------------

def strip_thinking_tokens(text: str) -> str:
    """Remove Qwen3/DeepSeek-style <think>...</think> reasoning blocks."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def strip_code_fences(text: str) -> str:
    """
    If the model wrapped its output in a markdown code fence, extract the
    code inside.  Works for ```bscript, ```branescript, ```bs, or plain ```.
    """
    # Try labelled fences first (bscript / branescript / bs)
    match = re.search(
        r'```(?:bscript|branescript|bs)\s*\n(.*?)```',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # Fall back to any fenced code block
    match = re.search(r'```(?:\w*)\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


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
# Filename helpers
# ---------------------------------------------------------------------------
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


def looks_like_branescript(text: str) -> bool:
    code = text.strip()
    if not code:
        return False

    if "let " in code and ":=" in code:
        return True
    if re.search(r"import\s+[A-Za-z]", code):
        return True
    if "::" in code:
        return True
    if code.startswith("package") or code.startswith("workflow"):
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
# Semantic check
# ---------------------------------------------------------------------------
def check_semantic(code: str, pkg_context: str) -> tuple[bool, str]:
    """
    Check that package and dataset names used in the code appear
    in the retrieved pkg_context.

    Returns (is_valid, error_message).
    """
    errors = []

    import_names = re.findall(
        r'import\s+([A-Za-z][A-Za-z0-9_\-]*)', code, re.IGNORECASE
    )
    pkg_calls = re.findall(
        r'([A-Za-z][A-Za-z0-9_\-]+)\s*::', code
    )

    referenced = set(import_names + pkg_calls)

    for name in referenced:
        if name.lower() in ("std", "io", "math"):
            continue
        if name not in pkg_context:
            errors.append(
                f"'{name}' is referenced in the code but not found in the package/dataset context."
            )

    if errors:
        return False, "\n".join(errors)
    return True, ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(user_query: str,
                 decomposer: IntentDecomposer,
                 pkg_retriever: PkgRetriever,
                 llm: Ollama,
                 few_shot_override: str = None,
                 no_think_prefix: str = "") -> str:

    few_shot = few_shot_override if few_shot_override is not None else BRANESCRIPT_FEW_SHOT

    # ── Step 1: Task breakdown + language spec retrieval ──────────────────
    lang_context, subtasks = decomposer.run(user_query)
    subtasks_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(subtasks))

    # ── Step 2: Package / dataset retrieval ───────────────────────────────
    pkg_context = pkg_retriever.run(subtasks, user_query)
    print("\n📦 Retrieved package/dataset context:")
    if pkg_context.strip():
        print(pkg_context[:1000] + ("..." if len(pkg_context) > 1000 else ""))
    else:
        print("   - (No package/dataset context returned)")

    prompt = ChatPromptTemplate.from_template(GENERATION_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    error_section = ""
    code = ""
    attempt = 1

    while attempt <= MAX_RETRIES:
        print(f"\n⚙️  Generation attempt {attempt}/{MAX_RETRIES}...")

        raw = chain.invoke({
            "question": user_query,
            "subtasks": subtasks_str,
            "lang_context": lang_context,
            "pkg_context": pkg_context,
            "few_shot": few_shot,
            "error_section": error_section,
            "no_think_prefix": no_think_prefix,
        })

        # Always strip thinking tokens and code fences — harmless on clean output
        print(f"   🔎 Raw LLM output ({len(raw)} chars): {repr(raw[:200])}")
        code = strip_thinking_tokens(raw)
        code = strip_code_fences(code)

        if is_clarification_request(code):
            answer = ask_for_clarification(code)
            if not answer:
                print("   ⚠️ No clarification answer provided. Aborting.")
                return code
            # Append clarification Q&A to the user query so the model has full context
            user_query = user_query + "\n\nClarification — " + code.strip() + "\nAnswer: " + answer
            error_section = ""
            attempt += 1
            continue

        if not looks_like_branescript(code):
            print("   ❌ Output does not look like BraneScript code.")
            print(f"   Output was:\n{code}\n")
            if attempt < MAX_RETRIES:
                error_section = NON_CODE_ERROR_SECTION
                attempt += 1
                continue
            else:
                print("   ⛔ Max retries reached on code validation. Returning last attempt.")
                return code

        syntax_ok, syntax_error = check_syntax(code)
        if not syntax_ok:
            print(f"   ❌ Syntax check failed: {syntax_error}")
            if attempt < MAX_RETRIES:
                error_section = SYNTAX_ERROR_SECTION.format(error=syntax_error)
                attempt += 1
                continue
            else:
                print("   ⛔ Max retries reached on syntax. Returning last attempt.")
                return code

        semantic_ok, semantic_error = check_semantic(code, pkg_context)
        if not semantic_ok:
            print(f"   ❌ Semantic check failed: {semantic_error}")
            if attempt < MAX_RETRIES:
                error_section = NON_CODE_ERROR_SECTION + "\n" + semantic_error + "\n"
                attempt += 1
                continue
            else:
                print("   ⛔ Max retries reached on semantic validation. Returning last attempt.")
                return code

        print("   ✅ Validation passed")
        break

    if code.strip():
        save_branescript_to_folder(code, user_query)

    return code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLP → BraneScript pipeline")
    parser.add_argument(
        "--model", default="qwen3.5:4b",
        help="Ollama model name (default: qwen2.5-coder:7b). "
             "Use 'qwen2.5-coder:3b' for the fastest option, "
             "'qwen3.5' for the larger model."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
        help="Sampling temperature (default: 0.4). "
             "Lower = more deterministic code output."
    )
    args = parser.parse_args()

    # Disable Qwen3 thinking mode — /no_think must be the very first token of the prompt
    # so it is injected via no_think_prefix into the template, NOT into the few_shot block.
    is_qwen3 = "qwen3" in args.model.lower()
    no_think_prefix = ""
    if is_qwen3:
        print(f"ℹ️  Qwen3 detected — injecting /no_think to disable reasoning mode")
        print(f" Chosen model: {args.model}")
        no_think_prefix = "/no_think\n"

    print(f"🔧 Initialising — model: {args.model}  temperature: {args.temperature}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    llm = Ollama(
        model=args.model,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    lang_db = Chroma(persist_directory=str(LANG_DB_PATH), embedding_function=embeddings)

    if os.path.exists(PKG_DB_PATH):
        print(f"✅ Found package DB at {PKG_DB_PATH}. Loading...")
        pkg_db = Chroma(persist_directory=str(PKG_DB_PATH), embedding_function=embeddings)
    else:
        print(f"⚠️  Package DB not found at {PKG_DB_PATH}. Using language DB as fallback.")
        pkg_db = lang_db

    decomposer = IntentDecomposer(llm=llm, lang_db=lang_db, k_per_subtask=3,
                                   no_think=is_qwen3)
    pkg_retriever = PkgRetriever(pkg_db=pkg_db, k=4)

    user_query = input("Enter the user request: ").strip()
    if not user_query:
        print("No request provided. Exiting.")
        exit(0)

    print(f"\n🧠 User request: {user_query}")
    print("⏳ Running pipeline...\n" + "─" * 50)

    result = run_pipeline(
        user_query=user_query,
        decomposer=decomposer,
        pkg_retriever=pkg_retriever,
        llm=llm,
        few_shot_override=BRANESCRIPT_FEW_SHOT,
        no_think_prefix=no_think_prefix,
    )

    print("\n" + "=" * 50)
    print("FINAL BRANESCRIPT OUTPUT:")
    print("=" * 50)
    print(result)