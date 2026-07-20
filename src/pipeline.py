"""
pipeline.py

Main entry point. Implements the simplified architecture:

  User request
    → Task breakdown + language spec retrieval (IntentDecomposer)
    → Package/dataset retrieval (PkgRetriever)
    → LLM inference (BraneScript generation)
    → Syntax/semantic check  (on fail: retry with error, max 3 attempts)
    → Save generated BraneScript to disk
    → [Optional] Submit to file-based job queue → run via job_watcher.py

Run:
    python pipeline.py [--execute]

Remote execution
    When --execute is passed, the generated BraneScript is written to
    ~/brane_jobs/pending/ on the Snellius filesystem and job_watcher.py
    on your local machine picks it up, runs it, and returns the result.
    See: scripts/remote_execution/README.md
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from intent_decomposer import IntentDecomposer
from pkg_retriever import PkgRetriever
from training_collector import TrainingCollector
from utils import strip_thinking_tokens, strip_code_fences, looks_like_branescript, detect_python_code, detect_json_string_assignment, load_hf_token
from prompts import GENERATION_SYSTEM_TEMPLATE, build_user_message, load_system_prompt

# Load HuggingFace token early so embeddings download authenticated
load_hf_token()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
BRANESCRIPT_OUTPUT_DIR = (PROJECT_ROOT / "generated_branescripts").resolve()
PKG_DB_PATH = PROJECT_ROOT / "brane_pkg_db"
SYNTAX_REFERENCE_PATH = PROJECT_ROOT / "data" / "syntax_reference.md"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RETRIES = 3

# Remote execution — file-based job queue on Snellius filesystem.
# job_watcher.py on the local machine polls this directory via SSH.
SNELLIUS_JOBS_DIR = os.path.expanduser(
    os.environ.get("SNELLIUS_JOBS_DIR", "~/brane_jobs")
)
EXECUTOR_TIMEOUT = int(os.environ.get("BRANE_EXECUTOR_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# Prompt templates — defined in src/prompts.py, imported above
# ---------------------------------------------------------------------------

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
    import compute;

    class Config {
        iterations: int;
        threshold:  int;
    }

    let cfg := new Config {
        iterations := 100,
        threshold  := 50,
    };

    let result := run(cfg);
    println(result);

Do NOT use: def, from X import Y, or ANY Python syntax.
Output ONLY BraneScript code. No prose. No fences.

"""

JSON_STRING_ERROR_SECTION = """⚠️  PREVIOUS ATTEMPT PASSED STRUCTURED DATA AS A JSON STRING — THAT IS WRONG.

You wrote something like:
    let p := "{\\"iterations\\": 100, \\"threshold\\": 50}";  ← WRONG

NEVER use backslash-escaped quotes (\\") inside string literals.
NEVER serialize structured data as a JSON string.

Instead, define a BraneScript `class` for each data type and instantiate it:

CORRECT:
    class Config {
        iterations: int;
        threshold:  int;
    }

    let cfg := new Config {
        iterations := 100,
        threshold  := 50,
    };

    let result := run(cfg);

Output ONLY BraneScript code. No prose. No fences.

"""

COMPILATION_ERROR_SECTION = """⚠️  THE PREVIOUSLY GENERATED BRANESCRIPT FAILED TO COMPILE:
{stderr}
The Brane runtime rejected the script above. Read the error message carefully.
Common causes:
- Calling a function that does not exist in the imported package — check the PACKAGE CONTEXT for exact function names.
- Wrong argument types or missing required arguments.
- Syntax the pre-check missed (wrong operator, wrong keyword, missing semicolon).

Fix the BraneScript so it compiles. Output ONLY corrected BraneScript — no prose, no fences.

"""

RUNTIME_ERROR_SECTION = """⚠️  THE PREVIOUSLY GENERATED BRANESCRIPT COMPILED BUT FAILED AT RUNTIME:
stdout:
{stdout}

stderr:
{stderr}

The workflow started but a task container exited with an error. Common causes:
- Passing wrong values (out-of-range, wrong type, missing required field) to a package function.
- Using a dataset name that does not exist — check the PACKAGE / DATASET CONTEXT for the exact registered name.
- Logic error causing an unhandled exception inside the package.

Fix the BraneScript to correct the error. Output ONLY corrected BraneScript — no prose, no fences.

"""

# ---------------------------------------------------------------------------
# Output cleanup helpers — imported from utils.py
# strip_thinking_tokens, strip_code_fences, detect_python_code, looks_like_branescript
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LLM call helper — chat template with thinking disabled
# ---------------------------------------------------------------------------
def _call_llm(
    text_gen_pipeline,
    tokenizer,
    system_msg: str,
    user_msg: str,
) -> str:
    """
    Format a system+user chat message using the tokenizer's chat template,
    with thinking disabled (enable_thinking=False for Qwen3).  Falls back
    gracefully if the tokenizer does not support that parameter.
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older transformers build — no enable_thinking support
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    result = text_gen_pipeline(prompt_text)
    return result[0]["generated_text"]


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


# looks_like_branescript imported from utils.py


def ask_for_clarification(question: str) -> str:
    print("\n❓ The model is asking for clarification:")
    print(question.strip())
    answer = input("Your answer: ")
    return answer.strip()


def execute_workflow(code: str, user_query: str = "",
                     model_name: str = "", attempt_number: int = 1) -> dict:
    """
    Submit the BraneScript to the file-based job queue on Snellius and
    block until job_watcher.py on the local machine returns a result.

    No port forwarding needed — writes to ~/brane_jobs/pending/ on the
    Snellius filesystem and polls ~/brane_jobs/done/ for the result.

    Returns a dict with keys:
        success     bool
        exit_code   int | None
        error_type  str | None  — "compilation", "runtime", "timeout", or None
        stdout      str
        stderr      str
    """
    import time as _time
    import uuid as _uuid

    jobs_dir = os.path.expanduser(
        os.environ.get("SNELLIUS_JOBS_DIR", "~/brane_jobs")
    )
    pending_dir = os.path.join(jobs_dir, "pending")
    done_dir = os.path.join(jobs_dir, "done")
    os.makedirs(pending_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)

    job_id = str(_uuid.uuid4())
    job = {
        "id": job_id,
        "workflow": code,
        "query": user_query,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "attempt_number": attempt_number,
    }

    job_path = os.path.join(pending_dir, f"{job_id}.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f)

    print(f"\n🚀 Job submitted: {job_id[:8]}…")
    print(f"   Waiting for job_watcher.py on your local machine to pick it up...")

    result_path = os.path.join(done_dir, f"{job_id}.json")
    poll_interval = 3
    waited = 0

    while waited < EXECUTOR_TIMEOUT:
        if os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as f:
                result = json.load(f)
            try:
                os.unlink(result_path)
            except OSError:
                pass

            success: bool = result.get("success", False)
            stdout: str = result.get("stdout", "")
            stderr: str = result.get("stderr", "")
            exit_code: int = result.get("exit_code", -1)
            error_type: Optional[str] = result.get("error_type")

            print(f"   exit_code={exit_code}  success={success}")
            if stdout:
                print("\n── Workflow stdout ──────────────────────────────────")
                print(stdout.rstrip())
                print("─────────────────────────────────────────────────────")
            if stderr and not success:
                print("\n── Workflow stderr ──────────────────────────────────")
                print(stderr.rstrip())
                print("─────────────────────────────────────────────────────")

            return {
                "success": success,
                "exit_code": exit_code,
                "error_type": error_type,
                "stdout": stdout,
                "stderr": stderr,
                "committed_data": result.get("committed_data", {}),
            }

        _time.sleep(poll_interval)
        waited += poll_interval

    # Timeout — clean up pending job
    try:
        os.unlink(job_path)
    except OSError:
        pass
    msg = f"Timed out after {EXECUTOR_TIMEOUT}s waiting for local executor."
    print(f"\n⚠️  {msg}")
    print("   Is job_watcher.py running on your local machine?")
    return {
        "success": False,
        "exit_code": -1,
        "error_type": "timeout",
        "stdout": "",
        "stderr": msg,
        "committed_data": {},
    }


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
    referenced = set(import_names)

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
                 decomposer,
                 pkg_retriever: PkgRetriever,
                 text_gen_pipeline,
                 tokenizer,
                 syntax_reference: str,
                 execute: bool = False,
                 collector: Optional[TrainingCollector] = None,
                 model_name: str = "") -> str:

    t_start = time.time()

    # Pre-build the system message — constant for all attempts of this query
    system_msg = GENERATION_SYSTEM_TEMPLATE.format(
        lang_context=syntax_reference,
    )

    # ── Step 1: Task breakdown ─────────────────────────────────────────────
    t0 = time.time()
    subtasks = decomposer.decompose(user_query)
    subtasks_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(subtasks))
    t_decompose = time.time() - t0

    # ── Step 2: Package / dataset retrieval ───────────────────────────────
    t0 = time.time()
    t0 = time.time()
    pkg_context = pkg_retriever.run(subtasks, user_query)
    t_retrieval = time.time() - t0
    print("\n📦 Retrieved package/dataset context:")
    if pkg_context.strip():
        print(pkg_context[:1000] + ("..." if len(pkg_context) > 1000 else ""))
    else:
        print("   - (No package/dataset context returned)")

    error_section = ""
    code = ""
    attempt = 1
    t_generation_total = 0.0
    t_execution_total = 0.0

    def _timing_snapshot() -> dict:
        """Return timing accumulated so far (partial during retries)."""
        return {
            "decompose_s":  round(t_decompose, 2),
            "retrieval_s":  round(t_retrieval, 2),
            "generation_s": round(t_generation_total, 2),
            "execution_s":  round(t_execution_total, 2),
            "total_s":      round(time.time() - t_start, 2),
            "attempts":     attempt,
        }

    while attempt <= MAX_RETRIES:
        print(f"\n⚙️  Generation attempt {attempt}/{MAX_RETRIES}...")

        user_msg = build_user_message(
            question=user_query,
            pkg_context=pkg_context,
            subtasks=subtasks_str,
            error_section=error_section,
        )
        print(f"   📏 Prompt length: {len(system_msg) + len(user_msg)} chars")

        t0 = time.time()
        raw = _call_llm(text_gen_pipeline, tokenizer, system_msg, user_msg)
        t_generation_total += time.time() - t0

        # strip_thinking_tokens is still called as a safety net (harmless when thinking is off)
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
            if detect_python_code(code):
                print("   ❌ Model generated Python code instead of BraneScript.")
                print(f"   Output was:\n{code[:300]}\n")
                if collector:
                    collector.log_fail(intent=user_query, generated_code=code,
                                       error_type="python_code",
                                       error_message="Model generated Python instead of BraneScript",
                                       attempt_number=attempt, model=model_name,
                                       timing=_timing_snapshot())
                if attempt < MAX_RETRIES:
                    error_section = PYTHON_CODE_ERROR_SECTION
                    attempt += 1
                    continue
            else:
                print("   ❌ Output does not look like BraneScript code.")
                print(f"   Output was:\n{code[:300]}\n")
                if collector:
                    collector.log_fail(intent=user_query, generated_code=code,
                                       error_type="non_code",
                                       error_message="Output does not look like BraneScript",
                                       attempt_number=attempt, model=model_name,
                                       timing=_timing_snapshot())
                if attempt < MAX_RETRIES:
                    error_section = NON_CODE_ERROR_SECTION
                    attempt += 1
                    continue
            print("   ⛔ Max retries reached on code validation. Returning last attempt.")
            return code

        # Catch the JSON-as-string antipattern: let x := "{\"key\": \"val\"}";
        if detect_json_string_assignment(code):
            print("   ❌ Code passes structured data as a JSON string with escaped quotes.")
            print(f"   Output was:\n{code[:300]}\n")
            if collector:
                collector.log_fail(intent=user_query, generated_code=code,
                                   error_type="json_string",
                                   error_message="Escaped JSON string used instead of class instantiation",
                                   attempt_number=attempt, model=model_name,
                                   timing=_timing_snapshot())
            if attempt < MAX_RETRIES:
                error_section = JSON_STRING_ERROR_SECTION
                attempt += 1
                continue
            else:
                print("   ⛔ Max retries reached on JSON-string check. Returning last attempt.")
                return code

        syntax_ok, syntax_error = check_syntax(code)
        if not syntax_ok:
            print(f"   ❌ Syntax check failed: {syntax_error}")
            if collector:
                collector.log_fail(intent=user_query, generated_code=code,
                                   error_type="syntax", error_message=syntax_error,
                                   attempt_number=attempt, model=model_name,
                                   timing=_timing_snapshot())
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
            if collector:
                collector.log_fail(intent=user_query, generated_code=code,
                                   error_type="semantic", error_message=semantic_error,
                                   attempt_number=attempt, model=model_name,
                                   timing=_timing_snapshot())
            if attempt < MAX_RETRIES:
                error_section = NON_CODE_ERROR_SECTION + "\n" + semantic_error + "\n"
                attempt += 1
                continue
            else:
                print("   ⛔ Max retries reached on semantic validation. Returning last attempt.")
                return code

        print("   ✅ Validation passed")

        # ── No execution requested — we are done ──────────────────────────
        if not execute:
            break

        # ── Execute and retry on failure ──────────────────────────────────
        t0 = time.time()
        exec_result = execute_workflow(code, user_query,
                                       model_name=model_name,
                                       attempt_number=attempt)
        t_execution_total += time.time() - t0
        if exec_result["success"]:
            print(f"\n✅ Workflow executed successfully.")
            if collector:
                collector.log_pass(intent=user_query, generated_code=code,
                                   stdout=exec_result["stdout"],
                                   committed_data=exec_result.get("committed_data"),
                                   execution_result=exec_result,
                                   attempt_number=attempt, model=model_name,
                                   timing=_timing_snapshot())
            break

        # Execution failed — log attempt and prepare retry
        error_type = exec_result.get("error_type") or "runtime"
        stderr = exec_result.get("stderr", "")
        stdout = exec_result.get("stdout", "")
        exit_code = exec_result.get("exit_code", -1)

        print(f"\n⚠️  Workflow execution failed [{error_type}] (attempt {attempt}/{MAX_RETRIES}).")
        if stderr:
            print(f"   stderr: {stderr[:300]}")

        if collector:
            collector.log_fail(intent=user_query, generated_code=code,
                               error_type=error_type,
                               error_message=stderr[:500],
                               stdout=stdout,
                               stderr=stderr,
                               exit_code=exit_code,
                               committed_data=exec_result.get("committed_data"),
                               execution_result=exec_result,
                               attempt_number=attempt, model=model_name,
                               timing=_timing_snapshot())

        if attempt < MAX_RETRIES:
            if error_type == "compilation":
                error_section = COMPILATION_ERROR_SECTION.format(
                    stderr=(stderr[:800] if stderr.strip() else "(no stderr output)")
                )
            else:
                error_section = RUNTIME_ERROR_SECTION.format(
                    stdout=(stdout[:400] if stdout.strip() else "(no stdout output)"),
                    stderr=(stderr[:400] if stderr.strip() else "(no stderr output)"),
                )
            attempt += 1
            continue
        else:
            print("   ⛔ Max retries reached on execution. Returning last attempt.")
            break

    if code.strip():
        save_branescript_to_folder(code, user_query)

    t_total = time.time() - t_start
    timing = {
        "decompose_s":   round(t_decompose, 2),
        "retrieval_s":   round(t_retrieval, 2),
        "generation_s":  round(t_generation_total, 2),
        "execution_s":   round(t_execution_total, 2),
        "total_s":       round(t_total, 2),
        "attempts":      attempt,
    }

    # No-execute path: log a validation-only pass
    if not execute and code.strip() and collector:
        collector.log_pass(intent=user_query, generated_code=code,
                           attempt_number=attempt, model=model_name,
                           timing=timing)

    print(f"\n⏱️  Timing summary:")
    print(f"   decompose   : {t_decompose:6.1f}s")
    print(f"   retrieval   : {t_retrieval:6.1f}s")
    print(f"   generation  : {t_generation_total:6.1f}s  ({attempt} attempt(s))")
    if execute:
        print(f"   execution   : {t_execution_total:6.1f}s")
    print(f"   ─────────────────────")
    print(f"   total       : {t_total:6.1f}s")

    return code


# ---------------------------------------------------------------------------
# Pipeline component setup (shared by pipeline.py and evaluate.py)
# ---------------------------------------------------------------------------
def build_pipeline_components(model_id: str, temperature: float) -> dict:
    """
    Load all pipeline components and return them as a dict.

    Returns
    -------
    {
        "text_gen_pipeline": raw HuggingFace text-generation pipeline,
        "tokenizer":         AutoTokenizer (needed for apply_chat_template),
        "decomposer":        IntentDecomposer,
        "pkg_retriever":     PkgRetriever,
        "syntax_reference":  str,
    }
    """
    print(f"📖 Loading syntax reference from {SYNTAX_REFERENCE_PATH}...")
    if not SYNTAX_REFERENCE_PATH.exists():
        raise FileNotFoundError(f"syntax_reference.md not found at {SYNTAX_REFERENCE_PATH}")
    syntax_reference = SYNTAX_REFERENCE_PATH.read_text(encoding="utf-8")
    print(f"   → {len(syntax_reference)} chars loaded")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.clean_up_tokenization_spaces = False

    print("📥 Loading model onto GPU...")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print("✅ Model loaded")

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=1024,       # safe now that thinking is disabled
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
    )

    if not PKG_DB_PATH.exists():
        raise FileNotFoundError(
            f"Package DB not found at {PKG_DB_PATH}. "
            "Run `python src/knowledgeBase.py` first to build the knowledge base."
        )
    print(f"✅ Found package DB at {PKG_DB_PATH}. Loading...")
    pkg_db = Chroma(persist_directory=str(PKG_DB_PATH), embedding_function=embeddings)

    decomposer = IntentDecomposer(text_gen_pipeline=text_gen_pipeline, tokenizer=tokenizer)
    pkg_retriever = PkgRetriever(pkg_db=pkg_db, k=4)

    return {
        "text_gen_pipeline": text_gen_pipeline,
        "tokenizer":         tokenizer,
        "decomposer":        decomposer,
        "pkg_retriever":     pkg_retriever,
        "syntax_reference":  syntax_reference,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLP → BraneScript pipeline")
    parser.add_argument(
        "--model", default="Qwen/Qwen3.6-9B",
        help="Hugging Face model id, e.g. Qwen/Qwen3.6-27B, Qwen/Qwen3-4B."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
        help="Sampling temperature (default: 0.4). Lower = more deterministic."
    )
    parser.add_argument(
        "--execute", action="store_true", default=False,
        help="Submit the generated BraneScript to the file-based job queue on Snellius "
             "(~/brane_jobs/pending/) and wait for job_watcher.py on your local machine "
             "to execute it. See scripts/remote_execution/README.md."
    )
    parser.add_argument(
        "--collect", action="store_true", default=False,
        help="Enable training data collection. Every generation attempt (pass and fail) "
             "is stored in TRAINING_DATA_DIR/runs/ (default: <project>/data/training). "
             "Set the TRAINING_DATA_DIR env var to override the directory."
    )
    parser.add_argument(
        "--query",
        help="Single natural-language intent to process. "
             "Mutually exclusive with --intents-file."
    )
    parser.add_argument(
        "--intents-file",
        dest="intents_file",
        help="Path to a plain-text file with one intent per line. "
             "Each intent is processed in sequence. "
             "Mutually exclusive with --query."
    )
    args = parser.parse_args()

    if args.query and args.intents_file:
        parser.error("--query and --intents-file are mutually exclusive.")

    print(f"🔧 Initialising — model: {args.model}  temperature: {args.temperature}  "
          f"execute: {args.execute}  collect: {args.collect}")

    components = build_pipeline_components(args.model, args.temperature)
    text_gen_pipeline = components["text_gen_pipeline"]
    tokenizer        = components["tokenizer"]
    decomposer       = components["decomposer"]
    pkg_retriever    = components["pkg_retriever"]
    syntax_reference = components["syntax_reference"]

    collector = None
    if args.collect:
        collector = TrainingCollector()
        print(f"📊 Training data collection enabled → {collector.runs_dir}")

    # ── Resolve intent list ──────────────────────────────────────────────────
    if args.intents_file:
        with open(args.intents_file, encoding="utf-8") as f:
            queries = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        print(f"📋 Loaded {len(queries)} intents from {args.intents_file}")
    elif args.query:
        queries = [args.query]
    else:
        # Fallback default intent for quick testing
        queries = [
            "Analyze the heart disease risk for a patient with the following profile: "
            "65-year-old female, blood pressure 170, heart rate 80, total cholesterol 230, "
            "medical history of hypertension. Run the analysis on node giovanni. Commit the result as a new dataset named 'patient_001_risk_analysis'."
        ]
        print("ℹ️  No --query or --intents-file provided. Using built-in default intent.")

    # ── Run pipeline for each intent ─────────────────────────────────────────
    for i, user_query in enumerate(queries, 1):
        if len(queries) > 1:
            print(f"\n{'='*50}")
            print(f"Intent {i}/{len(queries)}: {user_query[:80]}...")
            print('='*50)
        else:
            print(f"\n🧠 User request: {user_query}")

        print("⏳ Running pipeline...\n" + "─" * 50)

        t_intent_start = time.time()
        result = run_pipeline(
            user_query=user_query,
            decomposer=decomposer,
            pkg_retriever=pkg_retriever,
            text_gen_pipeline=text_gen_pipeline,
            tokenizer=tokenizer,
            syntax_reference=syntax_reference,
            execute=args.execute,
            collector=collector,
            model_name=args.model,
        )
        t_intent = time.time() - t_intent_start

        print("\n" + "=" * 50)
        print("FINAL BRANESCRIPT OUTPUT:")
        print("=" * 50)
        print(result)
        if len(queries) > 1:
            print(f"\n⏱️  Intent {i}/{len(queries)} wall time: {t_intent:.1f}s")

    if collector:
        stats = collector.stats()
        print(f"\n📊 Training data: {stats['total']} records "
              f"({stats['passes']} pass / {stats['fails']} fail) → {stats['runs_dir']}")

    print("\n👌 Pipeline execution completed.")