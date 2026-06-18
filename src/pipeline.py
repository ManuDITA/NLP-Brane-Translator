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
from datetime import datetime, timezone
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from intent_decomposer import IntentDecomposer
from pkg_retriever import PkgRetriever
from utils import strip_thinking_tokens, strip_code_fences, looks_like_branescript, detect_python_code, detect_json_string_assignment, load_hf_token

# Load HuggingFace token early so embeddings download authenticated
load_hf_token()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# ---------------------------------------------------------------------------
# BraneScript canonical few-shot examples injected into every prompt.
# These are intentionally domain-neutral — they teach syntax, not a specific
# package.  Real package/function names come from RAG retrieval at runtime.
# ---------------------------------------------------------------------------
BRANESCRIPT_FEW_SHOT = """
## BRANESCRIPT SYNTAX REFERENCE (few-shot examples)

Example 1 – import a package, define classes for structured input, call a function:
```
import compute;

class Config {
    iterations: int;
    threshold:  int;
}

let cfg := new Config {
    iterations := 100,
    threshold  := 50,
};

let result := compute::run(cfg);
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

Example 4 – nested classes, routing execution to a named node/site with #[on("name")]:
```
import analytics;

class Config {
    threshold:  int;
    iterations: int;
}

class Job {
    id:     string;
    config: Config;
}

let cfg := new Config {
    threshold  := 50,
    iterations := 100,
};

let job := new Job {
    id     := "job-001",
    config := cfg,
};

#[on("marco")]
let result := analytics::process(job);
println(result);
```

Example 5 – deeply nested classes (three levels), single package call, node routing:
```
import bioanalysis;

class Measurements {
    pressure: int;
    rate:     int;
}

class Labs {
    cholesterol: int;
}

class Subject {
    id:           string;
    age:          int;
    measurements: Measurements;
    labs:         Labs;
}

let m := new Measurements {
    pressure := 150,
    rate     := 75,
};

let l := new Labs {
    cholesterol := 210,
};

let subject := new Subject {
    id           := "S001",
    age          := 58,
    measurements := m,
    labs         := l,
};

#[on("amy")]
    let result := bioanalysis::evaluate(subject);
println(result);
```
NOTE: whenever the user says "on node X", "on site X", "at location X", or "run on X",
place `#[on("X")]` immediately before the relevant function call or block.

Example 6 – referencing a registered dataset (Data), getting an IntermediateResult, and committing it:
```
import copy_result;
import cat;

// Reference an existing dataset by name — Data is a builtin class with one field: name
let raw := new Data { name := "patient_records" };

// Pass the Data reference to a package function.
// Functions that output file/data results return an IntermediateResult (not a plain value).
// You cannot create an IntermediateResult yourself — it is always returned by a package function.
let processed := copy_result(raw);

// To persist the result beyond this workflow, commit it with a new name.
// commit_result("new-dataset-name", intermediate_result_variable)
commit_result("patient_records_copy", processed);
```
RULES for Data and IntermediateResult:
- Use `new Data {{ name := "dataset-name" }}` to reference a registered dataset.
- `IntermediateResult` is returned by package functions — you NEVER instantiate it yourself.
- Use `commit_result("name", result)` to save an IntermediateResult as a persistent dataset.
- Both `Data` and `IntermediateResult` can be passed as arguments to package functions.
- Do NOT try to access fields or content of a Data/IntermediateResult in BraneScript — they are opaque references.
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

# Remote execution — file-based job queue on Snellius filesystem.
# job_watcher.py on the local machine polls this directory via SSH.
SNELLIUS_JOBS_DIR = os.path.expanduser(
    os.environ.get("SNELLIUS_JOBS_DIR", "~/brane_jobs")
)
EXECUTOR_TIMEOUT = int(os.environ.get("BRANE_EXECUTOR_TIMEOUT", "300"))

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
3. Do NOT use `def`, `from X import Y`, `self.` (outside a class method), or any Python/Java syntax.
4. Do NOT wrap output in markdown code fences (no ```bscript, no ```python, no ``` of any kind).
5. Do NOT add prose, explanations, headers, or narrative — ONLY code.
6. Use exact BraneScript assignment syntax: `let <name> := <expression>;` — NEVER use `=` alone.
7. Do NOT invent packages or functions not present in the PACKAGE / DATASET CONTEXT below.
8. Define every variable with `let` before using it.
9. If the user mentions a node, site, or location name (e.g. "on node marco", "on site Amy"), place `#[on("name")]` immediately before the relevant function call or block.
10. If context is incomplete, ask ONE clarifying question — do not generate any code.
11. For complex structured data with multiple fields, define a BraneScript `class` for each data type, instantiate with `new <ClassName> {{ field := value, ... }}`, and pass the instance to the function. Do NOT represent structured data as a raw JSON string with escaped quotes.
12. NEVER use backslash-escaped quotes (like `\"`) anywhere in your output. If you need to pass structured data, define a class and use `new ClassName {{ ... }}`. Outputting `let x := "{{\\"key\\": \\"val\\"}}"` is always wrong.
13. Do NOT re-implement logic that the package function already handles internally. Your job is to define the input data, call the package function, and print the result. Do NOT manually compute scores, risk levels, or any derived values that the function returns.
14. BraneScript class fields can only have primitive types (`int`, `real`, `bool`, `string`) or other class types. Do NOT use `array<T>`, `list<T>`, or `List` as field types — these do not exist. If a field would be a list, either omit it or represent it as a `string`.
15. To reference a registered dataset, use `let ds := new Data {{ name := "dataset-name" }};` and pass `ds` to the package function. Do NOT pass the dataset name as a plain string.
16. Package functions that output data/files return an `IntermediateResult`. You CANNOT create an `IntermediateResult` yourself. If the user wants to save or persist output data, use `commit_result("new-name", result_variable);` after calling the function.
17. Do NOT attempt to access fields or inspect the content of a `Data` or `IntermediateResult` value in BraneScript — they are opaque references handled by the framework.

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
    import compute;

    class Config {
        iterations: int;
        threshold:  int;
    }

    let cfg := new Config {
        iterations := 100,
        threshold  := 50,
    };

    let result := compute::run(cfg);
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

    let result := mypackage::run(cfg);

Output ONLY BraneScript code. No prose. No fences.

"""

# ---------------------------------------------------------------------------
# Output cleanup helpers — imported from utils.py
# strip_thinking_tokens, strip_code_fences, detect_python_code, looks_like_branescript
# ---------------------------------------------------------------------------


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


def execute_workflow(code: str, user_query: str = "") -> tuple[bool, str]:
    """
    Submit the BraneScript to the file-based job queue on Snellius and
    block until job_watcher.py on the local machine returns a result.

    No port forwarding needed — writes to ~/brane_jobs/pending/ on the
    Snellius filesystem and polls ~/brane_jobs/done/ for the result.
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
            print(f"   exit_code={result.get('exit_code', '?')}  success={success}")
            if stdout:
                print("\n── Workflow stdout ──────────────────────────────────")
                print(stdout.rstrip())
                print("─────────────────────────────────────────────────────")
            if stderr and not success:
                print("\n── Workflow stderr ──────────────────────────────────")
                print(stderr.rstrip())
                print("─────────────────────────────────────────────────────")
            return success, stdout + (f"\n[stderr]\n{stderr}" if stderr else "")

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
    return False, msg


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
                 llm,
                 few_shot_override: str = None,
                 no_think_prefix: str = "",
                 execute: bool = False) -> str:

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
            if detect_python_code(code):
                print("   ❌ Model generated Python code instead of BraneScript.")
                print(f"   Output was:\n{code[:300]}\n")
                if attempt < MAX_RETRIES:
                    error_section = PYTHON_CODE_ERROR_SECTION
                    attempt += 1
                    continue
            else:
                print("   ❌ Output does not look like BraneScript code.")
                print(f"   Output was:\n{code[:300]}\n")
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

    # ── Optional: execute on local machine via SSH tunnel ─────────────────
    if execute and code.strip():
        exec_success, exec_output = execute_workflow(code, user_query)
        if not exec_success:
            print(f"\n⚠️  Workflow execution failed.")
        else:
            print(f"\n✅ Workflow executed successfully.")

    return code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLP → BraneScript pipeline")
    parser.add_argument(
        "--model", default="Qwen/Qwen3.6-27B",
        help="Hugging Face model id, e.g. Qwen/Qwen3.6-27B, Qwen/Qwen3-4B or Qwen/Qwen3-4B-Instruct-2507."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
        help="Sampling temperature (default: 0.4). "
             "Lower = more deterministic code output."
    )
    parser.add_argument(
        "--execute", action="store_true", default=False,
        help="Submit the generated BraneScript to the file-based job queue on Snellius "
             "(~/brane_jobs/pending/) and wait for job_watcher.py on your local machine "
             "to execute it. See scripts/remote_execution/README.md."
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

    print(f"🔧 Initialising — model: {args.model}  temperature: {args.temperature}  execute: {args.execute}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("📥 Loading model onto GPU...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("✅ Model loaded")

    # Override the model's baked-in generation_config (has max_length=20) to
    # avoid the "max_length=20 conflicts with max_new_tokens" warning.
    model.generation_config.max_length = 8192
    model.generation_config.max_new_tokens = None  # let pipeline_kwargs control this

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    # Pass max_new_tokens via pipeline_kwargs so LangChain's HuggingFacePipeline
    # respects it — without this kwarg, LangChain defaults to 256 tokens.
    llm = HuggingFacePipeline(
        pipeline=text_generation_pipeline,
        pipeline_kwargs={
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": 0.95,
        },
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

    user_query = (
        "Analyze the heart disease risk for a patient with the following profile: "
        "65-year-old female, blood pressure 170, heart rate 80, total cholesterol 230, "
        "medical history of hypertension. "
        "Use the healthcare::analyze_heart_disease function. "
        "Run the analysis on node giovanni."
    )
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
        execute=args.execute,
    )

    print("\n" + "=" * 50)
    print("FINAL BRANESCRIPT OUTPUT:")
    print("=" * 50)
    print(result)

    print("\n👌Pipeline execution completed.")