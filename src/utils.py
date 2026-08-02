"""
utils.py
Shared helpers
"""

import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# HuggingFace token loader
# ---------------------------------------------------------------------------

def load_hf_token(env_file: str | None = None) -> None:
    """
    Load HF_TOKEN from a .env file and set it in the environment so that
    HuggingFace Hub (used by HuggingFaceEmbeddings) authenticates automatically.

    Looks for .env in the project root (two levels up from this file) unless
    env_file is given explicitly.  Safe to call multiple times — a no-op if
    HF_TOKEN is already set in the environment.
    """
    if os.environ.get("HF_TOKEN"):
        return  # already set (e.g. exported in the shell)

    if env_file is None:
        # src/utils.py → src/ → project root
        env_file = str(Path(__file__).resolve().parent.parent / ".env")

    if not os.path.exists(env_file):
        print(f"⚠️  No .env file found at {env_file}.")
        print("    Copy .env.example → .env and add your HuggingFace token to avoid rate limits.")
        return

    from dotenv import load_dotenv
    load_dotenv(env_file, override=False)

    token = os.environ.get("HF_TOKEN", "")
    if token and token != "your_token_here":
        # Also set the legacy name some HF libraries still check
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        print("🔑 HuggingFace token loaded from .env")
    else:
        print("⚠️  HF_TOKEN not set or still set to placeholder in .env.")
        print("    Edit .env and replace 'your_token_here' with your actual token.")


def strip_thinking_tokens(text: str) -> str:
    """
    Remove Qwen3/DeepSeek-style reasoning blocks.

    Handles two cases:
    1. Complete blocks:   <think>...</think>  → removed
    2. Unclosed blocks:  <think>...EOF       → everything from <think> to end removed
       (This covers the case where the model starts thinking mid-output and the
       generation budget runs out before </think>, leaving just the preamble code.)
    """
    # Remove complete <think>...</think> blocks first
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove any remaining unclosed <think> block (truncates to end of string)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def strip_code_fences(text: str) -> str:
    """
    Extract code from markdown fences if the model wrapped its output.
    Handles ```bscript, ```branescript, ```bs, or plain ```.
    Returns the text unchanged if no fence is found.
    """
    match = re.search(
        r'```(?:bscript|branescript|bs)\s*\n(.*?)```',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    match = re.search(r'```(?:\w*)\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


def detect_python_code(text: str) -> bool:
    """
    Return True if the text appears to be Python rather than BraneScript.

    Checks for Python-exclusive syntax that can never appear in valid BraneScript.
    """
    checks = [
        r'\bdef\s+\w+\s*\(',            # def function(
        r'\bfrom\s+\w+\s+import\b',     # from X import Y
        r'\bself\s*\.',                  # self.attribute
        r'^\s*class\s+\w+\s*:\s*$',     # class Foo:  (Python uses colon, BraneScript uses brace)
        r'\bimport\s+(?:os|sys|re|json|math|datetime|typing|enum|random)\b',  # stdlib imports
    ]
    for pattern in checks:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def detect_json_string_assignment(text: str) -> bool:
    """
    Return True if the code passes structured data as an escaped JSON string, e.g.:
        let p := "{\"iterations\": 100, \"threshold\": 50}";

    This is the 'JSON-as-string' antipattern: the model serialises structured data
    into a JSON blob inside a BraneScript string literal, using backslash-escaped
    quotes (\").  Valid BraneScript never needs \" inside string literals — proper
    structured data should be represented with `class` + `new ClassName { ... }`.
    """
    return bool(re.search(r'\\"', text))


def looks_like_branescript(text: str) -> bool:
    """
    Return True only if the text looks like useful, complete BraneScript.

    Applies three layers:
      1. Hard disqualifiers — Python-only syntax that can never appear in BraneScript.
      2. Positive markers — at least one BraneScript-specific construct must be present.
      3. Completeness check — a bare `import pkg;` with no actual logic is not
         considered valid generated code (it is the residue left when a thinking
         block consumes most of the generation budget).
    """
    code = text.strip()
    if not code:
        return False

    # ── Layer 1: hard disqualifiers ──────────────────────────────────────────
    if detect_python_code(code):
        return False

    # ── Layer 2: at least one positive BraneScript marker ───────────────────
    has_marker = False
    if "let " in code and ":=" in code:
        has_marker = True
    elif re.search(r'import\s+[A-Za-z]\w*\s*;', code):
        has_marker = True
    elif re.search(r'\bfunc\s+\w+\s*\(', code):
        has_marker = True
    elif "#[on(" in code:
        has_marker = True
    elif re.search(r'\bnew\s+[A-Z]\w*\s*\{', code):
        has_marker = True
    elif code.startswith("workflow") or code.startswith("package"):
        has_marker = True

    if not has_marker:
        return False

    # ── Layer 3: completeness — must have at least one statement beyond imports ──
    # Strip comment lines and import lines; what remains must be non-empty.
    meaningful = [
        line for line in code.splitlines()
        if line.strip()
        and not line.strip().startswith("//")
        and not re.match(r'\s*import\s+\w+\s*;', line)
    ]
    if not meaningful:
        return False  # only import/comment lines — generation was cut short

    return True
