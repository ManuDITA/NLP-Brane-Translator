"""
utils.py

Shared helpers used by pipeline.py, example_generator.py, and intent_decomposer.py.
Import from here — do NOT copy-paste these functions into other modules.
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
    """Remove Qwen3/DeepSeek-style <think>...</think> reasoning blocks."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


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
        let patient := "{\"age\": 45, \"blood_pressure\": 100}";

    This is the 'JSON-as-string' antipattern: the model serialises structured data
    into a JSON blob inside a BraneScript string literal, using backslash-escaped
    quotes (\").  Valid BraneScript never needs \" inside string literals — proper
    structured data should be represented with `class` + `new ClassName { ... }`.
    """
    return bool(re.search(r'\\"', text))


def looks_like_branescript(text: str) -> bool:
    """
    Return True only if the text looks like valid BraneScript.

    Applies two layers:
      1. Hard disqualifiers — Python-only syntax that can never appear in BraneScript.
      2. Positive markers — at least one BraneScript-specific construct must be present.
    """
    code = text.strip()
    if not code:
        return False

    # ── Layer 1: hard disqualifiers ──────────────────────────────────────────
    if detect_python_code(code):
        return False

    # ── Layer 2: at least one positive BraneScript marker ───────────────────
    if "let " in code and ":=" in code:
        return True
    if re.search(r'import\s+[A-Za-z]\w*\s*;', code):   # import pkg; (semicolon = BraneScript)
        return True
    if "::" in code:                                     # pkg::function call
        return True
    if re.search(r'\bfunc\s+\w+\s*\(', code):           # func name(
        return True
    if "#[on(" in code:                                  # node routing attribute
        return True
    if re.search(r'\bnew\s+[A-Z]\w*\s*\{', code):       # new ClassName {
        return True
    if code.startswith("workflow") or code.startswith("package"):
        return True

    return False
