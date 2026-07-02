"""
generate_ground_truth.py

Uses an external LLM (Claude or OpenAI) to generate high-quality reference
BraneScript implementations for benchmark intents.

These references become the evaluation gold standard:
  - evaluate.py --llm-judge uses them to judge model outputs
  - fine_tuning/train.py uses them as targets in compute_metrics
  - They can also be added to the training set as high-quality examples

Output: benchmark/llm_references.jsonl
Each line: {id, intent, reference_code, model, syntax_ok, timestamp}

Usage:
    # Generate references for all benchmark items
    source .env
    python src/generate_ground_truth.py

    # Generate for a custom intents file (one intent per line)
    python src/generate_ground_truth.py --intents data/intents.txt --output data/llm_refs.jsonl

    # Force-regenerate even if a reference already exists
    python src/generate_ground_truth.py --overwrite

Env vars:
    JUDGE_API          "anthropic" (default) or "openai"
    JUDGE_MODEL        model name (overrides default)
    ANTHROPIC_API_KEY  required when JUDGE_API=anthropic
    OPENAI_API_KEY     required when JUDGE_API=openai
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
BENCHMARK_FILE  = PROJECT_ROOT / "benchmark" / "intents.jsonl"
DEFAULT_OUTPUT  = PROJECT_ROOT / "benchmark" / "llm_references.jsonl"
SYNTAX_REF_PATH = PROJECT_ROOT / "data" / "syntax_reference.md"
PKG_DB_PATH     = PROJECT_ROOT / "brane_pkg_db"


# ---------------------------------------------------------------------------
# External LLM call
# ---------------------------------------------------------------------------

def call_external_llm(system_msg: str, user_msg: str,
                      api: str, model: str) -> str:
    """Call external LLM and return raw text response."""
    if api == "anthropic":
        try:
            import anthropic
        except ImportError:
            print("❌ pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )
        return msg.content[0].text

    elif api == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ pip install openai")
            sys.exit(1)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
        )
        return resp.choices[0].message.content
    else:
        raise ValueError(f"Unknown api: {api!r}")


# ---------------------------------------------------------------------------
# Build prompt using the same template as pipeline.py
# ---------------------------------------------------------------------------

def build_prompt(intent: str, pkg_context: str, syntax_reference: str) -> tuple[str, str]:
    """Return (system_msg, user_msg) using the same templates as pipeline.py."""
    # Import templates from pipeline.py
    sys.path.insert(0, str(_HERE))
    from pipeline import (
        GENERATION_SYSTEM_TEMPLATE,
        GENERATION_USER_TEMPLATE,
        BRANESCRIPT_FEW_SHOT,
    )

    system_msg = GENERATION_SYSTEM_TEMPLATE.format(
        few_shot=BRANESCRIPT_FEW_SHOT,
        lang_context=syntax_reference,
    )
    user_msg = GENERATION_USER_TEMPLATE.format(
        question=intent,
        subtasks="(generate directly — no subtask breakdown)",
        pkg_context=pkg_context,
        error_section="",
    )
    return system_msg, user_msg


# ---------------------------------------------------------------------------
# Package context retrieval
# ---------------------------------------------------------------------------

def get_pkg_context(intent: str) -> str:
    """Retrieve relevant package/dataset context for an intent."""
    if not PKG_DB_PATH.exists():
        return "(package DB not available — include relevant imports manually)"

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        sys.path.insert(0, str(_HERE))
        from pkg_retriever import PkgRetriever

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        pkg_db     = Chroma(persist_directory=str(PKG_DB_PATH),
                            embedding_function=embeddings)
        retriever  = PkgRetriever(pkg_db=pkg_db, k=4)
        return retriever.run([intent], intent)
    except Exception as exc:
        print(f"   ⚠️  Could not load RAG: {exc} — proceeding without package context")
        return "(package context unavailable)"


# ---------------------------------------------------------------------------
# Post-process: strip code fences and thinking tokens
# ---------------------------------------------------------------------------

def clean_output(raw: str) -> str:
    sys.path.insert(0, str(_HERE))
    from utils import strip_thinking_tokens, strip_code_fences
    code = strip_thinking_tokens(raw)
    code = strip_code_fences(code)
    return code.strip()


def check_syntax(code: str) -> tuple[bool, str]:
    try:
        sys.path.insert(0, str(_HERE))
        from pipeline import check_syntax as _check
        return _check(code)
    except Exception:
        return True, ""  # can't check → assume ok


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_references(
    intents:   list[dict],
    output_path: Path,
    api:       str,
    model:     str,
    overwrite: bool,
) -> None:
    syntax_reference = SYNTAX_REF_PATH.read_text(encoding="utf-8")

    # Load already-generated references to support incremental runs
    existing: dict[str, dict] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                existing[rec["id"]] = rec
        print(f"📂 Found {len(existing)} existing references in {output_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    skip_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in intents:
            item_id = item.get("id", item.get("intent", "")[:40])
            intent  = item.get("intent", item.get("query", ""))

            if not overwrite and item_id in existing:
                out_f.write(json.dumps(existing[item_id], ensure_ascii=False) + "\n")
                skip_count += 1
                continue

            print(f"\n[{item_id}] Generating reference...")
            print(f"  Intent: {intent[:80]}")

            pkg_context = get_pkg_context(intent)
            system_msg, user_msg = build_prompt(intent, pkg_context, syntax_reference)

            try:
                raw = call_external_llm(system_msg, user_msg, api, model)
            except Exception as exc:
                print(f"  ❌ API call failed: {exc}")
                continue

            code = clean_output(raw)
            syntax_ok, syntax_err = check_syntax(code)

            status = "✅" if syntax_ok else "⚠️ "
            print(f"  {status} {len(code)} chars — syntax_ok={syntax_ok}")
            if not syntax_ok:
                print(f"     {syntax_err}")

            record = {
                "id":             item_id,
                "intent":         intent,
                "reference_code": code,
                "model":          f"{api}/{model}",
                "syntax_ok":      syntax_ok,
                "syntax_error":   syntax_err if not syntax_ok else None,
                "timestamp":      datetime.now(timezone.utc).isoformat(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            new_count += 1

    print(f"\n✅ Done: {new_count} generated, {skip_count} skipped")
    print(f"   Output: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM ground-truth BraneScript for benchmark intents"
    )
    parser.add_argument(
        "--intents", default=str(BENCHMARK_FILE),
        help=f"JSONL file with intent items (default: {BENCHMARK_FILE.name})"
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT.name})"
    )
    parser.add_argument(
        "--api", default=os.environ.get("JUDGE_API", "anthropic"),
        choices=["anthropic", "openai"],
        help="External LLM API to use (default: anthropic)"
    )
    parser.add_argument(
        "--model", default="",
        help="Model name (default: claude-3-5-sonnet-20241022 / gpt-4o)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate even for intents that already have a reference"
    )
    args = parser.parse_args()

    model = args.model
    if not model:
        model = "claude-3-5-sonnet-20241022" if args.api == "anthropic" else "gpt-4o"

    intents_path = Path(args.intents)
    if not intents_path.exists():
        print(f"❌ Intents file not found: {intents_path}")
        sys.exit(1)

    # Support both JSONL (with intent field) and plain text (one intent per line)
    intents = []
    for i, line in enumerate(intents_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            intents.append(json.loads(line))
        except json.JSONDecodeError:
            intents.append({"id": f"intent-{i+1:03d}", "intent": line})

    print(f"🎯 {len(intents)} intents  |  api={args.api}  model={model}")
    generate_references(intents, Path(args.output), args.api, model, args.overwrite)


if __name__ == "__main__":
    main()
