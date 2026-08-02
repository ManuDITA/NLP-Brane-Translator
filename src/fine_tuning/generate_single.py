#!/usr/bin/env python3
"""
generate_single.py

Generate a single BraneScript from a natural-language intent and write a
job file to ~/brane_jobs/pending/<req_id>.json so job_watcher.py picks it
up, executes it on the local Brane instance, and writes the result back.

Called automatically by sbatch_generate.sh (via the frontend Generate tab).
Can also be run manually:

    python src/fine_tuning/generate_single.py \\
        --intent "analyze diabetes risk for all patients in heal_pa_2" \\
        --model  "outputs/models/output_merged_qwen3.6-27b" \\
        --req-id "550e8400-e29b-41d4-a716-446655440000"

Required env vars (same as job_watcher.py):
    SNELLIUS_JOBS_DIR   Path on Snellius for the job queue (default: ~/brane_jobs)
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from prompts import load_system_prompt, build_user_message

JOBS_DIR = Path(os.environ.get("SNELLIUS_JOBS_DIR", str(Path.home() / "brane_jobs")))
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0


def _extract_bs(text: str) -> str:
    """Strip thinking tokens and markdown fences; return raw BraneScript."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _get_pkg_context(intent: str) -> str:
    """Load RAG context for the intent if brane_pkg_db is available."""
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from pkg_retriever import PkgRetriever

        db_path = _ROOT / "brane_pkg_db"
        if not db_path.exists():
            print("  ⚠️  brane_pkg_db not found — no package context injected.")
            print("       Run: python src/knowledgeBase.py")
            return ""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        db = Chroma(persist_directory=str(db_path), embedding_function=embeddings)
        retriever = PkgRetriever(pkg_db=db, k=4)
        ctx = retriever.run([], intent)
        print("  ✅ RAG context retrieved")
        return ctx
    except Exception as e:
        print(f"  ⚠️  RAG not available: {e}")
        return ""


def generate(intent: str, model_path: str,
             prev_script: str = "", error_feedback: str = "", attempt: int = 1) -> tuple[str, bool]:
    """
    Load model, generate BraneScript for intent, return (script, cache_hit).
    On a cache hit the model is never loaded.
    """
    # ── Semantic cache lookup (skip LLM if we've seen this before) ────────────
    _cache = None
    try:
        from semantic_cache import SemanticCache
        _cache = SemanticCache()
        hit = _cache.lookup(intent)
        if hit and attempt == 1:          # don't cache-hit on retries
            print(f"\n🎯 Cache hit (similarity={hit['similarity']:.4f}) — skipping LLM")
            print(f"   Matched: {hit['intent'][:80]}")
            return hit["branescript"], True
    except Exception as _ce:
        print(f"  ⚠️  Cache unavailable: {_ce}")
        _cache = None
    # ─────────────────────────────────────────────────────────────────────────

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n📥 Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("  ✅ Model loaded")

    pkg_context = _get_pkg_context(intent)

    # Build error_section for retry attempts
    error_section = ""
    if prev_script and error_feedback:
        error_section = (
            f"PREVIOUS ATTEMPT (attempt {attempt - 1}) FAILED.\n"
            f"The BraneScript below produced an error — fix it:\n\n"
            f"Previous script:\n```\n{prev_script}\n```\n\n"
            f"Error:\n{error_feedback}"
        )
        print(f"  🔄 Retry attempt {attempt} — injecting error feedback into prompt")

    system_prompt = load_system_prompt()
    user_message  = build_user_message(
        intent,
        pkg_context or "(no package context available)",
        error_section=error_section,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    # Disable thinking for Qwen3 models (faster, cleaner output)
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Collect EOS tokens so generation stops cleanly
    eos_ids = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)
    if not eos_ids:
        eos_ids = [tokenizer.eos_token_id]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    script = _extract_bs(raw)

    # ── Store in semantic cache (only on first attempt, not retries) ──────────
    if _cache and attempt == 1:
        try:
            _cache.store(intent=intent, branescript=script)
            print("  💾 Result stored in semantic cache")
        except Exception as _ce:
            print(f"  ⚠️  Cache store error: {_ce}")
    # ─────────────────────────────────────────────────────────────────────────

    return script, False


def main():
    parser = argparse.ArgumentParser(description="Generate a single BraneScript and enqueue it for execution.")
    parser.add_argument("--intent",       required=True, help="Natural-language intent to translate")
    parser.add_argument("--model",        required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--req-id",       required=True, dest="req_id",
                        help="Request UUID — used as job ID in the file queue")
    parser.add_argument("--context-file", default="",    dest="context_file",
                        help="Optional path to JSON file with {prev_script, error_feedback, attempt}")
    args = parser.parse_args()

    # Load retry context if provided
    prev_script = error_feedback = ""
    attempt = 1
    if args.context_file:
        ctx_path = Path(os.path.expanduser(args.context_file))
        if ctx_path.exists():
            try:
                ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                prev_script    = ctx.get("prev_script", "")
                error_feedback = ctx.get("error_feedback", "")
                attempt        = ctx.get("attempt", 2)
                print(f"  📋 Loaded retry context (attempt {attempt})")
            except Exception as e:
                print(f"  ⚠️  Could not load context file: {e}")

    print(f"\n{'='*60}")
    print(f"  Intent  : {args.intent[:80]}")
    print(f"  Model   : {args.model}")
    print(f"  Req ID  : {args.req_id}")
    if attempt > 1:
        print(f"  Attempt : {attempt}")
    print(f"{'='*60}")

    script, cache_hit = generate(args.intent, args.model,
                      prev_script=prev_script, error_feedback=error_feedback, attempt=attempt)

    src = "cache" if cache_hit else "LLM"
    print(f"\n✅ Generated BraneScript via {src} ({len(script)} chars):")
    print(script[:400] + ("..." if len(script) > 400 else ""))

    # Write job file so job_watcher.py picks it up and executes it locally
    pending_dir = JOBS_DIR / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    job = {
        "id":           args.req_id,
        "workflow":     script,
        "query":        args.intent,
        "model":        args.model,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    job_path = pending_dir / f"{args.req_id}.json"
    job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False))
    print(f"\n📤 Job written → {job_path}")
    print("   job_watcher.py will execute it locally and write the result back.")


if __name__ == "__main__":
    main()
