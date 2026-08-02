#!/usr/bin/env python3
"""
generate_paraphrases.py

Generate semantically equivalent paraphrases of every intent in the training
dataset using an LLM.  The output is a new JSONL file with the same
BraneScript reference but different intent wordings.

This data serves two purposes:
  1. Training data augmentation — more diverse phrasing examples improve SFT
     generalisation.
  2. Semantic cache testing — pairs of (original, paraphrase) intent with the
     same BraneScript can be used to validate that the semantic cache correctly
     routes paraphrased intents to existing results.

Run on Snellius (requires GPU for reasonable speed):
  sbatch sbatch_paraphrase.sh

Or locally with a small model (slow):
  python scripts/generate_paraphrases.py --model Qwen/Qwen3.5-4B --n 2

Output:
  data/training/paraphrases.jsonl

Each output line has:
  id              <original_id>_para_<k>
  original_id     the source example id
  source_file     "paraphrases"
  intent          paraphrased intent text
  branescript     same as source example (unchanged)
  messages        full SFT message list (system + user[paraphrase] + assistant[branescript])
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TRAIN_FILE    = PROJECT_ROOT / "data" / "training" / "train.jsonl"
VAL_FILE      = PROJECT_ROOT / "data" / "training" / "val.jsonl"
OUT_FILE      = PROJECT_ROOT / "data" / "training" / "paraphrases.jsonl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL       = os.environ.get("PARAPHRASE_MODEL", "Qwen/Qwen3.5-4B")
DEFAULT_N           = int(os.environ.get("PARAPHRASE_N", "3"))
MAX_NEW_TOKENS      = 512
TEMPERATURE         = 0.7
HF_CACHE            = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

SYSTEM_PROMPT = """\
You are a paraphrasing assistant for scientific computing intents.
Given an intent that describes a data analysis task, you must produce
alternative phrasings that mean exactly the same thing.

Rules:
- Keep all dataset names, column names, parameter values, and IDs
  exactly as they appear in the original.
- Change only the wording, sentence structure, or level of formality.
- Each paraphrase must be a single sentence or short paragraph — no bullet
  points, no numbering, no quotes around the text.
- Output exactly N paraphrases, one per line, with no other text."""


def _build_prompt(intent: str, n: int) -> list[dict]:
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content":
            f"Original intent: {intent}\n\n"
            f"Generate {n} paraphrase(s), one per line:"},
    ]


def _parse_paraphrases(text: str, n: int) -> list[str]:
    """Extract up to n non-empty lines from model output."""
    lines = [l.strip() for l in text.strip().splitlines()]
    # Strip leading numbering like "1." or "1)"
    cleaned = []
    for line in lines:
        line = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        if line and len(line) > 10:
            cleaned.append(line)
    return cleaned[:n]


def load_model(model_path: str):
    print(f"📥 Loading model: {model_path}", flush=True)
    os.environ["HF_HOME"] = HF_CACHE

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("  ✅ Model loaded", flush=True)
    return model, tokenizer


def generate_paraphrases(model, tokenizer, intent: str, n: int) -> list[str]:
    messages = _build_prompt(intent, n)
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
            do_sample=True,
            temperature=TEMPERATURE,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _parse_paraphrases(raw, n)


def _get_branescript(ex: dict) -> str:
    """Extract the reference BraneScript from a training example."""
    for msg in reversed(ex.get("messages", [])):
        if msg.get("role") == "assistant":
            content = msg["content"]
            # Strip thinking blocks
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            # Strip markdown fences
            content = re.sub(r"```[a-zA-Z]*\n?", "", content).strip()
            return content
    return ex.get("branescript", "")


def _build_messages(paraphrase: str, branescript: str) -> list[dict]:
    from src.prompts import SYSTEM_PROMPT as BS_SYSTEM
    from src.prompts import build_user_message
    return [
        {"role": "system",    "content": BS_SYSTEM},
        {"role": "user",      "content": build_user_message(question=paraphrase)},
        {"role": "assistant", "content": branescript},
    ]


def run(model_path: str, n: int, sources: list[Path], resume: bool):
    # Load source examples
    examples = []
    for src in sources:
        if not src.exists():
            print(f"⚠️  Not found: {src}", flush=True)
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except Exception:
                    pass
    print(f"📚 Loaded {len(examples)} source examples", flush=True)

    # Load already-done ids if resuming
    done_ids: set[str] = set()
    existing: list[dict] = []
    if resume and OUT_FILE.exists():
        for line in OUT_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done_ids.add(r.get("original_id", ""))
                    existing.append(r)
                except Exception:
                    pass
        print(f"  ↩️  Resuming — {len(done_ids)} already done", flush=True)

    pending = [e for e in examples if e.get("id") not in done_ids]
    if not pending:
        print("✅ All examples already processed.", flush=True)
        return

    model, tokenizer = load_model(model_path)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fout = OUT_FILE.open("a" if resume else "w", encoding="utf-8")

    # Write existing entries back if not resuming (shouldn't happen but safe)
    total_written = len(existing)
    total = len(pending)

    print(f"\n🚀 Generating {n} paraphrase(s) for {total} intents…\n", flush=True)
    t0 = time.time()

    for i, ex in enumerate(pending, start=1):
        intent     = ex.get("intent", "")
        orig_id    = ex.get("id", f"ex_{i}")
        branescript = _get_branescript(ex)

        print(f"[{i:4d}/{total}] {intent[:70]}…", flush=True)

        try:
            paraphrases = generate_paraphrases(model, tokenizer, intent, n)
        except Exception as e:
            print(f"  ⚠️  error: {e}", flush=True)
            paraphrases = []

        for k, para in enumerate(paraphrases, start=1):
            try:
                messages = _build_messages(para, branescript)
            except Exception:
                messages = []
            record = {
                "id":           f"{orig_id}_para_{k}",
                "original_id":  orig_id,
                "source_file":  "paraphrases",
                "intent":       para,
                "branescript":  branescript,
                "messages":     messages,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_written += 1

        fout.flush()

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate    = i / elapsed
            eta     = (total - i) / rate if rate > 0 else 0
            print(f"  ⏱  {i}/{total} done  |  {rate:.1f} ex/s  |  ETA {eta/60:.1f} min", flush=True)

    fout.close()
    print(f"\n✅ Done — {total_written} paraphrases written to {OUT_FILE}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate intent paraphrases using an LLM")
    ap.add_argument("--model",   default=DEFAULT_MODEL,
                    help="HuggingFace model ID or local path (default: %(default)s)")
    ap.add_argument("--n",       type=int, default=DEFAULT_N,
                    help="Paraphrases per intent (default: %(default)s)")
    ap.add_argument("--train-only", action="store_true",
                    help="Only paraphrase train.jsonl (skip val.jsonl)")
    ap.add_argument("--resume",  action="store_true",
                    help="Skip intents that already have output in paraphrases.jsonl")
    args = ap.parse_args()

    sources = [TRAIN_FILE] if args.train_only else [TRAIN_FILE, VAL_FILE]
    run(model_path=args.model, n=args.n, sources=sources, resume=args.resume)


if __name__ == "__main__":
    main()
