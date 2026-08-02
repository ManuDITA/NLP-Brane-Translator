#!/usr/bin/env python3
"""
populate_cache.py

Seed the semantic cache from the training dataset so that future intent
submissions can be served from cache instead of triggering an LLM call.

For each training example the reference BraneScript (the ground-truth answer
written by a human) is stored against the intent.  The cache then routes
semantically similar future intents to that script without re-generating.

Usage (run on Snellius where the cache DB lives):
    python scripts/populate_cache.py                   # from train + val
    python scripts/populate_cache.py --train-only
    python scripts/populate_cache.py --clear           # wipe cache first

Or via SLURM:
    sbatch --wrap="source .venv/bin/activate && python scripts/populate_cache.py" \\
           --partition=gpu_h100 --gpus=0 --mem=8G --time=00:30:00

Output:
    Populates the ChromaDB intent_cache collection in brane_pkg_db/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_cache import SemanticCache

TRAIN_FILE = PROJECT_ROOT / "data" / "training" / "train.jsonl"
VAL_FILE   = PROJECT_ROOT / "data" / "training" / "val.jsonl"


def _extract_branescript(ex: dict) -> str:
    """Pull the reference BraneScript from a training example."""
    for msg in reversed(ex.get("messages", [])):
        if msg.get("role") == "assistant":
            content = msg["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(r"```[a-zA-Z]*\n?", "", content).strip()
            return content
    return ex.get("branescript", "")


def populate(sources: list[Path], clear: bool, threshold: float):
    cache = SemanticCache(threshold=threshold)

    if clear:
        print("🗑️  Clearing existing cache…")
        # Delete all entries by listing and removing
        entries = cache.list_entries(limit=100_000)
        if entries:
            ids = cache._col.get()["ids"]
            if ids:
                cache._col.delete(ids=ids)
        print(f"   Removed {len(entries)} entries")

    stats_before = cache.stats()
    print(f"📊 Cache before: {stats_before['total_entries']} entries")

    examples = []
    for src in sources:
        if not src.exists():
            print(f"⚠️  Not found: {src}")
            continue
        count = 0
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                examples.append(json.loads(line))
                count += 1
            except Exception:
                pass
        print(f"📚 Loaded {count} examples from {src.name}")

    print(f"\n🚀 Populating cache with {len(examples)} examples…")
    stored = skipped = 0
    # Use a near-exact threshold for the pre-store dedup check so we only skip
    # true duplicates, not merely similar intents (which should each get their
    # own cache entry). Inference-time lookups still use the normal threshold.
    dedup_cache = SemanticCache(threshold=0.99)

    for i, ex in enumerate(examples, start=1):
        intent      = (ex.get("intent") or "").strip()
        branescript = _extract_branescript(ex)
        orig_id     = ex.get("id", f"ex_{i}")

        if not intent or not branescript:
            skipped += 1
            continue

        # Skip only true exact duplicates
        hit = dedup_cache.lookup(intent)
        if hit:
            skipped += 1
            continue

        cache.store(intent=intent, branescript=branescript, job_id=orig_id)
        stored += 1

        if i % 100 == 0:
            print(f"  [{i:4d}/{len(examples)}] stored={stored}  skipped={skipped}")

    stats_after = cache.stats()
    print(f"\n✅ Done.")
    print(f"   Stored  : {stored}")
    print(f"   Skipped : {skipped} (already cached or empty)")
    print(f"   Total in cache : {stats_after['total_entries']}")


def main():
    ap = argparse.ArgumentParser(description="Seed the semantic cache from training data")
    ap.add_argument("--train-only", action="store_true",
                    help="Only use train.jsonl (skip val.jsonl)")
    ap.add_argument("--clear",      action="store_true",
                    help="Wipe the cache before populating")
    ap.add_argument("--threshold",  type=float, default=0.92,
                    help="Similarity threshold for dedup check (default: 0.92)")
    args = ap.parse_args()

    sources = [TRAIN_FILE] if args.train_only else [TRAIN_FILE, VAL_FILE]
    populate(sources=sources, clear=args.clear, threshold=args.threshold)


if __name__ == "__main__":
    main()
