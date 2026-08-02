#!/usr/bin/env python3
"""
test_cache_lookup.py

Benchmark the semantic cache: given paraphrased intents (B), check how
often the cache correctly finds the original intent (A) and returns the
same BraneScript.

Prerequisites:
  1. Paraphrases generated:
       sbatch sbatch_paraphrase.sh                   (on Snellius)
       # → data/training/paraphrases.jsonl

  2. Cache populated:
       python scripts/populate_cache.py              (on Snellius)
       # → brane_pkg_db/intent_cache

Usage:
  python scripts/test_cache_lookup.py
  python scripts/test_cache_lookup.py --threshold 0.90
  python scripts/test_cache_lookup.py --output results/cache_benchmark.json

Metrics reported:
  hit_rate          % of B intents that found any cache entry >= threshold
  correct_rate      % of B intents that found the RIGHT BraneScript (exact match)
  fuzzy_correct     % with normalised script similarity >= 0.95 (ignores whitespace)
  false_positive    % of hits that returned the WRONG BraneScript
  similarity_stats  mean/min/max cosine similarity of all hits
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PARA_FILE  = PROJECT_ROOT / "data" / "training" / "paraphrases.jsonl"
TRAIN_FILE = PROJECT_ROOT / "data" / "training" / "train.jsonl"
VAL_FILE   = PROJECT_ROOT / "data" / "training" / "val.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(script: str) -> str:
    """Normalise BraneScript for fuzzy comparison (strip whitespace, comments)."""
    s = re.sub(r"//[^\n]*", "", script)        # strip line comments
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fuzzy_match(a: str, b: str) -> float:
    """Quick character-level similarity in [0, 1]."""
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    # Jaccard on bigrams
    def bigrams(s):
        return {s[i:i+2] for i in range(len(s) - 1)}
    sa, sb = bigrams(na), bigrams(nb)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _load_training_scripts(sources: list[Path]) -> dict[str, str]:
    """Load {original_id: branescript} from training JSONL files."""
    mapping: dict[str, str] = {}
    for src in sources:
        if not src.exists():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                bs = ""
                for msg in reversed(ex.get("messages", [])):
                    if msg.get("role") == "assistant":
                        content = msg["content"]
                        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                        content = re.sub(r"```[a-zA-Z]*\n?", "", content).strip()
                        bs = content
                        break
                if not bs:
                    bs = ex.get("branescript", "")
                if ex.get("id") and bs:
                    mapping[ex["id"]] = bs
            except Exception:
                pass
    return mapping


def _load_paraphrases() -> list[dict]:
    """Load all paraphrase entries."""
    if not PARA_FILE.exists():
        print(f"❌ Paraphrases file not found: {PARA_FILE}")
        print("   Run: sbatch sbatch_paraphrase.sh")
        sys.exit(1)
    entries = []
    for line in PARA_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return entries


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(threshold: float, output: Path | None):
    from semantic_cache import SemanticCache
    cache = SemanticCache(threshold=threshold)

    stats = cache.stats()
    if stats["total_entries"] == 0:
        print("❌ Cache is empty. Run: python scripts/populate_cache.py")
        sys.exit(1)

    print(f"📊 Cache: {stats['total_entries']} entries  (threshold={threshold})")

    # Load ground-truth scripts
    ref_scripts = _load_training_scripts([TRAIN_FILE, VAL_FILE])
    print(f"📚 Ground truth: {len(ref_scripts)} examples")

    paraphrases = _load_paraphrases()
    print(f"🔤 Paraphrases : {len(paraphrases)} entries\n")

    # Benchmark
    total       = 0
    hits        = 0
    correct     = 0
    fuzzy_ok    = 0
    false_pos   = 0
    no_ref      = 0
    similarities = []
    misses_by_src: dict[str, int] = defaultdict(int)

    for para in paraphrases:
        intent      = (para.get("intent") or "").strip()
        original_id = para.get("original_id", "")
        if not intent or not original_id:
            continue

        expected_bs = ref_scripts.get(original_id)
        if not expected_bs:
            no_ref += 1
            continue

        total += 1
        hit = cache.lookup(intent)

        if hit is None:
            misses_by_src[original_id] += 1
            continue

        hits += 1
        similarities.append(hit["similarity"])
        returned_bs = hit["branescript"]

        exact = _norm(returned_bs) == _norm(expected_bs)
        fuzz  = _fuzzy_match(returned_bs, expected_bs)

        if exact:
            correct += 1
            fuzzy_ok += 1
        elif fuzz >= 0.95:
            fuzzy_ok += 1
        else:
            false_pos += 1

    # ── Results ──────────────────────────────────────────────────────────────
    hit_rate       = hits / total * 100       if total else 0
    correct_rate   = correct / total * 100    if total else 0
    fuzzy_rate     = fuzzy_ok / total * 100   if total else 0
    fp_rate        = false_pos / hits * 100   if hits else 0
    sim_mean       = sum(similarities) / len(similarities) if similarities else 0
    sim_min        = min(similarities) if similarities else 0
    sim_max        = max(similarities) if similarities else 0

    bar = "─" * 50
    print(bar)
    print(f"  Threshold           : {threshold}")
    print(f"  Total B intents     : {total}")
    print(f"  Cache hits          : {hits}  ({hit_rate:.1f}%)")
    print(f"  Correct (exact)     : {correct}  ({correct_rate:.1f}%)")
    print(f"  Correct (fuzzy≥0.95): {fuzzy_ok}  ({fuzzy_rate:.1f}%)")
    print(f"  False positives     : {false_pos}  ({fp_rate:.1f}% of hits)")
    print(f"  Cache misses        : {total - hits}")
    print(f"  No reference found  : {no_ref}")
    print(f"  Similarity (mean)   : {sim_mean:.4f}")
    print(f"  Similarity (min)    : {sim_min:.4f}")
    print(f"  Similarity (max)    : {sim_max:.4f}")
    print(bar)

    results = {
        "threshold":        threshold,
        "total_b_intents":  total,
        "hits":             hits,
        "hit_rate_pct":     round(hit_rate, 2),
        "correct_exact":    correct,
        "correct_rate_pct": round(correct_rate, 2),
        "fuzzy_correct":    fuzzy_ok,
        "fuzzy_rate_pct":   round(fuzzy_rate, 2),
        "false_positives":  false_pos,
        "fp_rate_pct":      round(fp_rate, 2),
        "misses":           total - hits,
        "similarity_mean":  round(sim_mean, 4),
        "similarity_min":   round(sim_min, 4),
        "similarity_max":   round(sim_max, 4),
        "top_missed_originals": sorted(
            misses_by_src.items(), key=lambda x: -x[1]
        )[:10],
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n💾 Results saved → {output}")

    return results


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep(thresholds: list[float], output: Path | None):
    """Run benchmark at multiple thresholds — queries ChromaDB once per paraphrase,
    then applies each threshold as a post-filter (much faster than re-querying)."""
    from semantic_cache import SemanticCache

    ref_scripts = _load_training_scripts([TRAIN_FILE, VAL_FILE])
    paraphrases = _load_paraphrases()

    # Use threshold=0.0 to always get a result, capturing raw similarity
    cache = SemanticCache(threshold=0.0)
    stats = cache.stats()
    if stats["total_entries"] == 0:
        print("❌ Cache is empty. Run: python scripts/populate_cache.py")
        sys.exit(1)
    print(f"📊 Cache: {stats['total_entries']} entries")
    print(f"📚 Ground truth: {len(ref_scripts)} examples")
    print(f"🔤 Paraphrases : {len(paraphrases)} entries")
    print(f"⚡ Querying once per paraphrase, then sweeping thresholds...\n")

    # Single pass: collect (similarity, is_correct) for every paraphrase
    records = []   # (similarity, is_correct: bool, had_ref: bool)
    for i, para in enumerate(paraphrases, 1):
        intent      = (para.get("intent") or "").strip()
        original_id = para.get("original_id", "")
        if not intent or not original_id:
            continue
        expected_bs = ref_scripts.get(original_id)
        if not expected_bs:
            records.append((None, False, False))
            continue
        hit = cache.lookup(intent)
        if hit is None:
            records.append((0.0, False, True))
            continue
        is_correct = _norm(hit["branescript"]) == _norm(expected_bs)
        records.append((hit["similarity"], is_correct, True))
        if i % 200 == 0:
            print(f"  [{i}/{len(paraphrases)}] queried...")

    total_with_ref = sum(1 for _, _, has_ref in records if has_ref)

    print(f"\n{'Threshold':>10} {'Hit%':>8} {'Correct%':>10} {'FP%':>8}")
    print("─" * 44)

    sweep_results = []
    for t in thresholds:
        hits = correct = false_pos = 0
        for sim, is_correct, has_ref in records:
            if not has_ref:
                continue
            if sim is not None and sim >= t:
                hits += 1
                if is_correct:
                    correct += 1
                else:
                    false_pos += 1
        hr  = hits / total_with_ref * 100 if total_with_ref else 0
        cr  = correct / total_with_ref * 100 if total_with_ref else 0
        fpr = false_pos / hits * 100 if hits else 0
        print(f"{t:>10.2f} {hr:>7.1f}% {cr:>9.1f}% {fpr:>7.1f}%")
        sweep_results.append({
            "threshold": t, "hit_rate_pct": round(hr, 2),
            "correct_rate_pct": round(cr, 2), "fp_rate_pct": round(fpr, 2),
        })
        # Save incrementally after each threshold so partial results are never lost
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"sweep": sweep_results}, indent=2))

    if output:
        print(f"\n💾 Results saved → {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark semantic cache lookup accuracy")
    ap.add_argument("--threshold", type=float, default=0.92,
                    help="Cosine similarity threshold (default: 0.92)")
    ap.add_argument("--sweep",     action="store_true",
                    help="Run across multiple thresholds (0.80 to 0.99)")
    ap.add_argument("--output",    type=Path, default=None,
                    help="Save JSON results to this file")
    args = ap.parse_args()

    if args.sweep:
        sweep([0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98], output=args.output)
    else:
        benchmark(threshold=args.threshold, output=args.output)


if __name__ == "__main__":
    main()
