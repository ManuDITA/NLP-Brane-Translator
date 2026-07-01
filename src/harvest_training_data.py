"""
harvest_training_data.py

Merges two sources of (intent, BraneScript) pairs and writes the result to the
fine-tuning JSONL files consumed by train.py:

  Source 1 — curated hand-written examples in data/examples/*.jsonl
  Source 2 — pipeline runs that passed all validation checks (verdict=pass)
             stored under training_data/runs/ and indexed in training_data/index.jsonl

Deduplication: intents are normalised (lower-cased, whitespace-collapsed) before
deduplication — identical intents from different sources are merged, keeping the
curated example when both exist.

Benchmark exclusion: intents that appear in benchmark/intents.jsonl are excluded
from the training split so they remain a clean held-out test set.

Output (ChatML format, same as prepare_dataset.py):
    src/fine_tuning/train.jsonl   — 85 % split
    src/fine_tuning/val.jsonl     — 15 % split

Usage
-----
    python src/harvest_training_data.py

    # Exclude benchmark items (recommended — keeps test set clean):
    python src/harvest_training_data.py --exclude-benchmark

    # Override output directory:
    python src/harvest_training_data.py --output-dir /path/to/output

    # Custom split fraction (default 0.15):
    python src/harvest_training_data.py --val-fraction 0.20
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR    = Path(__file__).resolve().parent
ROOT       = SRC_DIR.parent
EXAMPLES_DIR      = ROOT / "data" / "examples"
TRAINING_INDEX    = ROOT / "training_data" / "index.jsonl"
BENCHMARK_FILE    = ROOT / "benchmark" / "intents.jsonl"
DEFAULT_OUTPUT    = SRC_DIR / "fine_tuning"

VAL_FRACTION = 0.15
RANDOM_SEED  = 42

SYSTEM_PROMPT = (
    "You are an expert in the Brane Framework and BraneScript. "
    "Given a user intent, generate ONLY valid BraneScript code. "
    "Do NOT output Python, Java, or any other language. "
    "Do NOT wrap the output in markdown code fences. "
    "Use `let <name> := <expr>;` for variable assignment. "
    "After importing a package, call functions directly as `function_name(args)` (never `<package>::<function>(args)`). "
    "Output raw BraneScript code only."
)


# ---------------------------------------------------------------------------
# Normalise an intent string for deduplication
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Load curated examples from data/examples/*.jsonl
# ---------------------------------------------------------------------------
def load_curated_examples(examples_dir: Path) -> list[dict]:
    examples = []
    for jsonl_file in sorted(examples_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "intent" in entry and "branescript" in entry:
                        examples.append({
                            "intent":      entry["intent"],
                            "branescript": entry["branescript"],
                            "source":      f"curated:{jsonl_file.name}",
                        })
                except json.JSONDecodeError:
                    continue
        print(f"  📄 {jsonl_file.name}: loaded")
    return examples


# ---------------------------------------------------------------------------
# Load passed pipeline runs from training_data/
# ---------------------------------------------------------------------------
def load_training_runs(index_file: Path) -> list[dict]:
    if not index_file.exists():
        print(f"  ⚠️  training_data/index.jsonl not found at {index_file} — skipping runs")
        return []

    runs = []
    with open(index_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("verdict") != "pass":
                continue

            run_dir = Path(rec["run_dir"])
            intent_file = run_dir / "intent.txt"
            code_file   = run_dir / "generated.bs"

            if not intent_file.exists() or not code_file.exists():
                continue

            intent      = intent_file.read_text(encoding="utf-8").strip()
            branescript = code_file.read_text(encoding="utf-8").strip()

            if intent and branescript:
                runs.append({
                    "intent":      intent,
                    "branescript": branescript,
                    "source":      f"run:{rec.get('id', 'unknown')[:8]}",
                })

    return runs


# ---------------------------------------------------------------------------
# Load benchmark intents to exclude from training
# ---------------------------------------------------------------------------
def load_benchmark_intents(benchmark_file: Path) -> set[str]:
    if not benchmark_file.exists():
        return set()
    intents = set()
    with open(benchmark_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "intent" in item:
                    intents.add(_normalise(item["intent"]))
            except json.JSONDecodeError:
                continue
    return intents


# ---------------------------------------------------------------------------
# Convert a (intent, branescript) pair to ChatML format
# ---------------------------------------------------------------------------
def to_chat_format(entry: dict) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": entry["intent"]},
            {"role": "assistant", "content": entry["branescript"]},
        ]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Harvest training data from runs + curated examples")
    parser.add_argument(
        "--exclude-benchmark", action="store_true", default=False,
        help="Exclude intents that appear in benchmark/intents.jsonl (recommended)."
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT),
        help="Directory to write train.jsonl and val.jsonl (default: src/fine_tuning/)."
    )
    parser.add_argument(
        "--val-fraction", type=float, default=VAL_FRACTION,
        help=f"Fraction of examples used for validation (default: {VAL_FRACTION})."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"\n📂 Loading curated examples from {EXAMPLES_DIR}...")
    curated = load_curated_examples(EXAMPLES_DIR)
    print(f"   → {len(curated)} curated examples loaded")

    print(f"\n📂 Loading passed pipeline runs from {TRAINING_INDEX}...")
    runs = load_training_runs(TRAINING_INDEX)
    print(f"   → {len(runs)} passing runs loaded")

    benchmark_intents: set[str] = set()
    if args.exclude_benchmark:
        benchmark_intents = load_benchmark_intents(BENCHMARK_FILE)
        print(f"\n🔒 Excluding {len(benchmark_intents)} benchmark intents from training data")

    # ── Merge with deduplication ──────────────────────────────────────────
    # Curated examples take priority; runs fill in the rest.
    seen: dict[str, dict] = {}
    skipped_benchmark = 0

    for entry in curated:
        key = _normalise(entry["intent"])
        if key in benchmark_intents:
            skipped_benchmark += 1
            continue
        seen[key] = entry

    runs_added = 0
    for entry in runs:
        key = _normalise(entry["intent"])
        if key in benchmark_intents:
            skipped_benchmark += 1
            continue
        if key not in seen:
            seen[key] = entry
            runs_added += 1

    all_examples = list(seen.values())
    print(f"\n📊 After deduplication:")
    print(f"   Curated kept : {len(all_examples) - runs_added}")
    print(f"   Runs added   : {runs_added}")
    print(f"   Benchmark skip: {skipped_benchmark}")
    print(f"   Total        : {len(all_examples)}")

    if not all_examples:
        print("\n❌ No examples available. Check data/examples/ and training_data/.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(all_examples)

    split = max(1, int(len(all_examples) * args.val_fraction))
    val_set   = all_examples[:split]
    train_set = all_examples[split:]

    print(f"\n🔀 Split: {len(train_set)} train / {len(val_set)} val")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train.jsonl"
    val_file   = output_dir / "val.jsonl"

    for filepath, subset in [(train_file, train_set), (val_file, val_set)]:
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in subset:
                f.write(json.dumps(to_chat_format(entry), ensure_ascii=False) + "\n")
        print(f"💾 Saved {filepath.name} ({len(subset)} entries)")

    print(f"\n✅ Done. Fine-tuning data written to {output_dir}/")
    print(f"   Train → {train_file}")
    print(f"   Val   → {val_file}")


if __name__ == "__main__":
    main()
