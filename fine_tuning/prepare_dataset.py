"""
prepare_dataset.py

Converts the curated (intent, BraneScript) example pairs in data/examples/*.jsonl
into a chat-format JSONL ready for QLoRA fine-tuning.

Output:
    fine_tuning/train.jsonl  — 85% split
    fine_tuning/val.jsonl    — 15% split

Each line in the output files follows the Alpaca/ChatML format:
    {
      "messages": [
        {"role": "system",    "content": "<system prompt>"},
        {"role": "user",      "content": "<intent>"},
        {"role": "assistant", "content": "<branescript>"}
      ]
    }

Run:
    python fine_tuning/prepare_dataset.py
"""

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "data" / "examples"
OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"

VAL_FRACTION = 0.15
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "You are an expert in the Brane Framework and BraneScript. "
    "Given a user intent, generate ONLY valid BraneScript code. "
    "Do NOT output Python, Java, or any other language. "
    "Do NOT wrap the output in markdown code fences. "
    "Use `let <name> := <expr>;` for variable assignment. "
    "Call package functions as `<package>::<function>(args)`. "
    "Output raw BraneScript code only."
)

# ---------------------------------------------------------------------------
# Load examples
# ---------------------------------------------------------------------------

def load_examples(examples_dir: Path) -> list[dict]:
    examples = []
    for jsonl_file in sorted(examples_dir.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "intent" in entry and "branescript" in entry:
                        examples.append(entry)
                except json.JSONDecodeError:
                    continue
        print(f"  📄 {jsonl_file.name}: loaded")
    return examples


# ---------------------------------------------------------------------------
# Convert to chat format
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
    print(f"📂 Loading examples from {EXAMPLES_DIR}...")
    examples = load_examples(EXAMPLES_DIR)
    print(f"✅ Total examples: {len(examples)}")

    if len(examples) == 0:
        print("❌ No examples found. Populate data/examples/*.jsonl first.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    split = max(1, int(len(examples) * VAL_FRACTION))
    val_examples = examples[:split]
    train_examples = examples[split:]

    print(f"📊 Train: {len(train_examples)}  |  Val: {len(val_examples)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filepath, subset in [(TRAIN_FILE, train_examples), (VAL_FILE, val_examples)]:
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in subset:
                f.write(json.dumps(to_chat_format(entry), ensure_ascii=False) + "\n")
        print(f"💾 Saved {filepath.name} ({len(subset)} entries)")

    print("\n✅ Dataset preparation complete.")
    print(f"   Train → {TRAIN_FILE}")
    print(f"   Val   → {VAL_FILE}")


if __name__ == "__main__":
    main()
