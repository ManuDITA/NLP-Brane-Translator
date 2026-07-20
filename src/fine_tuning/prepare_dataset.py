"""
prepare_dataset.py

Converts the curated (intent, BraneScript) example pairs in data/examples/*.jsonl
into a chat-format JSONL ready for QLoRA fine-tuning.

Only examples whose BraneScript executed successfully (exit_code == 0) are included,
using data/training/execution_results.jsonl as the filter. Examples not present in
execution_results.jsonl (e.g. syntax-only examples without a Brane run) are kept.

Output:
    data/training/train.jsonl  — 85% split
    data/training/val.jsonl    — 15% split

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
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
EXAMPLES_DIR = ROOT / "data" / "examples"
EXEC_RESULTS  = ROOT / "data" / "training" / "execution_results.jsonl"
OUTPUT_DIR = ROOT / "data" / "training"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"

VAL_FRACTION = 0.15
RANDOM_SEED = 42

# Make src/ importable
sys.path.insert(0, str(ROOT / "src"))
from prompts import load_system_prompt, build_user_message  # noqa: E402

SYSTEM_PROMPT = load_system_prompt()

# ---------------------------------------------------------------------------
# Package retriever — same RAG pipeline used at inference time
# ---------------------------------------------------------------------------
_PKG_RETRIEVER = None

def _get_pkg_retriever():
    global _PKG_RETRIEVER
    if _PKG_RETRIEVER is not None:
        return _PKG_RETRIEVER
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from pkg_retriever import PkgRetriever
        db_path = ROOT / "brane_pkg_db"
        if not db_path.exists():
            print("  ⚠️  brane_pkg_db not found — run: python src/knowledgeBase.py")
            return None
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        db = Chroma(persist_directory=str(db_path), embedding_function=embeddings)
        _PKG_RETRIEVER = PkgRetriever(pkg_db=db, k=4)
        print("  ✅ PkgRetriever ready for dataset preparation")
    except Exception as e:
        print(f"  ⚠️  Could not load PkgRetriever: {e}")
        _PKG_RETRIEVER = None
    return _PKG_RETRIEVER

# ---------------------------------------------------------------------------
# Load execution results: intent -> success bool
# ---------------------------------------------------------------------------

def load_execution_results(path: Path) -> dict[str, bool]:
    """Return a mapping of intent -> True/False based on execution_results.jsonl."""
    results: dict[str, bool] = {}
    if not path.exists():
        print(f"⚠️  {path.name} not found — skipping execution filter (all examples kept).")
        return results
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                intent = r.get("intent", "").strip()
                if intent:
                    results[intent] = bool(r.get("success", False))
            except json.JSONDecodeError:
                continue
    passed = sum(1 for v in results.values() if v)
    print(f"📊 Execution results loaded: {passed}/{len(results)} passed")
    return results


# ---------------------------------------------------------------------------
# Load examples
# ---------------------------------------------------------------------------

def load_examples(examples_dir: Path, exec_results: dict[str, bool]) -> list[dict]:
    examples = []
    total = kept = filtered = 0
    for jsonl_file in sorted(examples_dir.glob("*.jsonl")):
        file_total = file_kept = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "intent" not in entry or "branescript" not in entry:
                        continue
                    file_total += 1
                    intent = entry["intent"].strip()
                    # Keep if: passed execution, OR not present in results (no Brane run recorded)
                    if exec_results.get(intent, True):
                        # Ensure source_file is always populated
                        if "source_file" not in entry:
                            entry["source_file"] = jsonl_file.name
                        examples.append(entry)
                        file_kept += 1
                    else:
                        pass  # failed execution — drop
                except json.JSONDecodeError:
                    continue
        total += file_total
        kept += file_kept
        filtered += file_total - file_kept
        print(f"  📄 {jsonl_file.name}: {file_kept}/{file_total} kept")
    print(f"\n✅ Total: {kept} kept, {filtered} dropped (failed execution)")
    return examples


# ---------------------------------------------------------------------------
# Convert to chat format
# ---------------------------------------------------------------------------

def to_chat_format(entry: dict) -> dict:
    retriever = _get_pkg_retriever()
    intent = entry["intent"]
    pkg_context = retriever.run([], intent) if retriever else "(No package context available.)"
    return {
        "id":          entry.get("id", ""),
        "source_file": entry.get("source_file", ""),
        "intent":      intent,
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": build_user_message(
                question=intent, pkg_context=pkg_context)},
            {"role": "assistant", "content": entry["branescript"]},
        ]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"📂 Loading examples from {EXAMPLES_DIR}...")
    exec_results = load_execution_results(EXEC_RESULTS)
    examples = load_examples(EXAMPLES_DIR, exec_results)

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
