"""
evaluate.py

Evaluates any model (base, SFT, or GRPO) on a held-out test set and reports
three metrics:
  compile_rate      — % of generated scripts that compile without syntax errors
  execution_rate    — % of generated scripts that run successfully (exit_code==0)
  output_match_rate — % of successful scripts whose stdout matches the reference

Usage
─────
  # Evaluate base Qwen3.5-9B (zero-shot)
  python src/fine_tuning/evaluate.py --model Qwen/Qwen3.5-9B

  # Evaluate SFT-merged model
  python src/fine_tuning/evaluate.py --model src/fine_tuning/output_merged_qwen3.5-9b

  # Evaluate GRPO-merged model
  python src/fine_tuning/evaluate.py --model src/fine_tuning/output_merged_qwen3.5-9b_grpo

  # Evaluate a model with a separate LoRA adapter (before merging)
  python src/fine_tuning/evaluate.py \\
      --model Qwen/Qwen3.5-9B \\
      --adapter src/fine_tuning/output_qwen3.5-9b

  # Compare all three variants for a given model size in one run
  python src/fine_tuning/evaluate.py --model Qwen/Qwen3.5-9B --compare-all

Environment variables
─────────────────────
  BRANE_INSTANCE   Brane instance name (default: local-instance)

Output
──────
  Prints a results table to stdout and saves full details to
  outputs/eval/<model_slug>_<timestamp>.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Make src/ importable from fine_tuning/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import load_system_prompt, build_user_message  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE          = Path(__file__).resolve().parent
PROJECT_ROOT   = _HERE.parent.parent

# Default test file — override via --test-file or EVAL_TEST_FILE env var
_DEFAULT_TEST_FILE = PROJECT_ROOT / "outputs" / "eval" / "test_intents.jsonl"
TEST_FILE      = Path(os.environ.get("EVAL_TEST_FILE", str(_DEFAULT_TEST_FILE)))
TEST_RESULTS   = PROJECT_ROOT / "outputs" / "eval" / "test_results.jsonl"
EXEC_RESULTS   = PROJECT_ROOT / "data" / "training" / "execution_results.jsonl"
EVAL_DIR       = PROJECT_ROOT / "outputs" / "eval"

BRANE_INSTANCE = os.environ.get("BRANE_INSTANCE", "local-instance")
BRANE_TIMEOUT  = 30
BRANE_WORKERS  = 8
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.0   # greedy for reproducible evaluation

# Built once at module load — includes full syntax_reference.md
SYSTEM_PROMPT  = load_system_prompt()

# ---------------------------------------------------------------------------
# Package retriever — lazy-initialised once on first generate call
# ---------------------------------------------------------------------------
_PKG_RETRIEVER = None

def _get_pkg_retriever():
    """
    Return a PkgRetriever backed by brane_pkg_db.
    Initialised once; returns None if the DB is not present (Snellius without sync).
    """
    global _PKG_RETRIEVER
    if _PKG_RETRIEVER is not None:
        return _PKG_RETRIEVER
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from pkg_retriever import PkgRetriever
        db_path = PROJECT_ROOT / "brane_pkg_db"
        if not db_path.exists():
            print("  ⚠️  brane_pkg_db not found — no package context will be injected.")
            print("       Run: python src/knowledgeBase.py  (or sync brane_pkg_db/ from local)")
            return None
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        db = Chroma(persist_directory=str(db_path), embedding_function=embeddings)
        _PKG_RETRIEVER = PkgRetriever(pkg_db=db, k=4)
        print("  ✅ PkgRetriever ready")
    except Exception as e:
        print(f"  ⚠️  Could not load PkgRetriever: {e}")
        _PKG_RETRIEVER = None
    return _PKG_RETRIEVER

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_branescript(text: str) -> str:
    """Strip thinking blocks and markdown fences from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _run_branescript(code: str) -> dict:
    """Execute a BraneScript and return execution metadata."""
    code = _extract_branescript(code)
    if not code or len(code) < 5:
        return {"exit_code": -1, "stdout": "", "stderr": "empty output", "success": False,
                "error_type": "empty"}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bs", encoding="utf-8", delete=False
    ) as f:
        f.write(code)
        path = f.name

    try:
        result = subprocess.run(
            ["brane", "workflow", "run", BRANE_INSTANCE, path],
            capture_output=True, text=True, timeout=BRANE_TIMEOUT,
        )
        stderr = result.stderr.lower()
        if result.returncode != 0:
            if ("compilation of workflow failed" in stderr
                    or "parse error" in stderr
                    or "does not exist" in stderr
                    or "undefined function" in stderr):
                error_type = "compile"
            else:
                error_type = "runtime"
        else:
            error_type = None

        return {
            "exit_code":  result.returncode,
            "stdout":     result.stdout.strip(),
            "stderr":     result.stderr.strip(),
            "success":    result.returncode == 0,
            "error_type": error_type,
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "TIMEOUT", "success": False,
                "error_type": "timeout"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _load_test_set(test_file: Path | None = None) -> list[dict]:
    path = test_file or TEST_FILE
    if not path.exists():
        print(f"❌ Test file not found: {path}")
        sys.exit(1)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _get_reference_bs(ex: dict) -> str:
    """Extract the reference BraneScript from an example dict.

    Supports two formats:
      - Raw format  (data/examples/*.jsonl):  top-level "branescript" key
      - ChatML format (data/training/train.jsonl): messages[assistant].content
    """
    if ex.get("branescript"):
        return ex["branescript"]
    for msg in ex.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _load_reference_outputs() -> dict[str, str]:
    """
    Load intent → reference stdout.
    Checks execution_results.jsonl (full 607 suite) first, then test_results.jsonl.
    Both files are merged so coverage is maximised.
    Returns: (refs dict, time_dependent set of intents)
    """
    refs = {}
    time_dependent: set[str] = set()
    for src in [EXEC_RESULTS, TEST_RESULTS]:
        if not src.exists():
            continue
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                intent = r.get("intent", "")
                if r.get("success") and intent and intent not in refs:
                    refs[intent] = (r.get("stdout") or "").strip()
                if r.get("time_dependent") and intent:
                    time_dependent.add(intent)
            except Exception:
                pass
    if not refs:
        print("⚠️  No reference outputs found — output_match_rate will be skipped.")
    return refs, time_dependent


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(model_path: str, adapter_path: str | None = None):
    """Load model + tokenizer, optionally with a LoRA adapter."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"📥 Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"📥 Loading model in 4-bit: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        print(f"🔌 Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# Max wall-clock seconds to wait for a single generate() call before giving up.
GENERATION_TIMEOUT = 120


def generate_branescript(model, tokenizer, intent: str) -> str:
    """
    Generate BraneScript for a given intent using greedy decoding.
    Uses PkgRetriever to fetch only the relevant package context for this intent
    (same RAG pipeline as the interactive pipeline.py).
    Runs inside a thread so it can be abandoned if it hangs past
    GENERATION_TIMEOUT seconds (returns empty string on timeout).
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    retriever = _get_pkg_retriever()
    pkg_context = retriever.run([], intent) if retriever else "(No package context available.)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_message(
            question=intent, pkg_context=pkg_context)},
    ]
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

    def _generate():
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_generate)
        try:
            return future.result(timeout=GENERATION_TIMEOUT)
        except FuturesTimeout:
            raise TimeoutError(
                f"generate() timed out after {GENERATION_TIMEOUT}s"
            )


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _write_checkpoint(path: Path, model_label: str,
                      results: list, i: int, total: int) -> None:
    path.write_text(
        json.dumps({"model": model_label, "examples": results},
                   indent=2, ensure_ascii=False, default=str)
    )
    print(f"   💾 Checkpoint saved ({i}/{total})", flush=True)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model_path: str, adapter_path: str | None = None,
             label: str | None = None,
             test_file: Path | None = None,
             generate_only: bool = False,
             resume: bool = False) -> dict:
    """
    Run full evaluation on a test set.

    generate_only=True  → skip Brane execution (use on Snellius or any host
                          without a running Brane instance).  Results contain
                          generated_bs for every example; execution metrics are
                          omitted.  A companion *_generated.json is saved so
                          the scripts can be batch-executed locally afterwards.
    resume=True         → load an existing in-progress checkpoint and skip
                          already-processed examples (safe to re-submit after
                          a timeout or OOM kill).
    """
    test_set    = _load_test_set(test_file)
    if not generate_only:
        ref_outs, time_dep_intents = _load_reference_outputs()
    else:
        ref_outs, time_dep_intents = {}, set()
    model_label = label or Path(model_path).name
    suffix      = "_generated" if generate_only else ""

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    slug        = re.sub(r"[^a-z0-9_-]", "_", model_label.lower())
    ckpt_path   = EVAL_DIR / f"{slug}_checkpoint{suffix}.json"
    CHECKPOINT_EVERY = 50

    # ── Resume: load already-done results from checkpoint ────────────────────
    done_results: list[dict] = []
    done_ids: set[str] = set()
    if resume and ckpt_path.exists():
        try:
            prev = json.loads(ckpt_path.read_text())
            done_results = prev.get("examples", [])
            done_ids = {r["id"] for r in done_results if r.get("id")}
            print(f"♻️  Resuming from checkpoint — {len(done_results)} examples already done.")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint ({e}) — starting fresh.")
            done_results, done_ids = [], set()

    pending = [ex for ex in test_set
               if not (resume and ex.get("id") and ex["id"] in done_ids)]

    mode_str = "generate-only (no Brane execution)" if generate_only else "full (generate + execute)"
    print(f"\n{'='*60}")
    print(f"🔍 Evaluating : {model_label}")
    print(f"   Mode        : {mode_str}")
    print(f"   Test file   : {test_file or TEST_FILE}")
    print(f"   Total       : {len(test_set)}  |  Pending: {len(pending)}  |  Done: {len(done_results)}")
    print(f"   Checkpoint  : {ckpt_path.name}  (every {CHECKPOINT_EVERY} examples)")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_path, adapter_path)

    results = list(done_results)
    total   = len(test_set)

    for i, ex in enumerate(pending, start=len(done_results) + 1):
        intent = ex["intent"]
        print(f"[{i:3d}/{total}] {intent[:62]}…", end=" ", flush=True)

        # ── Generate ─────────────────────────────────────────────────────────
        try:
            generated = generate_branescript(model, tokenizer, intent)
        except Exception as exc:
            print(f"💥 generation error: {exc}", flush=True)
            results.append({
                "id":           ex.get("id", ""),
                "source_file":  ex.get("source_file", ""),
                "intent":       intent,
                "reference_bs": _get_reference_bs(ex),
                "generated_bs": "",
                "error":        f"generation_error: {exc}",
                **({"execution": {"exit_code": -1, "stdout": "", "stderr": str(exc),
                                  "success": False, "error_type": "generation"},
                    "ref_stdout": "", "output_match": None}
                   if not generate_only else {}),
            })
            if i % CHECKPOINT_EVERY == 0:
                _write_checkpoint(ckpt_path, model_label, results, i, total)
            continue

        # ── Generate-only path ────────────────────────────────────────────────
        if generate_only:
            print("✏️ generated", flush=True)
            results.append({
                "id":           ex.get("id", ""),
                "source_file":  ex.get("source_file", ""),
                "intent":       intent,
                "reference_bs": _get_reference_bs(ex),
                "generated_bs": generated,
            })
        else:
            # ── Execute path ──────────────────────────────────────────────────
            try:
                execution = _run_branescript(generated)
            except Exception as exc:
                print(f"💥 execution error: {exc}", flush=True)
                execution = {"exit_code": -1, "stdout": "", "stderr": str(exc),
                             "success": False, "error_type": "execution_error"}

            ref_stdout = ref_outs.get(intent, "")
            is_time_dep = intent in time_dep_intents
            output_match = (execution["stdout"] == ref_stdout
                            if execution["success"] and ref_stdout and not is_time_dep
                            else None)

            status    = "✅" if execution["success"] else ("🔶" if execution["error_type"] == "runtime" else "❌")
            match_str = (" match" if output_match else (" no-match" if output_match is False else ""))
            print(f"{status}{match_str}", flush=True)

            results.append({
                "id":           ex.get("id", ""),
                "source_file":  ex.get("source_file", ""),
                "intent":       intent,
                "reference_bs": _get_reference_bs(ex),
                "generated_bs": generated,
                "execution":    execution,
                "ref_stdout":   ref_stdout,
                "output_match": output_match,
            })

        # Periodic checkpoint
        if i % CHECKPOINT_EVERY == 0:
            _write_checkpoint(ckpt_path, model_label, results, i, total)

    # ── Build final metrics dict ──────────────────────────────────────────────
    metrics: dict = {
        "model":        model_label,
        "model_path":   model_path,
        "adapter_path": adapter_path,
        "total":        len(results),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "examples":     results,
    }

    if generate_only:
        metrics["mode"] = "generate_only"
        _save_results(metrics, suffix=suffix)
        print(f"\n✏️  Generated {len(results)} scripts — run execute_generated.py locally to get execution metrics.\n")
    else:
        n = len(results)
        compiled  = sum(1 for r in results if r["execution"]["error_type"] not in ("compile", "empty"))
        executed  = sum(1 for r in results if r["execution"]["success"])
        matchable = [r for r in results if r["output_match"] is not None]
        matched   = sum(1 for r in matchable if r["output_match"])

        metrics.update({
            "mode":              "full",
            "compile_rate":      round(compiled / n * 100, 1),
            "execution_rate":    round(executed / n * 100, 1),
            "output_match_rate": round(matched / len(matchable) * 100, 1) if matchable else None,
            "output_match_n":    len(matchable),
        })
        _print_metrics(metrics)
        _save_results(metrics)

    # Clean up checkpoint on successful completion
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"🗑️  Checkpoint removed (job complete).")

    return metrics


def _print_metrics(m: dict) -> None:
    print(f"\n{'─'*40}")
    print(f"  Model            : {m['model']}")
    print(f"  Total examples   : {m['total']}")
    print(f"  Compile rate     : {m['compile_rate']}%")
    print(f"  Execution rate   : {m['execution_rate']}%")
    if m.get("output_match_rate") is not None:
        print(f"  Output match rate: {m['output_match_rate']}%  (over {m['output_match_n']} comparable examples)")
    print(f"{'─'*40}\n")


def _save_results(m: dict, suffix: str = "") -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9_-]", "_", m["model"].lower())
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EVAL_DIR / f"{slug}_{ts}{suffix}.json"
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str))
    print(f"💾 Results saved → {path}")


# ---------------------------------------------------------------------------
# Compare-all mode: base / SFT / GRPO for a given model
# ---------------------------------------------------------------------------

def compare_all(base_model: str, test_file: Path | None = None,
               generate_only: bool = False, resume: bool = False) -> None:
    """Evaluate base, SFT-merged, and GRPO-merged variants and print comparison."""
    slug        = base_model.split("/")[-1].lower()
    models_dir  = PROJECT_ROOT / "outputs" / "models"
    sft_merged  = str(models_dir / f"output_merged_{slug}")
    grpo_merged = str(models_dir / f"output_merged_{slug}_grpo")

    variants = [(base_model, None, f"{slug} (base)")]
    if Path(sft_merged).exists():
        variants.append((sft_merged, None, f"{slug} (SFT)"))
    else:
        print(f"⚠️  SFT merged model not found at {sft_merged} — skipping.")

    if Path(grpo_merged).exists():
        variants.append((grpo_merged, None, f"{slug} (GRPO)"))
    else:
        print(f"⚠️  GRPO merged model not found at {grpo_merged} — skipping.")

    all_metrics = [
        evaluate(mp, ap, label, test_file=test_file,
                 generate_only=generate_only, resume=resume)
        for mp, ap, label in variants
    ]

    if generate_only:
        return

    print("\n" + "="*65)
    print(f"{'Model':<30} {'Compile':>8} {'Execute':>8} {'Match':>8}")
    print("─"*65)
    for m in all_metrics:
        match_str = f"{m['output_match_rate']}%" if m.get("output_match_rate") is not None else "—"
        print(f"{m['model']:<30} {m['compile_rate']:>7}% {m['execution_rate']:>7}% {match_str:>8}")
    print("="*65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on the BraneScript test set")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or path to merged model directory")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter directory (optional, for pre-merge evaluation)")
    parser.add_argument("--label", default=None,
                        help="Human-readable label for this model in results (e.g. 'qwen3-4b baseline')")
    parser.add_argument("--test-file", default=None,
                        help="Path to a .jsonl test file (default: outputs/eval/test_intents.jsonl). "
                             "Pass data/training/train.jsonl to benchmark on the full training suite.")
    parser.add_argument("--generate-only", action="store_true",
                        help="Skip Brane execution — only generate BraneScript and save to JSON. "
                             "Use on Snellius or any host without a running Brane instance.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing checkpoint — skips already-processed examples. "
                             "Safe to use after a job timeout or OOM kill.")
    parser.add_argument("--compare-all", action="store_true",
                        help="Evaluate base + SFT + GRPO variants for the given model in sequence")
    args = parser.parse_args()

    tf = Path(args.test_file) if args.test_file else None

    if args.compare_all:
        compare_all(args.model, test_file=tf,
                    generate_only=args.generate_only, resume=args.resume)
    else:
        evaluate(args.model, adapter_path=args.adapter, label=args.label,
                 test_file=tf, generate_only=args.generate_only, resume=args.resume)
