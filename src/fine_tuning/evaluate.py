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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE          = Path(__file__).resolve().parent
PROJECT_ROOT   = _HERE.parent.parent
TEST_FILE      = PROJECT_ROOT / "outputs" / "eval" / "test_intents.jsonl"
TEST_RESULTS   = PROJECT_ROOT / "outputs" / "eval" / "test_results.jsonl"
EVAL_DIR       = PROJECT_ROOT / "outputs" / "eval"

BRANE_INSTANCE = os.environ.get("BRANE_INSTANCE", "local-instance")
BRANE_TIMEOUT  = 30
BRANE_WORKERS  = 8
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.0   # greedy for reproducible evaluation

SYSTEM_PROMPT = (
    "You are an expert in the Brane Framework and BraneScript. "
    "Given a user intent, generate ONLY valid BraneScript code. "
    "Do NOT output Python, Java, or any other language. "
    "Do NOT wrap the output in markdown code fences. "
    "Use `let <name> := <expr>;` for variable assignment. "
    "After importing a package, call functions directly as `function_name(args)` "
    "(never `<package>::<function>(args)`). Output raw BraneScript code only."
)

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


def _load_test_set() -> list[dict]:
    if not TEST_FILE.exists():
        print(f"❌ Test file not found: {TEST_FILE}")
        print("   Run: python scripts/batch_execute.py --filter test to generate it.")
        sys.exit(1)
    return [json.loads(l) for l in TEST_FILE.read_text().splitlines() if l.strip()]


def _load_reference_outputs() -> dict[str, str]:
    """Load intent → reference stdout from test_results.jsonl."""
    if not TEST_RESULTS.exists():
        print(f"⚠️  {TEST_RESULTS} not found — output_match_rate will be skipped.")
        return {}
    refs = {}
    for line in TEST_RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if r.get("success") and r.get("intent"):
                refs[r["intent"]] = (r.get("stdout") or "").strip()
        except Exception:
            pass
    return refs


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


def generate_branescript(model, tokenizer, intent: str) -> str:
    """Generate BraneScript for a given intent using greedy decoding."""
    import torch

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": intent},
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


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model_path: str, adapter_path: str | None = None,
             label: str | None = None) -> dict:
    """
    Run full evaluation on the test set.
    Returns a dict with metrics and per-example details.
    """
    test_set  = _load_test_set()
    ref_outs  = _load_reference_outputs()
    model_label = label or Path(model_path).name

    print(f"\n{'='*60}")
    print(f"🔍 Evaluating: {model_label}")
    print(f"   Test examples : {len(test_set)}")
    print(f"   Reference outs: {len(ref_outs)}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_path, adapter_path)

    results = []
    for i, ex in enumerate(test_set, 1):
        intent = ex["intent"]
        print(f"[{i:3d}/{len(test_set)}] {intent[:65]}…", end=" ", flush=True)

        generated = generate_branescript(model, tokenizer, intent)
        execution = _run_branescript(generated)

        ref_stdout = ref_outs.get(intent, "")
        if execution["success"] and ref_stdout:
            output_match = execution["stdout"] == ref_stdout
        else:
            output_match = None  # can't determine

        status = "✅" if execution["success"] else ("🔶" if execution["error_type"] == "runtime" else "❌")
        match_str = (" match" if output_match else (" no-match" if output_match is False else ""))
        print(f"{status}{match_str}", flush=True)

        results.append({
            "intent":        intent,
            "reference_bs":  ex.get("branescript", ""),
            "generated_bs":  generated,
            "execution":     execution,
            "ref_stdout":    ref_stdout,
            "output_match":  output_match,
        })

    # Compute metrics
    n = len(results)
    compiled   = sum(1 for r in results if r["execution"]["error_type"] != "compile"
                     and r["execution"]["error_type"] != "empty")
    executed   = sum(1 for r in results if r["execution"]["success"])
    matchable  = [r for r in results if r["output_match"] is not None]
    matched    = sum(1 for r in matchable if r["output_match"])

    metrics = {
        "model":            model_label,
        "model_path":       model_path,
        "adapter_path":     adapter_path,
        "total":            n,
        "compile_rate":     round(compiled / n * 100, 1),
        "execution_rate":   round(executed / n * 100, 1),
        "output_match_rate": round(matched / len(matchable) * 100, 1) if matchable else None,
        "output_match_n":   len(matchable),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "examples":         results,
    }

    _print_metrics(metrics)
    _save_results(metrics)
    return metrics


def _print_metrics(m: dict) -> None:
    print(f"\n{'─'*40}")
    print(f"  Model            : {m['model']}")
    print(f"  Total examples   : {m['total']}")
    print(f"  Compile rate     : {m['compile_rate']}%")
    print(f"  Execution rate   : {m['execution_rate']}%")
    if m["output_match_rate"] is not None:
        print(f"  Output match rate: {m['output_match_rate']}%  (over {m['output_match_n']} comparable examples)")
    print(f"{'─'*40}\n")


def _save_results(m: dict) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9_-]", "_", m["model"].lower())
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EVAL_DIR / f"{slug}_{ts}.json"
    # Save without per-example details for the summary, full details separately
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str))
    print(f"💾 Results saved → {path}")


# ---------------------------------------------------------------------------
# Compare-all mode: base / SFT / GRPO for a given model
# ---------------------------------------------------------------------------

def compare_all(base_model: str) -> None:
    """Evaluate base, SFT-merged, and GRPO-merged variants and print comparison."""
    slug       = base_model.split("/")[-1].lower()
    sft_merged = str(_HERE / f"output_merged_{slug}")
    grpo_merged= str(_HERE / f"output_merged_{slug}_grpo")

    variants = [
        (base_model,  None, f"{slug} (base)"),
    ]
    if Path(sft_merged).exists():
        variants.append((sft_merged, None, f"{slug} (SFT)"))
    else:
        print(f"⚠️  SFT merged model not found at {sft_merged} — skipping.")

    if Path(grpo_merged).exists():
        variants.append((grpo_merged, None, f"{slug} (GRPO)"))
    else:
        print(f"⚠️  GRPO merged model not found at {grpo_merged} — skipping.")

    all_metrics = [evaluate(mp, ap, label) for mp, ap, label in variants]

    # Summary table
    print("\n" + "="*65)
    print(f"{'Model':<30} {'Compile':>8} {'Execute':>8} {'Match':>8}")
    print("─"*65)
    for m in all_metrics:
        match_str = f"{m['output_match_rate']}%" if m["output_match_rate"] is not None else "—"
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
                        help="Human-readable label for this model in results (e.g. 'qwen3.5-9b SFT')")
    parser.add_argument("--compare-all", action="store_true",
                        help="Evaluate base + SFT + GRPO variants for the given model in sequence")
    args = parser.parse_args()

    if args.compare_all:
        compare_all(args.model)
    else:
        evaluate(args.model, adapter_path=args.adapter, label=args.label)
