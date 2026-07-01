"""
evaluate.py

Evaluation harness for the NLP-Brane-Translator pipeline.

Modes
-----
  generate  — load the model, run the pipeline on every benchmark intent, and
              save per-intent results to results.jsonl in the output directory.
              Requires a GPU.

  score     — read a pre-generated results.jsonl, compare each generated script
              against the ground-truth using a static functional-equivalence
              metric, and write report.json.  No GPU required.

  full      — generate + score in one pass.

Functional equivalence metric (score 0 – 1)
--------------------------------------------
  +0.20  syntax check passes (balanced braces/parens, correct := assignment)
  +0.30  all expected packages are imported
  +0.30  all expected functions are called
  +0.20  all expected datasets are referenced
  ─────
  functional_match = True when score ≥ 0.80

Execution-based comparison (optional, requires --execute)
---------------------------------------------------------
  When --execute is passed in generate / full mode the pipeline also submits both
  the generated script AND the ground-truth script to the job queue.  The results
  are compared by normalised stdout and committed-dataset keys.

Usage
-----
  # On Snellius (GPU node) — generate results for a model:
  python src/evaluate.py --mode generate \\
      --model Qwen/Qwen3-27B \\
      --benchmark benchmark/intents.jsonl \\
      --output evaluation_results/qwen3-27b/

  # Locally — score pre-generated results:
  python src/evaluate.py --mode score \\
      --results evaluation_results/qwen3-27b/results.jsonl

  # Full pipeline (generate + score, with execution):
  python src/evaluate.py --mode full --execute \\
      --model Qwen/Qwen3-27B \\
      --output evaluation_results/qwen3-27b/
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_BENCHMARK  = PROJECT_ROOT / "benchmark" / "intents.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"

# ---------------------------------------------------------------------------
# Static functional-equivalence scoring
# ---------------------------------------------------------------------------

def _extract_imports(code: str) -> set[str]:
    return {m.lower() for m in re.findall(r'\bimport\s+([A-Za-z][A-Za-z0-9_\-]*)', code)}


def _extract_function_calls(code: str) -> set[str]:
    """Heuristic: identifiers followed by '(' that aren't keywords."""
    _keywords = {"if", "for", "while", "func", "return", "new", "let", "class",
                 "println", "print", "range", "parallel", "commit_result"}
    calls = set(re.findall(r'\b([A-Za-z][A-Za-z0-9_]*)\s*\(', code))
    return {c for c in calls if c not in _keywords}


def _extract_datasets(code: str) -> set[str]:
    """Find dataset names referenced via new Data { name := "..." }."""
    return {m.lower() for m in re.findall(
        r'new\s+Data\s*\{\s*name\s*:=\s*"([^"]+)"', code, re.IGNORECASE
    )}


def _check_syntax(code: str) -> tuple[bool, str]:
    errors = []
    if code.count("{") != code.count("}"):
        errors.append("Unbalanced braces.")
    if code.count("(") != code.count(")"):
        errors.append("Unbalanced parentheses.")
    let_lines = [l for l in code.splitlines() if re.match(r'\s*let\s+\w+\s*=', l)]
    for ll in let_lines:
        if ":=" not in ll:
            errors.append(f"Missing ':=' in: {ll.strip()}")
    return (not errors), "\n".join(errors)


def compute_functional_score(
    generated: str,
    benchmark_item: dict,
) -> tuple[float, dict]:
    """
    Compute a 0–1 functional equivalence score between *generated* and the
    requirements encoded in *benchmark_item*.

    Returns (score, detail_dict).
    """
    expected_packages  = set(benchmark_item.get("expected_packages", []))
    expected_functions = set(benchmark_item.get("expected_functions", []))
    expected_datasets  = set(b.lower() for b in benchmark_item.get("expected_datasets", []))

    gen_imports   = _extract_imports(generated)
    gen_functions = _extract_function_calls(generated)
    gen_datasets  = _extract_datasets(generated)

    # ── Syntax (0.20) ──────────────────────────────────────────────────
    syntax_ok, _ = _check_syntax(generated)
    syntax_score = 0.20 if syntax_ok else 0.0

    # ── Packages (0.30) ────────────────────────────────────────────────
    if expected_packages:
        matched_pkgs = {p for p in expected_packages if p.lower() in gen_imports}
        pkg_score = 0.30 * (len(matched_pkgs) / len(expected_packages))
    else:
        matched_pkgs = set()
        pkg_score = 0.30  # no packages expected → full credit

    # ── Functions (0.30) ───────────────────────────────────────────────
    if expected_functions:
        matched_fns = {f for f in expected_functions if f.lower() in
                       {g.lower() for g in gen_functions}}
        fn_score = 0.30 * (len(matched_fns) / len(expected_functions))
    else:
        matched_fns = set()
        fn_score = 0.30  # no functions expected → full credit

    # ── Datasets (0.20) ────────────────────────────────────────────────
    if expected_datasets:
        matched_ds = {d for d in expected_datasets if d in gen_datasets}
        ds_score = 0.20 * (len(matched_ds) / len(expected_datasets))
    else:
        matched_ds = set()
        ds_score = 0.20  # no datasets expected → full credit

    total = syntax_score + pkg_score + fn_score + ds_score

    detail = {
        "syntax_ok":         syntax_ok,
        "pkg_score":         round(pkg_score, 3),
        "fn_score":          round(fn_score, 3),
        "ds_score":          round(ds_score, 3),
        "matched_packages":  sorted(matched_pkgs),
        "matched_functions": sorted(matched_fns),
        "matched_datasets":  sorted(matched_ds),
        "missing_packages":  sorted(expected_packages - {p.lower() for p in matched_pkgs}),
        "missing_functions": sorted(expected_functions - {f.lower() for f in matched_fns}),
        "missing_datasets":  sorted(expected_datasets - matched_ds),
    }
    return round(total, 3), detail


FUNCTIONAL_MATCH_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Execution-result comparison (used when --execute is passed)
# ---------------------------------------------------------------------------

def _normalise_stdout(stdout: str) -> str:
    """Strip whitespace and lowercase for loose comparison."""
    return re.sub(r"\s+", " ", stdout.strip().lower())


def compare_execution_results(gen_result: dict, gt_result: dict) -> tuple[bool, str]:
    """
    Compare execution results of generated vs ground-truth script.

    Returns (match: bool, reason: str).
    """
    if not gen_result.get("success") and not gt_result.get("success"):
        return False, "both scripts failed execution"
    if not gt_result.get("success"):
        return False, "ground-truth script itself failed — benchmark item may be broken"
    if not gen_result.get("success"):
        return False, f"generated script failed: {gen_result.get('error_type', 'unknown')}"

    # Both succeeded — compare stdout (normalised)
    gen_out = _normalise_stdout(gen_result.get("stdout", ""))
    gt_out  = _normalise_stdout(gt_result.get("stdout", ""))
    stdout_match = (gen_out == gt_out)

    # Compare committed dataset keys (order-insensitive)
    gen_committed = set((gen_result.get("committed_data") or {}).keys())
    gt_committed  = set((gt_result.get("committed_data") or {}).keys())
    commit_match  = (gen_committed == gt_committed)

    if stdout_match and commit_match:
        return True, "stdout and committed datasets match"
    reasons = []
    if not stdout_match:
        reasons.append(f"stdout differs: got {repr(gen_out[:80])!r}, expected {repr(gt_out[:80])!r}")
    if not commit_match:
        reasons.append(f"committed datasets differ: got {sorted(gen_committed)}, expected {sorted(gt_committed)}")
    return False, "; ".join(reasons)


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------

def load_benchmark(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Skipping malformed line: {e}")
    return items


# ---------------------------------------------------------------------------
# Score mode — reads results.jsonl, writes report.json
# ---------------------------------------------------------------------------

def compute_report(results: list[dict], model_name: str, results_file: str) -> dict:
    total = len(results)
    if total == 0:
        return {"error": "no results"}

    validation_passes = sum(1 for r in results if r.get("validation_passed"))
    functional_matches = sum(1 for r in results if r.get("functional_match"))
    first_attempt = sum(1 for r in results if r.get("attempts", 1) == 1 and r.get("validation_passed"))
    attempt_counts = [r.get("attempts", 1) for r in results]
    mean_attempts = round(sum(attempt_counts) / len(attempt_counts), 2)

    # Execution metrics (optional)
    exec_results = [r for r in results if "execution_match" in r]
    exec_match_count = sum(1 for r in exec_results if r.get("execution_match"))

    # By difficulty
    by_difficulty: dict[str, dict] = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        by_difficulty.setdefault(diff, {"count": 0, "functional_matches": 0, "validation_passes": 0})
        by_difficulty[diff]["count"] += 1
        if r.get("functional_match"):
            by_difficulty[diff]["functional_matches"] += 1
        if r.get("validation_passed"):
            by_difficulty[diff]["validation_passes"] += 1
    for diff, data in by_difficulty.items():
        n = data["count"]
        data["functional_match_rate"]  = round(data["functional_matches"] / n, 3)
        data["validation_pass_rate"]   = round(data["validation_passes"] / n, 3)

    # By tag
    by_tag: dict[str, dict] = {}
    for r in results:
        for tag in r.get("tags", []):
            by_tag.setdefault(tag, {"count": 0, "functional_matches": 0})
            by_tag[tag]["count"] += 1
            if r.get("functional_match"):
                by_tag[tag]["functional_matches"] += 1
    for tag, data in by_tag.items():
        data["functional_match_rate"] = round(data["functional_matches"] / data["count"], 3)

    report = {
        "model":                   model_name,
        "timestamp":               datetime.now(timezone.utc).isoformat(),
        "results_file":            results_file,
        "total":                   total,
        "validation_pass_rate":    round(validation_passes / total, 3),
        "functional_match_rate":   round(functional_matches / total, 3),
        "first_attempt_rate":      round(first_attempt / total, 3),
        "mean_attempts":           mean_attempts,
        "by_difficulty":           by_difficulty,
        "by_tag":                  by_tag,
    }

    if exec_results:
        report["execution_run_count"]  = len(exec_results)
        report["execution_match_rate"] = round(exec_match_count / len(exec_results), 3)

    return report


def score_mode(results_path: Path, output_dir: Path) -> None:
    print(f"\n📊 Scoring results from {results_path}...")
    results = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Re-run functional scoring in case benchmark changed
    benchmark_map: dict[str, dict] = {}
    if DEFAULT_BENCHMARK.exists():
        for item in load_benchmark(DEFAULT_BENCHMARK):
            benchmark_map[item["id"]] = item

    for r in results:
        item = benchmark_map.get(r.get("id", ""), {})
        if item and r.get("generated"):
            score, detail = compute_functional_score(r["generated"], item)
            r["functional_score"]  = score
            r["functional_match"]  = score >= FUNCTIONAL_MATCH_THRESHOLD
            r["functional_detail"] = detail

    model_name = results[0].get("model", "unknown") if results else "unknown"
    report = compute_report(results, model_name, str(results_path))

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Model            : {report['model']}")
    print(f"  Total items      : {report['total']}")
    print(f"  Validation pass  : {report['validation_pass_rate']:.1%}")
    print(f"  Functional match : {report['functional_match_rate']:.1%}")
    print(f"  First-attempt    : {report['first_attempt_rate']:.1%}")
    print(f"  Mean attempts    : {report['mean_attempts']}")
    if "execution_match_rate" in report:
        print(f"  Execution match  : {report['execution_match_rate']:.1%}  "
              f"(of {report['execution_run_count']} executed)")
    print(f"{'='*55}")
    print(f"\n  By difficulty:")
    for diff in ("easy", "medium", "hard", "unknown"):
        if diff in report["by_difficulty"]:
            d = report["by_difficulty"][diff]
            print(f"    {diff:8s}: {d['functional_match_rate']:.1%} functional match  "
                  f"({d['count']} items)")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📄 Report written to {report_path}")

    # Also update results.jsonl with recomputed scores
    updated_results_path = output_dir / "results.jsonl"
    with open(updated_results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"📄 Updated results written to {updated_results_path}")


# ---------------------------------------------------------------------------
# Generate mode — loads the model and runs the full pipeline
# ---------------------------------------------------------------------------

def generate_mode(
    benchmark: list[dict],
    output_dir: Path,
    model_id: str,
    temperature: float,
    execute: bool,
) -> Path:
    # Add src/ to path so we can import pipeline components
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from pipeline import (
        build_pipeline_components,
        run_pipeline,
        execute_workflow,
        BRANESCRIPT_FEW_SHOT,
    )
    from training_collector import TrainingCollector

    print(f"\n🔧 Loading pipeline components (model: {model_id})...")
    components = build_pipeline_components(model_id, temperature)
    text_gen_pipeline = components["text_gen_pipeline"]
    tokenizer         = components["tokenizer"]
    decomposer        = components["decomposer"]
    pkg_retriever     = components["pkg_retriever"]
    syntax_reference  = components["syntax_reference"]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    # Separate collector so evaluation runs are tracked independently
    eval_collector = TrainingCollector(log_dir=str(output_dir / "training_data"))

    total = len(benchmark)
    results: list[dict] = []

    for idx, item in enumerate(benchmark, 1):
        item_id  = item["id"]
        intent   = item["intent"]
        ground_truth = item.get("ground_truth", "")

        print(f"\n{'─'*55}")
        print(f"[{idx}/{total}] {item_id}  ({item['difficulty']})  —  {intent[:70]}…"
              if len(intent) > 70 else f"[{idx}/{total}] {item_id}  ({item['difficulty']})  —  {intent}")

        # Run the pipeline (generation + validation)
        generated = run_pipeline(
            user_query=intent,
            decomposer=decomposer,
            pkg_retriever=pkg_retriever,
            text_gen_pipeline=text_gen_pipeline,
            tokenizer=tokenizer,
            syntax_reference=syntax_reference,
            few_shot_override=BRANESCRIPT_FEW_SHOT,
            execute=False,   # we handle execution separately below
            collector=eval_collector,
            model_name=model_id,
        )

        # Retrieve how many attempts it took from the collector's last entry
        attempts = 1
        if eval_collector.log_file.exists():
            with open(eval_collector.log_file, encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if lines:
                try:
                    last = json.loads(lines[-1])
                    attempts = last.get("attempt_number", 1)
                except json.JSONDecodeError:
                    pass

        validation_passed = bool(generated and generated.strip())

        # Functional score (static)
        score, detail = compute_functional_score(generated, item)
        functional_match = score >= FUNCTIONAL_MATCH_THRESHOLD

        result_entry: dict = {
            "id":                item_id,
            "intent":            intent,
            "ground_truth":      ground_truth,
            "generated":         generated,
            "model":             model_id,
            "attempts":          attempts,
            "validation_passed": validation_passed,
            "functional_score":  score,
            "functional_match":  functional_match,
            "functional_detail": detail,
            "difficulty":        item.get("difficulty", "unknown"),
            "tags":              item.get("tags", []),
        }

        # Optional execution comparison
        if execute and validation_passed and generated.strip():
            print(f"\n  🚀 Executing generated script...")
            gen_exec = execute_workflow(generated, intent)
            result_entry["exec_result"] = gen_exec

            if ground_truth.strip():
                print(f"  🚀 Executing ground-truth script...")
                gt_exec = execute_workflow(ground_truth, f"[GT] {intent}")
                result_entry["gt_exec_result"] = gt_exec

                exec_match, exec_reason = compare_execution_results(gen_exec, gt_exec)
                result_entry["execution_match"]  = exec_match
                result_entry["execution_reason"] = exec_reason
                print(f"  {'✅' if exec_match else '❌'} Execution match: {exec_reason}")

        match_icon = "✅" if functional_match else "❌"
        print(f"\n  {match_icon} functional_score={score:.2f}  attempts={attempts}")

        results.append(result_entry)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Generation complete. Results saved to {results_path}")
    return results_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="NLP-Brane-Translator evaluation harness")
    parser.add_argument(
        "--mode", choices=["generate", "score", "full"], default="full",
        help="generate: run model on benchmark; score: compute metrics; full: both (default)."
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-27B",
        help="HuggingFace model id (only used in generate/full modes)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
        help="Sampling temperature (default: 0.4)."
    )
    parser.add_argument(
        "--benchmark", default=str(DEFAULT_BENCHMARK),
        help=f"Path to benchmark JSONL file (default: {DEFAULT_BENCHMARK})."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory. Defaults to evaluation_results/<model_slug>_<date>/."
    )
    parser.add_argument(
        "--results", default=None,
        help="Path to existing results.jsonl (only used in score mode)."
    )
    parser.add_argument(
        "--execute", action="store_true", default=False,
        help="Execute both generated and ground-truth scripts and compare outputs."
    )
    args = parser.parse_args()

    # ── Resolve output directory ──────────────────────────────────────────
    if args.output:
        output_dir = Path(args.output)
    else:
        model_slug = args.model.replace("/", "_").replace(".", "-")
        date_str   = datetime.now().strftime("%Y-%m-%d")
        output_dir = DEFAULT_OUTPUT_DIR / f"{model_slug}_{date_str}"

    # ── Score-only mode ───────────────────────────────────────────────────
    if args.mode == "score":
        results_path = Path(args.results) if args.results else output_dir / "results.jsonl"
        if not results_path.exists():
            print(f"❌ Results file not found: {results_path}")
            print("   Pass --results <path> or run with --mode generate first.")
            sys.exit(1)
        score_mode(results_path, output_dir)
        return

    # ── Generate (or full) mode ───────────────────────────────────────────
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"❌ Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    benchmark = load_benchmark(benchmark_path)
    print(f"📋 Loaded {len(benchmark)} benchmark items from {benchmark_path}")

    results_path = generate_mode(
        benchmark=benchmark,
        output_dir=output_dir,
        model_id=args.model,
        temperature=args.temperature,
        execute=args.execute,
    )

    if args.mode == "full":
        score_mode(results_path, output_dir)


if __name__ == "__main__":
    main()
