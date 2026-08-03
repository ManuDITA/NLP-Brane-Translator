#!/usr/bin/env python3
"""
process_snellius.py
───────────────────
One command to go from Snellius outputs → full evaluation results.

Usage
─────
    # Process all *_generated.json files in output_snellius/
    python scripts/process_snellius.py

    # Process a specific file or directory
    python scripts/process_snellius.py output_snellius/qwen3_4b_generated.json

    # Re-execute even if an output file already exists
    python scripts/process_snellius.py --force

    # Print summary of already-executed results without re-running
    python scripts/process_snellius.py --summary

Workflow
────────
  1. Scans output_snellius/ for *_generated.json (Snellius generate-only outputs)
     and any *.json that contain un-executed examples (mode == generate_only).
  2. For each file, runs every generated BraneScript against the local Brane
     instance (parallel, same logic as execute_generated.py).
  3. Saves the full result to outputs/eval/<same_stem_without_generated>.json.
  4. Prints a per-model summary table and an overall comparison.

Output saved to outputs/eval/ is compatible with the frontend eval browser.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNELLIUS_DIR = PROJECT_ROOT / "outputs" / "snellius"
EVAL_DIR     = PROJECT_ROOT / "outputs" / "eval"
EXEC_RESULTS = PROJECT_ROOT / "data" / "training" / "execution_results.jsonl"
TEST_RESULTS = PROJECT_ROOT / "outputs" / "eval" / "test_results.jsonl"

BRANE_INSTANCE = os.environ.get("BRANE_INSTANCE", "local-instance")
BRANE_TIMEOUT  = int(os.environ.get("BRANE_TIMEOUT", "60"))
BRANE_WORKERS  = int(os.environ.get("BRANE_WORKERS", "6"))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from brane_utils import (
    extract_commit_names, pre_clean_committed,
    patch_commit_names, read_and_clear_committed, compare_committed,
)


# ---------------------------------------------------------------------------
# Brane execution
# ---------------------------------------------------------------------------

def _extract_bs(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _run(code: str, timeout: int = BRANE_TIMEOUT) -> dict:
    code = _extract_bs(code)
    if not code or len(code) < 5:
        return {"exit_code": -1, "stdout": "", "stderr": "empty output",
                "success": False, "error_type": "empty", "committed_results": {}}

    commit_names = extract_commit_names(code)
    patched, name_map = patch_commit_names(code) if commit_names else (code, {})
    unique_names = list(name_map.values()) if name_map else commit_names
    pre_clean_committed(unique_names)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bs",
                                     encoding="utf-8", delete=False) as f:
        f.write(patched)
        path = f.name

    try:
        r = subprocess.run(
            ["brane", "workflow", "run", BRANE_INSTANCE, path],
            capture_output=True, text=True, timeout=timeout,
        )
        committed = (read_and_clear_committed(commit_names, name_map=name_map)
                     if r.returncode == 0 else {})
        stderr_lower = r.stderr.lower()
        if r.returncode != 0:
            error_type = "compile" if any(x in stderr_lower for x in (
                "compilation of workflow failed", "parse error",
                "does not exist", "undefined function", "syntax error",
                "could not define variable",
            )) else "runtime"
        else:
            error_type = None
        return {
            "exit_code":         r.returncode,
            "stdout":            r.stdout.strip(),
            "stderr":            r.stderr.strip(),
            "success":           r.returncode == 0,
            "error_type":        error_type,
            "committed_results": committed,
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "TIMEOUT",
                "success": False, "error_type": "timeout", "committed_results": {}}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Reference output loading
# ---------------------------------------------------------------------------

def _load_refs() -> tuple[dict, set]:
    """id → {stdout, committed_results} from execution_results + test_results.
    Also returns a set of time-dependent entry IDs (stdout must not be compared).
    """
    refs: dict = {}
    time_dependent: set[str] = set()
    for src in [EXEC_RESULTS, TEST_RESULTS]:
        if not src.exists():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                eid = r.get("id") or ""
                if r.get("success") and eid and eid not in refs:
                    refs[eid] = {
                        "stdout":            (r.get("stdout") or "").strip(),
                        "committed_results": r.get("committed_results") or {},
                    }
                if r.get("time_dependent") and eid:
                    time_dependent.add(eid)
            except Exception:
                pass
    return refs, time_dependent


# ---------------------------------------------------------------------------
# Execute a single generated file
# ---------------------------------------------------------------------------

def _needs_execution(data: dict) -> bool:
    """True if the file has un-executed examples."""
    mode = data.get("mode", "")
    if "generate" in mode:
        return True
    examples = data.get("examples", [])
    # Also handle files where execution field is missing or empty
    if examples and "execution" not in examples[0]:
        return True
    return False


def execute_file(input_path: Path, refs: dict, time_dependent: set,
                 force: bool = False,
                 workers: int = BRANE_WORKERS,
                 timeout: int = BRANE_TIMEOUT) -> Path | None:
    """Execute a generated file and save result. Returns output path, or None if skipped."""
    data = json.loads(input_path.read_text(encoding="utf-8"))
    examples = data.get("examples", [])
    if not examples:
        print(f"  ⚠️  {input_path.name}: no examples — skipping")
        return None

    # Determine output path
    stem = input_path.stem
    stem_clean = re.sub(r"_generated$", "", stem)
    out_path = EVAL_DIR / f"{stem_clean}.json"

    if out_path.exists() and not force:
        print(f"  ⏭  {input_path.name} → already executed ({out_path.name}), use --force to redo")
        return out_path

    if not _needs_execution(data) and not force:
        print(f"  ⏭  {input_path.name} → mode=full, already has execution results")
        return input_path

    model = data.get("model", input_path.stem)
    print(f"\n{'='*62}")
    print(f"  📂 {input_path.name}")
    print(f"  🤖 Model    : {model}")
    print(f"  📝 Examples : {len(examples)}")
    print(f"  🔗 Brane    : {BRANE_INSTANCE}  (timeout {timeout}s, {workers} workers)")
    print(f"{'='*62}")

    results = []

    def _process(ex):
        eid       = ex.get("id", "")
        generated = ex.get("generated_bs", "") or ex.get("generated_code", "")
        try:
            execution = _run(generated, timeout=timeout)
        except Exception as exc:
            execution = {"exit_code": -1, "stdout": "", "stderr": str(exc),
                         "success": False, "error_type": "execution_error",
                         "committed_results": {}}

        ref            = refs.get(eid) or refs.get(ex.get("intent", ""), {})
        ref_stdout     = ref.get("stdout", "") if ref else ""
        ref_committed  = ref.get("committed_results", {}) if ref else {}
        is_time_dep    = eid in time_dependent

        stdout_match = (execution["stdout"] == ref_stdout
                        if execution["success"] and ref_stdout and not is_time_dep
                        else None)
        committed_match = (compare_committed(ref_committed,
                                             execution.get("committed_results", {}))
                           if execution["success"] and ref_committed else None)
        matches      = [m for m in (stdout_match, committed_match) if m is not None]
        output_match = all(matches) if matches else None

        return {
            "id":              eid,
            "source_file":     ex.get("source_file", ""),
            "intent":          ex.get("intent", ""),
            "reference_bs":    ex.get("reference_bs", ""),
            "generated_bs":    generated,
            "execution":       execution,
            "ref_stdout":      ref_stdout,
            "ref_committed":   ref_committed,
            "stdout_match":    stdout_match,
            "committed_match": committed_match,
            "output_match":    output_match,
        }

    total    = len(examples)
    done_map = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, ex): (i, ex) for i, ex in enumerate(examples, 1)}
        for fut in as_completed(futures):
            i, ex = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "id": ex.get("id", ""), "source_file": ex.get("source_file", ""),
                    "intent": ex.get("intent", ""), "reference_bs": ex.get("reference_bs", ""),
                    "generated_bs": ex.get("generated_bs", ""),
                    "execution": {"exit_code": -1, "stdout": "", "stderr": str(exc),
                                  "success": False, "error_type": "execution_error",
                                  "committed_results": {}},
                    "ref_stdout": "", "ref_committed": {},
                    "stdout_match": None, "committed_match": None, "output_match": None,
                }
            done_map[i] = result
            ok = result["execution"]["success"]
            status = "✅" if ok else (
                "🔶" if result["execution"]["error_type"] == "runtime" else "❌")
            print(f"  [{i:3d}/{total}] {result['intent'][:55]:<55} {status}", flush=True)

    results = [done_map[i] for i in sorted(done_map)]

    # Compute metrics
    n              = len(results)
    compiled       = sum(1 for r in results
                         if r["execution"]["error_type"] not in ("compile", "empty"))
    executed       = sum(1 for r in results if r["execution"]["success"])
    matchable      = [r for r in results if r["output_match"] is not None]
    matched        = sum(1 for r in matchable if r["output_match"])
    c_matchable    = [r for r in results if r.get("committed_match") is not None]
    c_matched      = sum(1 for r in c_matchable if r["committed_match"])

    metrics = {
        **{k: v for k, v in data.items() if k != "examples"},
        "mode":                 "full",
        "total":                n,
        "compile_rate":         round(compiled / n * 100, 1) if n else 0,
        "execution_rate":       round(executed / n * 100, 1) if n else 0,
        "output_match_rate":       round(matched / len(matchable) * 100, 1) if matchable else None,
        "output_match_rate_total": round(matched / n * 100, 1) if n else 0,
        "output_match_n":          len(matchable),
        "committed_match_rate": round(c_matched / len(c_matchable) * 100, 1) if c_matchable else None,
        "committed_match_n":    len(c_matchable),
        "executed_at":          datetime.now(timezone.utc).isoformat(),
        "examples":             results,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))

    print(f"\n  {'─'*40}")
    print(f"  Compile rate  : {metrics['compile_rate']}%")
    print(f"  Exec rate     : {metrics['execution_rate']}%")
    if metrics["output_match_rate"] is not None:
        print(f"  Match rate    : {metrics['output_match_rate']}%  ({metrics['output_match_n']} comparable)")
    print(f"  💾 Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(result_files: list[Path]) -> None:
    rows = []
    for p in sorted(result_files):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "model" not in d or "total" not in d:
            continue
        rows.append({
            "model":    d["model"],
            "total":    d["total"],
            "compile":  d.get("compile_rate"),
            "exec":     d.get("execution_rate"),
            "match":    d.get("output_match_rate"),
            "file":     p.name,
        })

    if not rows:
        print("No completed evaluation results found.")
        return

    print(f"\n{'═'*78}")
    print(f"  {'MODEL':<38} {'TOTAL':>5}  {'COMPILE':>7}  {'EXEC':>5}  {'MATCH':>6}")
    print(f"  {'─'*38} {'─'*5}  {'─'*7}  {'─'*5}  {'─'*6}")
    for r in rows:
        compile_s = f"{r['compile']:5.1f}%" if r['compile'] is not None else "   n/a"
        exec_s    = f"{r['exec']:5.1f}%"    if r['exec']    is not None else "  n/a"
        match_s   = f"{r['match']:5.1f}%"   if r['match']   is not None else "  n/a"
        print(f"  {r['model']:<38} {r['total']:>5}  {compile_s:>7}  {exec_s:>5}  {match_s:>6}")
    print(f"{'═'*78}\n")


# ---------------------------------------------------------------------------
# Find candidate files
# ---------------------------------------------------------------------------

def _find_generated(paths: list[Path]) -> list[Path]:
    """Resolve user-provided paths or scan SNELLIUS_DIR for generated files."""
    if paths:
        found = []
        for p in paths:
            if p.is_dir():
                found.extend(sorted(p.glob("*.json")))
            else:
                found.append(p)
        return found

    if not SNELLIUS_DIR.exists():
        print(f"⚠️  {SNELLIUS_DIR} does not exist — create it and drop Snellius output files there.")
        return []

    files = sorted(f for f in SNELLIUS_DIR.glob("*.json")
                   if f.name != "README.md")
    if not files:
        print(f"⚠️  No .json files found in {SNELLIUS_DIR}")
        print( "   Copy Snellius *_generated.json files there, then re-run.")
    return files


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute Snellius-generated BraneScripts and produce evaluation results.")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="Specific file(s) or dir to process (default: output_snellius/)")
    parser.add_argument("--force", action="store_true",
                        help="Re-execute even if output already exists")
    parser.add_argument("--summary", action="store_true",
                        help="Only print summary of existing outputs/eval/*.json, no execution")
    parser.add_argument("--workers", type=int, default=BRANE_WORKERS,
                        help=f"Parallel Brane workers (default: {BRANE_WORKERS})")
    parser.add_argument("--timeout", type=int, default=BRANE_TIMEOUT,
                        help=f"Per-script timeout in seconds (default: {BRANE_TIMEOUT})")
    args = parser.parse_args()

    if args.summary:
        print_summary(list(EVAL_DIR.glob("*.json")))
        return

    gen_files = _find_generated(args.paths)
    if not gen_files:
        sys.exit(0)

    refs, time_dependent = _load_refs()
    print(f"📚 Loaded {len(refs)} reference outputs ({len(time_dependent)} time-dependent, stdout not compared)")

    out_paths = []
    for f in gen_files:
        if not f.suffix == ".json":
            continue
        try:
            json.loads(f.read_text(encoding="utf-8"))  # validate it's JSON
        except Exception:
            print(f"  ⚠️  {f.name}: not valid JSON — skipping")
            continue
        out = execute_file(f, refs, time_dependent, force=args.force,
                           workers=args.workers, timeout=args.timeout)
        if out:
            out_paths.append(out)

    # Final summary across all processed files + any existing ones
    all_results = sorted(set(out_paths) | set(EVAL_DIR.glob("*.json")))
    all_results = [p for p in all_results if "checkpoint" not in p.name
                   and "test_" not in p.name]
    if all_results:
        print("\n📊 Overall evaluation summary:")
        print_summary(all_results)


if __name__ == "__main__":
    main()
