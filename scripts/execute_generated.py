"""
execute_generated.py

Takes a *_generated.json produced by evaluate.py --generate-only on Snellius
and runs each generated BraneScript against the local Brane instance to fill
in the execution metrics.

Usage
─────
    python scripts/execute_generated.py outputs/eval/qwen3_4b_*_generated.json

    # Specify a different Brane instance
    BRANE_INSTANCE=my-instance python scripts/execute_generated.py <file>

Output
──────
    Writes <same_path_without _generated>.json  (same format as a full evaluate.py run)
    Also prints a metrics summary table.
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

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
EXEC_RESULTS   = PROJECT_ROOT / "data" / "training" / "execution_results.jsonl"
TEST_RESULTS   = PROJECT_ROOT / "outputs" / "eval" / "test_results.jsonl"

BRANE_INSTANCE = os.environ.get("BRANE_INSTANCE", "local-instance")
BRANE_TIMEOUT  = 30
BRANE_WORKERS  = 8

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from brane_utils import (
    extract_commit_names, pre_clean_committed,
    patch_commit_names, read_and_clear_committed, compare_committed,
)


# ---------------------------------------------------------------------------
# Brane execution (same logic as evaluate.py)
# ---------------------------------------------------------------------------

def _extract_branescript(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _run_branescript(code: str) -> dict:
    code = _extract_branescript(code)
    if not code or len(code) < 5:
        return {"exit_code": -1, "stdout": "", "stderr": "empty output",
                "success": False, "error_type": "empty", "committed_results": {}}

    commit_names = extract_commit_names(code)

    # Rewrite commit names to be unique so parallel workers never collide,
    # then pre-clean those unique names as a safety net for crashed prior runs.
    patched_code, name_map = patch_commit_names(code) if commit_names else (code, {})
    unique_names = list(name_map.values()) if name_map else commit_names
    pre_clean_committed(unique_names)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bs", encoding="utf-8", delete=False
    ) as f:
        f.write(patched_code)
        path = f.name

    try:
        result = subprocess.run(
            ["brane", "workflow", "run", BRANE_INSTANCE, path],
            capture_output=True, text=True, timeout=BRANE_TIMEOUT,
        )
        # Read unique-named directories; results are keyed by original names
        committed = (
            read_and_clear_committed(commit_names, name_map=name_map)
            if result.returncode == 0 else {}
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
            "exit_code":         result.returncode,
            "stdout":            result.stdout.strip(),
            "stderr":            result.stderr.strip(),
            "success":           result.returncode == 0,
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


def _load_reference_outputs() -> dict[str, dict]:
    """Return {intent: {stdout, committed_results}} from ground-truth files."""
    refs: dict[str, dict] = {}
    for src in [EXEC_RESULTS, TEST_RESULTS]:
        if not src.exists():
            continue
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("success") and r.get("intent") and r["intent"] not in refs:
                    refs[r["intent"]] = {
                        "stdout":            (r.get("stdout") or "").strip(),
                        "committed_results": r.get("committed_results") or {},
                    }
            except Exception:
                pass
    return refs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def execute_generated(input_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    examples = data.get("examples", [])
    if not examples:
        print(f"❌ No examples found in {input_path}")
        sys.exit(1)

    ref_outs = _load_reference_outputs()
    print(f"\n{'='*60}")
    print(f"🔍 Executing generated scripts")
    print(f"   Source   : {input_path.name}")
    print(f"   Model    : {data.get('model', '?')}")
    print(f"   Examples : {len(examples)}")
    print(f"   Brane    : {BRANE_INSTANCE}")
    print(f"{'='*60}")

    results = []

    def _process(ex):
        intent    = ex["intent"]
        generated = ex.get("generated_bs", "")
        try:
            execution = _run_branescript(generated)
        except Exception as exc:
            execution = {"exit_code": -1, "stdout": "", "stderr": str(exc),
                         "success": False, "error_type": "execution_error",
                         "committed_results": {}}

        ref       = ref_outs.get(intent, {})
        ref_stdout     = ref.get("stdout", "")
        ref_committed  = ref.get("committed_results", {})

        # Per-output match flags
        stdout_match = (
            execution["stdout"] == ref_stdout
            if execution["success"] and ref_stdout else None
        )
        committed_match = (
            compare_committed(ref_committed, execution.get("committed_results", {}))
            if execution["success"] else None
        )

        # Overall match: True only if every available comparison passes
        matches = [m for m in (stdout_match, committed_match) if m is not None]
        output_match = all(matches) if matches else None

        return {
            "id":              ex.get("id", ""),
            "source_file":     ex.get("source_file", ""),
            "intent":          intent,
            "reference_bs":    ex.get("reference_bs", ""),
            "generated_bs":    generated,
            "execution":       execution,
            "ref_stdout":      ref_stdout,
            "ref_committed":   ref_committed,
            "stdout_match":    stdout_match,
            "committed_match": committed_match,
            "output_match":    output_match,
        }

    # Run with thread pool (each worker calls brane CLI)
    pending  = list(enumerate(examples, 1))
    total    = len(pending)
    done_map = {}

    with ThreadPoolExecutor(max_workers=BRANE_WORKERS) as pool:
        futures = {pool.submit(_process, ex): (i, ex) for i, ex in pending}
        for fut in as_completed(futures):
            i, ex = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                # Catch anything that slipped past _process (shouldn't happen,
                # but guarantees no single example kills the whole run)
                print(f"⚠️  [{i}/{total}] unhandled error — skipping: {exc}", flush=True)
                result = {
                    "id":              ex.get("id", ""),
                    "source_file":     ex.get("source_file", ""),
                    "intent":          ex.get("intent", ""),
                    "reference_bs":    ex.get("reference_bs", ""),
                    "generated_bs":    ex.get("generated_bs", ""),
                    "execution":       {"exit_code": -1, "stdout": "", "stderr": str(exc),
                                        "success": False, "error_type": "execution_error",
                                        "committed_results": {}},
                    "ref_stdout":      "",
                    "ref_committed":   {},
                    "stdout_match":    None,
                    "committed_match": None,
                    "output_match":    None,
                }
            done_map[i] = result
            status = "✅" if result["execution"]["success"] else (
                "🔶" if result["execution"]["error_type"] == "runtime" else "❌")
            print(f"[{i:3d}/{total}] {result['intent'][:60]}… {status}", flush=True)

    results = [done_map[i] for i in sorted(done_map)]

    # Metrics
    n         = len(results)
    compiled  = sum(1 for r in results if r["execution"]["error_type"] not in ("compile", "empty"))
    executed  = sum(1 for r in results if r["execution"]["success"])
    matchable = [r for r in results if r["output_match"] is not None]
    matched   = sum(1 for r in matchable if r["output_match"])
    commit_matchable = [r for r in results if r.get("committed_match") is not None]
    commit_matched   = sum(1 for r in commit_matchable if r["committed_match"])

    metrics = {
        **{k: v for k, v in data.items() if k != "examples"},
        "mode":                    "full",
        "total":                   n,
        "compile_rate":            round(compiled / n * 100, 1),
        "execution_rate":          round(executed / n * 100, 1),
        "output_match_rate":       round(matched / len(matchable) * 100, 1) if matchable else None,
        "output_match_n":          len(matchable),
        "committed_match_rate":    round(commit_matched / len(commit_matchable) * 100, 1) if commit_matchable else None,
        "committed_match_n":       len(commit_matchable),
        "executed_at":             datetime.now(timezone.utc).isoformat(),
        "examples":                results,
    }

    # Print summary
    print(f"\n{'─'*40}")
    print(f"  Model            : {metrics['model']}")
    print(f"  Total examples   : {n}")
    print(f"  Compile rate     : {metrics['compile_rate']}%")
    print(f"  Execution rate   : {metrics['execution_rate']}%")
    if metrics["output_match_rate"] is not None:
        print(f"  Output match rate: {metrics['output_match_rate']}%  ({metrics['output_match_n']} comparable)")
    if metrics["committed_match_rate"] is not None:
        print(f"  Commit match rate: {metrics['committed_match_rate']}%  ({metrics['committed_match_n']} comparable)")
    print(f"{'─'*40}\n")

    # Save — strip _generated suffix if present
    out_path = Path(str(input_path).replace("_generated.json", ".json"))
    if out_path == input_path:
        stem     = input_path.stem
        out_path = input_path.parent / f"{stem}_executed.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print(f"💾 Full results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute BraneScripts generated by evaluate.py --generate-only")
    parser.add_argument("input", nargs="+",
                        help="Path(s) to *_generated.json files from evaluate.py")
    args = parser.parse_args()

    for p in args.input:
        execute_generated(Path(p))
