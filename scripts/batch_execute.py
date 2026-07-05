#!/usr/bin/env python3
"""
batch_execute.py

Runs every BraneScript example found in data/examples/*.jsonl through
  brane workflow run local-instance <file>
and saves the complete output (stdout, stderr, exit_code, timing) to
  training_data/execution_results.jsonl

Usage:
    python scripts/batch_execute.py [--timeout 60] [--resume]

Options:
    --timeout N   Per-script timeout in seconds (default: 60)
    --resume      Skip examples that already have a result entry
    --dry-run     Parse BraneScript only (no Docker), instant but no real output
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
EXAMPLES_GLOB  = str(PROJECT_ROOT / "data" / "examples" / "*.jsonl")
RESULTS_FILE   = PROJECT_ROOT / "training_data" / "execution_results.jsonl"
BRANE_INSTANCE = "local-instance"
DEFAULT_TIMEOUT = 60   # seconds per script

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_examples() -> list[dict]:
    """Load all examples from data/examples/*.jsonl, assign stable IDs."""
    examples = []
    for path in sorted(glob.glob(EXAMPLES_GLOB)):
        fname = Path(path).stem
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  JSON error in {fname}:{lineno} — {e}", flush=True)
                    continue
                ex["id"]          = f"{fname}-{lineno:04d}"
                ex["source_file"] = Path(path).name
                examples.append(ex)
    return examples


def load_done_ids(results_file: Path) -> set[str]:
    """Return IDs already present in the results file."""
    done = set()
    if not results_file.exists():
        return done
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def run_script(bs_code: str, timeout: int, dry_run: bool = False) -> dict:
    """
    Write bs_code to a temp file, run brane workflow run, return result dict.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bs", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(bs_code)
        tmp_path = tmp.name

    flags = ["--dry-run"] if dry_run else []
    cmd = ["brane", "workflow", "run"] + flags + [BRANE_INSTANCE, tmp_path]

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        return {
            "stdout":          proc.stdout,
            "stderr":          proc.stderr,
            "exit_code":       proc.returncode,
            "success":         proc.returncode == 0,
            "timed_out":       False,
            "execution_time_s": round(elapsed, 3),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return {
            "stdout":          "",
            "stderr":          f"TIMEOUT after {timeout}s",
            "exit_code":       -1,
            "success":         False,
            "timed_out":       True,
            "execution_time_s": round(elapsed, 3),
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return {
            "stdout":          "",
            "stderr":          str(exc),
            "exit_code":       -2,
            "success":         False,
            "timed_out":       False,
            "execution_time_s": round(elapsed, 3),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch-execute all BraneScript examples")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Per-script timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip examples already present in the results file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pass --dry-run to brane (no Docker, instant)")
    parser.add_argument("--filter", type=str, default="",
                        help="Only run examples from files matching this substring")
    args = parser.parse_args()

    examples = load_examples()
    if args.filter:
        examples = [e for e in examples if args.filter in e["source_file"]]

    done_ids = load_done_ids(RESULTS_FILE) if args.resume else set()

    pending = [e for e in examples if e["id"] not in done_ids]
    total   = len(pending)

    print(f"📋 Loaded {len(examples)} examples  |  Pending: {total}  |  Already done: {len(done_ids)}")
    print(f"⏱  Timeout: {args.timeout}s  |  Dry-run: {args.dry_run}  |  Results → {RESULTS_FILE}")
    print()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    passed = failed = timed_out = 0
    t_batch_start = time.monotonic()

    with open(RESULTS_FILE, "a", encoding="utf-8") as out_f:
        for idx, ex in enumerate(pending, start=1):
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"[{idx:4d}/{total}] {ex['id']}  ", end="", flush=True)

            result = run_script(ex["branescript"], timeout=args.timeout, dry_run=args.dry_run)

            # Build full record
            record = {
                "id":              ex["id"],
                "source_file":     ex["source_file"],
                "intent":          ex["intent"],
                "branescript":     ex["branescript"],
                "stdout":          result["stdout"],
                "stderr":          result["stderr"],
                "exit_code":       result["exit_code"],
                "success":         result["success"],
                "timed_out":       result["timed_out"],
                "execution_time_s": result["execution_time_s"],
                "timestamp":       ts,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            if result["success"]:
                passed += 1
                status = "✅"
            elif result["timed_out"]:
                timed_out += 1
                status = "⏰"
            else:
                failed += 1
                status = "❌"

            print(f"{status}  {result['execution_time_s']:.1f}s  "
                  f"exit={result['exit_code']}", flush=True)

    elapsed_total = time.monotonic() - t_batch_start
    print()
    print(f"{'='*60}")
    print(f"✅  Passed:    {passed}")
    print(f"❌  Failed:    {failed}")
    print(f"⏰  Timed out: {timed_out}")
    print(f"📝  Total:     {total}")
    print(f"⏱  Wall time: {elapsed_total/60:.1f} min")
    print(f"💾  Results  → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
