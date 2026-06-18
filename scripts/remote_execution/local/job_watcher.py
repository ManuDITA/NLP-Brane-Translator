#!/usr/bin/env python3
"""
job_watcher.py — File-based job queue watcher for Brane remote execution.

No port forwarding needed. Polls ~/brane_jobs/pending/ on Snellius via normal
SSH, downloads job files, runs `brane workflow run` locally, and uploads
results back to ~/brane_jobs/done/ on Snellius.

Flow
----
  [Snellius] pipeline.py / run_workflow.sh
    writes ~/brane_jobs/pending/<uuid>.json
                ↓ SSH poll every ~3s
  [Local] job_watcher.py
    downloads job → runs brane → uploads ~/brane_jobs/done/<uuid>.json
                ↑
  [Snellius] polls ~/brane_jobs/done/<uuid>.json → reads result

Usage
-----
    source .env
    python scripts/remote_execution/local/job_watcher.py

Required environment variables (in .env)
-----------------------------------------
    SNELLIUS_USER          Your Snellius username (e.g. esimeone)
    SNELLIUS_HOST          Snellius login node   (e.g. int5.snellius.surf.nl)

Optional environment variables
-------------------------------
    SNELLIUS_JOBS_DIR      Path on Snellius for the job queue (default: ~/brane_jobs)
    SNELLIUS_SSH_KEY       Path to SSH key for Snellius (default: SSH agent / ~/.ssh/config)
    BRANE_INSTANCE         brane --instance value (default: local-instance)
    BRANE_EXECUTOR_TIMEOUT Max seconds per workflow (default: 300)
    POLL_INTERVAL          Seconds between Snellius polls (default: 3)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SNELLIUS_USER = os.environ.get("SNELLIUS_USER", "")
SNELLIUS_HOST = os.environ.get("SNELLIUS_HOST", "snellius.surf.nl")
SNELLIUS_SSH_KEY = os.environ.get("SNELLIUS_SSH_KEY", "")
SNELLIUS_JOBS_DIR = os.environ.get("SNELLIUS_JOBS_DIR", "~/brane_jobs")
BRANE_INSTANCE = os.environ.get("BRANE_INSTANCE", "local-instance")
BRANE_EXECUTOR_TIMEOUT = int(os.environ.get("BRANE_EXECUTOR_TIMEOUT", "300"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "3"))

REMOTE = f"{SNELLIUS_USER}@{SNELLIUS_HOST}" if SNELLIUS_USER else SNELLIUS_HOST
PENDING_DIR = f"{SNELLIUS_JOBS_DIR}/pending"
DONE_DIR = f"{SNELLIUS_JOBS_DIR}/done"

# Where Brane stores committed results on the local machine
BRANE_DATA_DIR = Path.home() / ".local/share/brane/data"

# Per-file size cap for committed output (bytes); files larger than this are summarised
COMMITTED_FILE_SIZE_LIMIT = 20_000   # 20 KB
# Total size cap across all committed output in one result
COMMITTED_TOTAL_LIMIT = 200_000      # 200 KB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _ssh_cmd() -> list:
    """Base SSH command with common options."""
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    if SNELLIUS_SSH_KEY:
        cmd += ["-i", SNELLIUS_SSH_KEY]
    cmd.append(REMOTE)
    return cmd


def ssh_run(remote_cmd: str, input_data: str = None) -> tuple:
    """Run a command on Snellius. Returns (exit_code, stdout, stderr)."""
    proc = subprocess.run(
        _ssh_cmd() + [remote_cmd],
        capture_output=True,
        text=True,
        input=input_data,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# Committed output collection
# ---------------------------------------------------------------------------
def snapshot_data_dirs() -> float:
    """
    Return the current wall-clock time (float seconds since epoch).
    Used as a "before" marker to detect dataset files written after this point.
    """
    return time.time()


def collect_committed_outputs(before: float) -> dict:
    """
    Detect dataset directories that have files newer than *before* timestamp,
    and read their output files.

    Returns a dict keyed by dataset name:
        {
          "healthcare_reports3": {
              "files": {"summary.html": "<content>", "reports/PAT001.json": "..."}
          }
        }

    Files larger than COMMITTED_FILE_SIZE_LIMIT are truncated.
    Total content is capped at COMMITTED_TOTAL_LIMIT.
    """
    if not BRANE_DATA_DIR.exists():
        return {}

    # Find all dataset dirs that have at least one file newer than `before`
    changed = []
    for p in BRANE_DATA_DIR.iterdir():
        if not p.is_dir():
            continue
        data_sub = p / "data"
        if not data_sub.exists():
            continue
        for fpath in data_sub.rglob("*"):
            if fpath.is_file() and fpath.stat().st_mtime > before:
                changed.append(p.name)
                break

    if not changed:
        return {}

    result = {}
    total_bytes = 0

    for name in sorted(changed):
        data_dir = BRANE_DATA_DIR / name / "data"
        files = {}
        for fpath in sorted(data_dir.rglob("*")):
            if not fpath.is_file():
                continue
            rel = str(fpath.relative_to(data_dir))
            size = fpath.stat().st_size

            if total_bytes >= COMMITTED_TOTAL_LIMIT:
                files[rel] = f"<omitted — total output cap ({COMMITTED_TOTAL_LIMIT // 1024}KB) reached>"
                continue

            if size > COMMITTED_FILE_SIZE_LIMIT:
                files[rel] = (
                    f"<file too large: {size} bytes — first {COMMITTED_FILE_SIZE_LIMIT} chars shown>\n"
                    + fpath.read_text(encoding="utf-8", errors="replace")[:COMMITTED_FILE_SIZE_LIMIT]
                )
            else:
                try:
                    files[rel] = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    files[rel] = f"<read error: {exc}>"
            total_bytes += min(size, COMMITTED_FILE_SIZE_LIMIT)

        result[name] = {"files": files}

    return result


# ---------------------------------------------------------------------------
# Job queue operations (all via SSH to Snellius)
# ---------------------------------------------------------------------------
def setup_dirs() -> None:
    rc, _, err = ssh_run(f"mkdir -p {PENDING_DIR} {DONE_DIR}")
    if rc != 0:
        _log(f"WARNING: Could not create job dirs on Snellius: {err.strip()}")


def list_pending() -> list[str]:
    rc, stdout, _ = ssh_run(f"ls {PENDING_DIR}/*.json 2>/dev/null")
    if rc != 0 or not stdout.strip():
        return []
    return [line.strip() for line in stdout.strip().splitlines() if line.strip()]


def download_job(remote_path: str) -> dict:
    rc, stdout, err = ssh_run(f"cat {remote_path}")
    if rc != 0:
        raise RuntimeError(f"Failed to download {remote_path}: {err.strip()}")
    return json.loads(stdout)


def delete_pending(remote_path: str) -> None:
    ssh_run(f"rm -f {remote_path}")


def upload_result(job_id: str, result: dict) -> None:
    result_json = json.dumps(result)
    rc, _, err = ssh_run(
        f"cat > {DONE_DIR}/{job_id}.json",
        input_data=result_json,
    )
    if rc != 0:
        _log(f"WARNING: Failed to upload result for {job_id}: {err.strip()}")


# ---------------------------------------------------------------------------
# Local Brane execution
# ---------------------------------------------------------------------------
def run_brane(workflow: str) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bs", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(workflow)
        tmp_path = tmp.name

    # Snapshot existing data dirs before execution so we can detect new ones
    before_snapshot = snapshot_data_dirs()

    try:
        cmd = ["brane", "workflow", "run", "a", tmp_path]
        _log(f"Executing: {' '.join(cmd)}")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BRANE_EXECUTOR_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )

        _log(f"exit_code={proc.returncode}  success={proc.returncode == 0}")
        # Classify error type for training data collection:
        #   0  → pass
        #   2  → compilation error (brane rejected the BraneScript syntax)
        #   1  → runtime error (script compiled but a task container failed)
        #  -1  → timeout / brane not found
        if proc.returncode == 0:
            error_type = None
        elif proc.returncode == 2:
            error_type = "compilation"
        else:
            error_type = "runtime"

        committed = collect_committed_outputs(before_snapshot)
        if committed:
            _log(f"  committed datasets: {list(committed.keys())}")

        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "error_type": error_type,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "committed_data": committed,
        }

    except subprocess.TimeoutExpired:
        _log(f"TIMEOUT after {BRANE_EXECUTOR_TIMEOUT}s")
        return {
            "success": False,
            "exit_code": -1,
            "error_type": "timeout",
            "stdout": "",
            "stderr": f"Workflow timed out after {BRANE_EXECUTOR_TIMEOUT}s.",
            "committed_data": {},
        }
    except FileNotFoundError:
        _log("ERROR: 'brane' not found in PATH")
        return {
            "success": False,
            "exit_code": -1,
            "error_type": "runtime",
            "stdout": "",
            "stderr": "'brane' not found in PATH. Is Brane installed?",
            "committed_data": {},
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def process_job(remote_path: str) -> None:
    _log(f"Downloading job: {remote_path}")
    try:
        job = download_job(remote_path)
    except Exception as exc:
        _log(f"ERROR downloading job: {exc}")
        return

    job_id = job.get("id", "unknown")
    workflow = job.get("workflow", "")
    query = job.get("query", "")

    _log(f"Job {job_id[:8]}… — query: {query[:60]!r}")

    if not workflow.strip():
        result = {"success": False, "exit_code": -1, "stdout": "", "stderr": "Empty workflow."}
    else:
        result = run_brane(workflow)

    result["id"] = job_id
    result["executed_at"] = _ts()

    upload_result(job_id, result)
    delete_pending(remote_path)
    _log(f"Job {job_id[:8]}… complete — success={result['success']}")
    if not result['success'] and result.get('stderr'):
        _log(f"  stderr: {result['stderr'][:300]}")


def main() -> None:
    if not SNELLIUS_USER:
        print("ERROR: SNELLIUS_USER not set. Source .env first.", file=sys.stderr)
        sys.exit(1)

    _log(f"Job watcher started")
    _log(f"Polling {REMOTE}:{PENDING_DIR} every {POLL_INTERVAL}s")
    _log(f"Brane instance: {BRANE_INSTANCE}")
    _log(f"Workflow timeout: {BRANE_EXECUTOR_TIMEOUT}s")
    _log("Press Ctrl-C to stop.")

    setup_dirs()

    try:
        while True:
            jobs = list_pending()
            if jobs:
                _log(f"Found {len(jobs)} pending job(s)")
                for job_path in jobs:
                    process_job(job_path)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        _log("Watcher stopped.")


if __name__ == "__main__":
    main()
