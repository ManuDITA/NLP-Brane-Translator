#!/usr/bin/env bash
# run_workflow.sh — Submit a BraneScript file to the local machine via file queue.
#
# Usage (run this ON SNELLIUS):
#   bash scripts/remote_execution/snellius/run_workflow.sh <file.bs> [--intent "description"] [--collect]
#
# Options:
#   --intent "text"   Natural-language description of what the workflow does.
#                     Used as the intent label in training_log.jsonl.
#                     Defaults to the filename if omitted.
#   --collect         Append the result to TRAINING_DATA_DIR/index.jsonl
#                     (default: <project>/training_data). Also activates automatically
#                     if the TRAINING_DATA_DIR env var is already set.
#
# How it works (no port forwarding needed):
#   1. Writes a job file to ~/brane_jobs/pending/<uuid>.json on Snellius filesystem
#   2. job_watcher.py on local machine polls this dir via SSH, runs brane, uploads result
#   3. This script polls ~/brane_jobs/done/<uuid>.json until the result appears
#   4. (Optional) Appends a labelled record to training_log.jsonl
#
# Requirements:
#   - job_watcher.py running on your local machine:
#       source .env && python scripts/remote_execution/local/job_watcher.py

set -eo pipefail

JOBS_DIR="${SNELLIUS_JOBS_DIR:-${HOME}/brane_jobs}"
PENDING_DIR="${JOBS_DIR}/pending"
DONE_DIR="${JOBS_DIR}/done"
TIMEOUT="${BRANE_EXECUTOR_TIMEOUT:-300}"
POLL_INTERVAL=3

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BS_FILE=""
INTENT=""
COLLECT=false

# Auto-enable collection if TRAINING_DATA_DIR is already in the environment
if [[ -n "${TRAINING_DATA_DIR:-}" ]]; then
    COLLECT=true
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --intent)   INTENT="$2"; shift 2 ;;
        --collect)  COLLECT=true; shift ;;
        -*)         echo "Unknown option: $1"; exit 1 ;;
        *)          BS_FILE="$1"; shift ;;
    esac
done

if [[ -z "${BS_FILE}" ]]; then
    echo "Usage: $0 <workflow.bs> [--intent \"description\"] [--collect]"
    exit 1
fi

if [[ ! -f "${BS_FILE}" ]]; then
    echo "❌ File not found: ${BS_FILE}"
    exit 1
fi

# Default intent to filename if not provided
if [[ -z "${INTENT}" ]]; then
    INTENT="${BS_FILE}"
fi

mkdir -p "${PENDING_DIR}" "${DONE_DIR}"

# Write job file via Python (handles JSON encoding cleanly)
JOB_ID=$(python3 -c "import uuid; print(uuid.uuid4())")

python3 -c "
import json, os, sys
job_id = sys.argv[1]
bs_file = sys.argv[2]
query   = sys.argv[3]
workflow = open(bs_file, encoding='utf-8').read()
job = {
    'id': job_id,
    'workflow': workflow,
    'query': query,
    'submitted_at': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
}
pending_dir = os.environ.get('SNELLIUS_JOBS_DIR', os.path.expanduser('~/brane_jobs')) + '/pending'
path = f'{pending_dir}/{job_id}.json'
with open(path, 'w') as f:
    json.dump(job, f)
" "${JOB_ID}" "${BS_FILE}" "${INTENT}"

echo "📤 Job ${JOB_ID:0:8}... submitted ($(wc -l < "${BS_FILE}") lines from ${BS_FILE})"
echo "   Waiting for job_watcher.py on your local machine..."

# ---------------------------------------------------------------------------
# Poll for result
# ---------------------------------------------------------------------------
WAITED=0
while (( WAITED < TIMEOUT )); do
    RESULT_FILE="${DONE_DIR}/${JOB_ID}.json"
    if [[ -f "${RESULT_FILE}" ]]; then
        python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print()
print('── Result ──────────────────────────────────────────')
print(f'  success   : {d.get(\"success\", \"?\")}')
print(f'  exit_code : {d.get(\"exit_code\", \"?\")}')
stdout = d.get('stdout', '').strip()
stderr = d.get('stderr', '').strip()
if stdout:
    print()
    print('── stdout ───────────────────────────────────────────')
    print(stdout)
if stderr and not d.get('success'):
    print()
    print('── stderr ───────────────────────────────────────────')
    print(stderr)
print('─────────────────────────────────────────────────────')
" "${RESULT_FILE}"

        # ── Training data collection ─────────────────────────────────────
        if [[ "${COLLECT}" == "true" ]]; then
            python3 -c "
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[4]), '..', '..', '..', 'src'))
from training_collector import TrainingCollector

result   = json.load(open(sys.argv[1]))
intent   = sys.argv[2]
bs_file  = sys.argv[3]
code     = open(bs_file, encoding='utf-8').read()
committed = result.get('committed_data') or {}

collector = TrainingCollector()

if result.get('success'):
    rec_id = collector.log_pass(
        intent=intent,
        generated_code=code,
        stdout=result.get('stdout', ''),
        committed_data=committed,
    )
    print(f'📊 Logged pass  → {collector.log_file} ({rec_id[:8]}…)')
else:
    rec_id = collector.log_fail(
        intent=intent,
        generated_code=code,
        error_type=result.get('error_type') or 'runtime',
        error_message=result.get('stderr', '')[:500],
        stdout=result.get('stdout', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code'),
        committed_data=committed,
    )
    print(f'📊 Logged fail  → {collector.log_file} ({rec_id[:8]}…)')
" "${RESULT_FILE}" "${INTENT}" "${BS_FILE}" "$0"
        fi
        # ─────────────────────────────────────────────────────────────────

        rm -f "${RESULT_FILE}"
        exit 0
    fi
    sleep "${POLL_INTERVAL}"
    WAITED=$((WAITED + POLL_INTERVAL))
done

echo "❌ Timed out after ${TIMEOUT}s — is job_watcher.py running on your local machine?"
rm -f "${PENDING_DIR}/${JOB_ID}.json"
exit 1
