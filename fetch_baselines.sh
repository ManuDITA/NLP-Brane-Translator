#!/bin/bash
# fetch_baselines.sh — Fetch Snellius baseline results and process them locally.
#
# Copies *_generated.json files from Snellius → outputs/snellius/
# then runs process_snellius.py to execute the BraneScripts and
# produce full evaluation results in outputs/eval/.
#
# Usage (run on your LOCAL machine after submit_baselines.sh jobs finish):
#   bash fetch_baselines.sh
#
# Optional: filter by a timestamp prefix to only grab today's results:
#   SINCE=20260731 bash fetch_baselines.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Load .env ─────────────────────────────────────────────────────────────────
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

SNELLIUS_USER="${SNELLIUS_USER:?ERROR: SNELLIUS_USER not set in .env}"
SNELLIUS_HOST="${SNELLIUS_HOST:-snellius.surf.nl}"
SNELLIUS_PROJECT_DIR="${SNELLIUS_PROJECT_DIR:?ERROR: SNELLIUS_PROJECT_DIR not set in .env}"
SSH_KEY="${SNELLIUS_SSH_KEY:-}"
SINCE="${SINCE:-}"                  # optional timestamp prefix filter (e.g. 20260731)

REMOTE="${SNELLIUS_USER}@${SNELLIUS_HOST}"
REMOTE_EVAL="${SNELLIUS_PROJECT_DIR}/outputs/eval"
LOCAL_SNELLIUS="${SCRIPT_DIR}/outputs/snellius"

mkdir -p "${LOCAL_SNELLIUS}"

SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=10)
[[ -n "${SSH_KEY}" ]] && SSH_ARGS+=(-i "${SSH_KEY}")

# ── Python / venv ─────────────────────────────────────────────────────────────
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi
PYTHON="${PYTHON:-python3}"

# ── Check what's on Snellius ──────────────────────────────────────────────────
echo "🔍 Scanning ${REMOTE}:${REMOTE_EVAL}/ for generated files…"

if [[ -n "${SINCE}" ]]; then
    PATTERN="*${SINCE}*_generated.json"
    echo "   Filtering by: ${PATTERN}"
else
    PATTERN="*_generated.json"
fi

# List matching files on Snellius
FILES=$(ssh "${SSH_ARGS[@]}" "${REMOTE}" "ls ${REMOTE_EVAL}/${PATTERN} 2>/dev/null || true")

if [[ -z "${FILES}" ]]; then
    echo "❌ No *_generated.json files found in ${REMOTE_EVAL}/"
    echo "   Wait for SLURM jobs to finish, then re-run."
    exit 1
fi

echo "   Found:"
echo "${FILES}" | while read -r f; do printf "     %s\n" "$(basename "${f}")"; done
echo ""

# ── SCP files locally ─────────────────────────────────────────────────────────
echo "📥 Copying to ${LOCAL_SNELLIUS}/"
scp "${SSH_ARGS[@]}" "${REMOTE}:${REMOTE_EVAL}/${PATTERN}" "${LOCAL_SNELLIUS}/"
echo "   Done."
echo ""

# ── Run process_snellius.py ───────────────────────────────────────────────────
echo "⚙️  Processing with process_snellius.py…"
echo "──────────────────────────────────────────────────────"
"${PYTHON}" "${SCRIPT_DIR}/scripts/process_snellius.py"

echo ""
echo "✅ Done. Results saved to outputs/eval/"
echo "   Open http://localhost:5001 → Evaluation tab to compare models."
