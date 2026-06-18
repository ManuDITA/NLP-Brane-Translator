#!/usr/bin/env bash
# run_workflow.sh — Send a BraneScript file to the local executor via SSH tunnel.
#
# Usage (run this ON SNELLIUS):
#   bash scripts/remote_execution/snellius/run_workflow.sh <file.bs>
#
# Requirements:
#   - BRANE_EXECUTOR_TOKEN exported in your environment
#   - start_tunnel.sh running on your local machine
#   - brane_executor.py running on your local machine

set -eo pipefail

EXECUTOR_URL="${BRANE_EXECUTOR_URL:-http://localhost:${TUNNEL_PORT:-9753}}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <workflow.bs>"
    exit 1
fi

if [[ -z "${BRANE_EXECUTOR_TOKEN:-}" ]]; then
    echo "❌ BRANE_EXECUTOR_TOKEN is not set."
    echo "   Run: export BRANE_EXECUTOR_TOKEN=<your_token>"
    echo "   Or add it to ~/.bashrc on Snellius."
    exit 1
fi

BS_FILE="$1"

if [[ ! -f "${BS_FILE}" ]]; then
    echo "❌ File not found: ${BS_FILE}"
    exit 1
fi

# Read file and JSON-encode it for the request body
WORKFLOW=$(python3 -c "import json,sys; print(json.dumps(open(sys.argv[1]).read()))" "${BS_FILE}")

echo "📤 Sending $(wc -l < "${BS_FILE}") lines from ${BS_FILE} to ${EXECUTOR_URL}/run ..."

# -s = silent, but NOT -f so we see HTTP error bodies too
RESPONSE=$(curl -s --max-time 320 -X POST "${EXECUTOR_URL}/run" \
    -H "Authorization: Bearer ${BRANE_EXECUTOR_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"workflow\": ${WORKFLOW}, \"query\": \"${BS_FILE}\"}" 2>&1) || true

if [[ -z "${RESPONSE}" ]]; then
    echo ""
    echo "❌ No response from executor at ${EXECUTOR_URL}"
    echo "   Check on your local machine:"
    echo "     1. Is brane_executor.py running?  (Terminal 1)"
    echo "     2. Is start_tunnel.sh connected?  (Terminal 2)"
    echo "     3. Run: curl -s http://localhost:9753/health  (from Snellius login node)"
    exit 1
fi

SUCCESS=$(echo "${RESPONSE}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('success','?'))")
EXIT_CODE=$(echo "${RESPONSE}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('exit_code','?'))")
STDOUT=$(echo "${RESPONSE}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('stdout',''))")
STDERR=$(echo "${RESPONSE}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('stderr',''))")

echo ""
echo "── Result ──────────────────────────────────────────"
echo "  success   : ${SUCCESS}"
echo "  exit_code : ${EXIT_CODE}"
if [[ -n "${STDOUT}" ]]; then
    echo ""
    echo "── stdout ───────────────────────────────────────────"
    echo "${STDOUT}"
fi
if [[ -n "${STDERR}" && "${SUCCESS}" != "True" ]]; then
    echo ""
    echo "── stderr ───────────────────────────────────────────"
    echo "${STDERR}"
fi
echo "─────────────────────────────────────────────────────"
