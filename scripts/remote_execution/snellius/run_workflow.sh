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

set -euo pipefail

EXECUTOR_URL="${BRANE_EXECUTOR_URL:-http://localhost:${TUNNEL_PORT:-9753}}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <workflow.bs>"
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

RESPONSE=$(curl -sf -X POST "${EXECUTOR_URL}/run" \
    -H "Authorization: Bearer ${BRANE_EXECUTOR_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"workflow\": ${WORKFLOW}, \"query\": \"${BS_FILE}\"}")

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
