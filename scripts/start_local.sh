#!/usr/bin/env bash
# start_local.sh — Start all Brane local services:
#
#   1. job_watcher.py   Polls Snellius, runs brane workflows, saves results
#   2. Dashboard server Visualises results at http://localhost:5001
#
# Usage:
#   bash scripts/start_local.sh
#
# Logs are written to outputs/logs/job_watcher.log and outputs/logs/dashboard.log.
# Press Ctrl-C to stop watching logs (services keep running in background).
# To stop the services: kill $(cat outputs/logs/job_watcher.pid) $(cat outputs/logs/dashboard.pid)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"
WATCHER_SCRIPT="${PROJECT_ROOT}/scripts/remote_execution/local/job_watcher.py"
DASHBOARD_SCRIPT="${FRONTEND_DIR}/server.py"

mkdir -p "${LOG_DIR}"

# ── Load .env ────────────────────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a; source "${PROJECT_ROOT}/.env"; set +a
  echo "✅ Loaded .env"
else
  echo "⚠️  .env not found — continuing without it (SNELLIUS_USER etc. may not be set)"
fi

# ── Validate required env ────────────────────────────────────────────────────
if [[ -z "${SNELLIUS_USER:-}" ]]; then
  echo "❌ SNELLIUS_USER is not set. Set it in .env or export it before running."
  exit 1
fi

# ── Python interpreter ───────────────────────────────────────────────────────
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi
echo "🐍 Python: ${PYTHON}"

# ── Install frontend deps if missing ─────────────────────────────────────────
if ! "${PYTHON}" -c "import flask, yaml" 2>/dev/null; then
  echo "📦 Installing Flask and PyYAML for the dashboard..."
  "${PYTHON}" -m pip install -q flask pyyaml
fi

# ── Stop any previously running services (by saved PID) ──────────────────────
for pidfile in "${LOG_DIR}/job_watcher.pid" "${LOG_DIR}/dashboard.pid"; do
  if [[ -f "${pidfile}" ]]; then
    old_pid="$(cat "${pidfile}")"
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "🔄 Stopping old process (PID ${old_pid})..."
      kill "${old_pid}" 2>/dev/null || true
      sleep 1
    fi
    rm -f "${pidfile}"
  fi
done

# ── Start job_watcher ─────────────────────────────────────────────────────────
echo ""
echo "▶  Starting job_watcher.py..."
"${PYTHON}" "${WATCHER_SCRIPT}" \
  >> "${LOG_DIR}/job_watcher.log" 2>&1 &
JW_PID=$!
echo ${JW_PID} > "${LOG_DIR}/job_watcher.pid"
echo "   PID ${JW_PID}  →  ${LOG_DIR}/job_watcher.log"

# ── Start dashboard server ────────────────────────────────────────────────────
echo "▶  Starting dashboard server..."
"${PYTHON}" "${DASHBOARD_SCRIPT}" \
  >> "${LOG_DIR}/dashboard.log" 2>&1 &
DASH_PID=$!
echo ${DASH_PID} > "${LOG_DIR}/dashboard.pid"
echo "   PID ${DASH_PID}  →  ${LOG_DIR}/dashboard.log"

# ── Wait a moment and verify processes are alive ─────────────────────────────
sleep 2

all_ok=true
if ! kill -0 "${JW_PID}" 2>/dev/null; then
  echo "❌ job_watcher failed to start — check ${LOG_DIR}/job_watcher.log"
  all_ok=false
fi
if ! kill -0 "${DASH_PID}" 2>/dev/null; then
  echo "❌ dashboard failed to start — check ${LOG_DIR}/dashboard.log"
  all_ok=false
fi

if ${all_ok}; then
  DASHBOARD_PORT="${DASHBOARD_PORT:-5001}"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  ✅ All services running"
  echo ""
  echo "  📊 Dashboard   →  http://localhost:${DASHBOARD_PORT}"
  echo "  🔍 job_watcher →  PID ${JW_PID}"
  echo ""
  echo "  To stop:  kill ${JW_PID} ${DASH_PID}"
  echo "            (or: kill \$(cat outputs/logs/job_watcher.pid) \$(cat outputs/logs/dashboard.pid))"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "  Tailing logs (Ctrl-C stops watching, services keep running)..."
  echo ""
  tail -f "${LOG_DIR}/job_watcher.log" "${LOG_DIR}/dashboard.log"
fi
