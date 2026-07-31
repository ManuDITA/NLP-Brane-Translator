#!/bin/bash
# start.sh — Start all local services for NLP-Brane-Translator.
#
# Services started:
#   1. job_watcher.py  — polls Snellius for pending BraneScript jobs, runs them
#                        locally via `brane workflow run`, writes results to
#                        outputs/pipeline/ (dashboard data dir)
#   2. frontend/server.py — Flask dashboard + Generate API on http://localhost:5001
#
# Usage:
#   ./start.sh          start both services (skip if already running)
#   ./start.sh --restart  kill existing processes and restart fresh
#   ./start.sh --stop     stop all services and exit
#
# All output is logged to outputs/logs/.
# PIDs are saved in outputs/logs/{job_watcher,dashboard}.pid.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/outputs/logs"
mkdir -p "${LOG_DIR}"

WATCHER_PID_FILE="${LOG_DIR}/job_watcher.pid"
DASHBOARD_PID_FILE="${LOG_DIR}/dashboard.pid"
WATCHER_LOG="${LOG_DIR}/job_watcher.log"
DASHBOARD_LOG="${LOG_DIR}/dashboard.log"

# ── Helpers ───────────────────────────────────────────────────────────────────

_is_running() {
    local pid_file="$1"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

_stop_service() {
    local name="$1"
    local pid_file="$2"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "  Stopping ${name} (PID ${pid})…"
            kill "${pid}"
            # Wait up to 5 seconds for it to exit
            local i=0
            while kill -0 "${pid}" 2>/dev/null && (( i < 10 )); do
                sleep 0.5; (( i++ ))
            done
            kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" || true
        fi
        rm -f "${pid_file}"
    fi
}

# ── Parse args ────────────────────────────────────────────────────────────────

RESTART=false
STOP=false
for arg in "$@"; do
    case "${arg}" in
        --restart) RESTART=true ;;
        --stop)    STOP=true ;;
    esac
done

# ── Stop mode ─────────────────────────────────────────────────────────────────

if "${STOP}"; then
    echo "🛑 Stopping all services…"
    _stop_service "job_watcher" "${WATCHER_PID_FILE}"
    _stop_service "dashboard"   "${DASHBOARD_PID_FILE}"
    echo "✅ All services stopped."
    exit 0
fi

# ── Load environment ──────────────────────────────────────────────────────────

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a; source "${SCRIPT_DIR}/.env"; set +a
    echo "✅ Loaded .env"
else
    echo "⚠️  No .env found — using existing environment variables."
fi

# ── Validate required vars ────────────────────────────────────────────────────

missing=()
[[ -z "${SNELLIUS_USER:-}" ]]         && missing+=("SNELLIUS_USER")
[[ -z "${SNELLIUS_HOST:-}" ]]         && missing+=("SNELLIUS_HOST")
[[ -z "${SNELLIUS_PROJECT_DIR:-}" ]]  && missing+=("SNELLIUS_PROJECT_DIR")

if (( ${#missing[@]} > 0 )); then
    echo "❌ Missing required env vars: ${missing[*]}"
    echo ""
    echo "   Add them to your .env file:"
    for var in "${missing[@]}"; do
        case "${var}" in
            SNELLIUS_USER)         echo "     export SNELLIUS_USER=<your_username>" ;;
            SNELLIUS_HOST)         echo "     export SNELLIUS_HOST=snellius.surf.nl" ;;
            SNELLIUS_PROJECT_DIR)  echo "     export SNELLIUS_PROJECT_DIR=/path/to/project/on/snellius" ;;
        esac
    done
    exit 1
fi

# ── Python / venv ─────────────────────────────────────────────────────────────

if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
    PYTHON=python
else
    PYTHON="${PYTHON:-python3}"
fi

# ── Restart: kill existing processes first ────────────────────────────────────

if "${RESTART}"; then
    echo "🔄 Restarting services…"
    _stop_service "job_watcher" "${WATCHER_PID_FILE}"
    _stop_service "dashboard"   "${DASHBOARD_PID_FILE}"
fi

# ── Start job_watcher ─────────────────────────────────────────────────────────

if _is_running "${WATCHER_PID_FILE}"; then
    echo "  job_watcher already running (PID $(cat "${WATCHER_PID_FILE}"))"
else
    echo "🚀 Starting job_watcher.py…"
    "${PYTHON}" "${SCRIPT_DIR}/scripts/remote_execution/local/job_watcher.py" \
        >> "${WATCHER_LOG}" 2>&1 &
    echo $! > "${WATCHER_PID_FILE}"
    echo "  PID $(cat "${WATCHER_PID_FILE}") | log → ${WATCHER_LOG}"
fi

# ── Start dashboard (frontend/server.py) ──────────────────────────────────────

if _is_running "${DASHBOARD_PID_FILE}"; then
    echo "  dashboard already running (PID $(cat "${DASHBOARD_PID_FILE}"))"
else
    echo "🚀 Starting frontend/server.py…"
    "${PYTHON}" "${SCRIPT_DIR}/frontend/server.py" \
        >> "${DASHBOARD_LOG}" 2>&1 &
    echo $! > "${DASHBOARD_PID_FILE}"
    echo "  PID $(cat "${DASHBOARD_PID_FILE}") | log → ${DASHBOARD_LOG}"
fi

# ── Health check ──────────────────────────────────────────────────────────────

sleep 2

PORT="${DASHBOARD_PORT:-5001}"
HOST="${DASHBOARD_HOST:-127.0.0.1}"

echo ""
if curl -sf "http://${HOST}:${PORT}/api/health" > /dev/null 2>&1; then
    echo "✅ Dashboard is up → http://${HOST}:${PORT}"
else
    echo "⚠️  Dashboard did not respond yet (may still be starting)"
    echo "   Check: tail -f ${DASHBOARD_LOG}"
fi

if _is_running "${WATCHER_PID_FILE}"; then
    echo "✅ job_watcher is running"
else
    echo "❌ job_watcher failed to start — check: tail -f ${WATCHER_LOG}"
fi

echo ""
echo "📋 Useful commands:"
echo "   tail -f ${WATCHER_LOG}    # job_watcher live log"
echo "   tail -f ${DASHBOARD_LOG}  # dashboard live log"
echo "   ./start.sh --stop         # stop all services"
echo "   ./start.sh --restart      # restart all services"
