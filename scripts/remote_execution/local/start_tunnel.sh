#!/usr/bin/env bash
# start_tunnel.sh — Maintain a reverse SSH tunnel from this machine to Snellius.
#
# What this does
# --------------
#   Creates a reverse port forward:
#     Snellius login node :9753  →  this machine :9753
#
#   When the Snellius compute node forwards its own localhost:9753 through the
#   login node (via setup_compute_tunnel.sh), the full path becomes:
#     compute node :9753  →  login node :9753  →  this machine :9753
#
# Usage
# -----
#   # One-time setup first:
#   bash scripts/remote_execution/local/setup_local.sh
#
#   # Start the tunnel (runs in foreground; use a tmux/screen session or systemd):
#   bash scripts/remote_execution/local/start_tunnel.sh
#
# Configuration (in .env or exported before running)
# ---------------------------------------------------
#   SNELLIUS_USER      Your Snellius username (default: $USER)
#   SNELLIUS_HOST      Snellius login node (default: snellius.surf.nl)
#   TUNNEL_PORT        Port to forward (default: 9753)
#   TUNNEL_SSH_KEY     Path to the dedicated tunnel SSH key (default: ~/.ssh/brane_tunnel_key)

set -euo pipefail

# ── Load .env if present ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source <(grep -E '^[A-Z_]+=.*' "${ENV_FILE}" | grep -v '^#')
fi

# ── Configuration ─────────────────────────────────────────────────────────
SNELLIUS_USER="${SNELLIUS_USER:-${USER}}"
SNELLIUS_HOST="${SNELLIUS_HOST:-snellius.surf.nl}"
TUNNEL_PORT="${TUNNEL_PORT:-9753}"
TUNNEL_SSH_KEY="${TUNNEL_SSH_KEY:-${HOME}/.ssh/brane_tunnel_key}"
RETRY_DELAY="${RETRY_DELAY:-10}"    # seconds between reconnect attempts

# ── Validate SSH key ──────────────────────────────────────────────────────
if [[ ! -f "${TUNNEL_SSH_KEY}" ]]; then
    echo "❌ SSH key not found: ${TUNNEL_SSH_KEY}"
    echo "   Run setup_local.sh first to generate the key."
    exit 1
fi

REMOTE="${SNELLIUS_USER}@${SNELLIUS_HOST}"

echo "═══════════════════════════════════════════════════════"
echo " Brane Reverse SSH Tunnel"
echo "═══════════════════════════════════════════════════════"
echo " Local port  : 127.0.0.1:${TUNNEL_PORT}"
echo " Remote      : ${REMOTE}"
echo " Remote port : localhost:${TUNNEL_PORT} (login node)"
echo " SSH key     : ${TUNNEL_SSH_KEY}"
echo "═══════════════════════════════════════════════════════"
echo ""
echo " The tunnel will auto-reconnect on disconnect."
echo " Press Ctrl-C to stop."
echo ""

# ── Common SSH options ────────────────────────────────────────────────────
SSH_OPTS=(
    -i "${TUNNEL_SSH_KEY}"
    -o "ServerAliveInterval=30"       # send keepalive every 30s
    -o "ServerAliveCountMax=3"        # disconnect after 3 missed keepalives
    -o "ExitOnForwardFailure=yes"     # fail fast if port forward can't bind
    -o "StrictHostKeyChecking=accept-new"  # auto-accept new host keys on first connect
    -o "BatchMode=yes"                # never prompt for a password
    -N                                # no shell, just port forwarding
    # Reverse forward: remote_port:local_host:local_port
    -R "${TUNNEL_PORT}:localhost:${TUNNEL_PORT}"
)

# ── Use autossh if available (more reliable reconnect handling) ───────────
if command -v autossh &>/dev/null; then
    echo "✅ Using autossh for reliable keepalive monitoring"
    export AUTOSSH_GATETIME=0       # don't count first connection time
    export AUTOSSH_POLL=30          # polling interval
    export AUTOSSH_LOGFILE=""       # log to stderr

    # autossh uses a monitoring port pair; 0 disables the extra SSH monitoring
    # connection and relies on ServerAlive instead.
    exec autossh -M 0 "${SSH_OPTS[@]}" "${REMOTE}"
fi

# ── Fallback: plain SSH with retry loop ───────────────────────────────────
echo "ℹ️  autossh not found — using plain SSH with retry loop"
echo "   (Install autossh for more robust reconnection: sudo apt install autossh)"
echo ""

ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${TIMESTAMP}] Connecting... (attempt ${ATTEMPT})"

    # Run ssh; capture exit code without errexit killing us
    ssh "${SSH_OPTS[@]}" "${REMOTE}" || EXIT_CODE=$?

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${TIMESTAMP}] SSH exited with code ${EXIT_CODE:-0}. Reconnecting in ${RETRY_DELAY}s..."
    sleep "${RETRY_DELAY}"
done
