#!/usr/bin/env bash
# setup_compute_tunnel.sh — Forward the executor port from a Snellius compute
# node to the login node (which already has the reverse tunnel to local).
#
# Background
# ----------
#   SLURM jobs run on compute nodes, not login nodes. The reverse SSH tunnel
#   (start_tunnel.sh on your local machine) connects TO the login node, making
#   port 9753 available on the LOGIN node's localhost.
#
#   Compute nodes cannot directly reach the login node's localhost, so this
#   script creates a second SSH hop:
#
#     compute:9753  (this script)
#       → login_node:9753  (reverse tunnel endpoint)
#         → local machine:9753  (brane_executor.py)
#
# Usage
# -----
#   Source this file near the TOP of sbatch.sh (before calling pipeline.py):
#
#       source scripts/remote_execution/snellius/setup_compute_tunnel.sh
#
#   The script is smart enough to skip itself if already running on a login node.
#
# Requirements
# ------------
#   - BRANE_EXECUTOR_TOKEN exported in the environment (set in your ~/.bashrc on Snellius)
#   - The SSH reverse tunnel must already be running on your local machine
#   - SSH key-based auth between compute nodes and login nodes (standard on Snellius)

# ── Detect login node ─────────────────────────────────────────────────────
# SLURM_SUBMIT_HOST = hostname of the node that ran `sbatch`; this is the
# login node where the reverse tunnel is listening.
LOGIN_NODE="${SLURM_SUBMIT_HOST:-snellius1.surf.nl}"
TUNNEL_PORT="${TUNNEL_PORT:-9753}"
EXECUTOR_URL="http://localhost:${TUNNEL_PORT}"
MAX_WAIT=30   # seconds to wait for the tunnel to be ready

echo "[tunnel] Login node (reverse tunnel host): ${LOGIN_NODE}"
echo "[tunnel] Forwarding localhost:${TUNNEL_PORT} → ${LOGIN_NODE}:${TUNNEL_PORT}"

# ── Skip if we ARE the login node (e.g. interactive session on login) ─────
if [[ "$(hostname)" == "${LOGIN_NODE}"* ]]; then
    echo "[tunnel] Running on login node — no extra forwarding needed."
    export BRANE_EXECUTOR_URL="${EXECUTOR_URL}"
    return 0 2>/dev/null || true
fi

# ── Check SSH connectivity to the login node ──────────────────────────────
if ! ssh -o "BatchMode=yes" \
         -o "ConnectTimeout=5" \
         -o "StrictHostKeyChecking=no" \
         "${LOGIN_NODE}" true 2>/dev/null; then
    echo "[tunnel] ❌ Cannot SSH to login node ${LOGIN_NODE}."
    echo "[tunnel]    Check that key-based SSH between compute and login nodes is configured."
    echo "[tunnel]    Continuing without executor — workflow will be saved but NOT executed."
    export BRANE_EXECUTOR_URL=""
    return 0 2>/dev/null || true
fi

# ── Start the port-forward in the background ──────────────────────────────
ssh -fNL "${TUNNEL_PORT}:localhost:${TUNNEL_PORT}" \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=5" \
    -o "ExitOnForwardFailure=yes" \
    -o "BatchMode=yes" \
    -o "StrictHostKeyChecking=no" \
    "${LOGIN_NODE}" &

TUNNEL_PID=$!
echo "[tunnel] SSH port-forward started (PID ${TUNNEL_PID})"

# ── Trap to kill the tunnel when the SLURM job exits ─────────────────────
_cleanup_tunnel() {
    echo "[tunnel] Cleaning up SSH port-forward (PID ${TUNNEL_PID})..."
    kill "${TUNNEL_PID}" 2>/dev/null || true
}
trap '_cleanup_tunnel' EXIT INT TERM

# ── Wait until the executor is reachable ─────────────────────────────────
echo "[tunnel] Waiting for executor to become reachable..."
WAITED=0
while (( WAITED < MAX_WAIT )); do
    if curl -sf \
            -H "Authorization: Bearer ${BRANE_EXECUTOR_TOKEN}" \
            "${EXECUTOR_URL}/health" &>/dev/null; then
        echo "[tunnel] ✅ Executor reachable at ${EXECUTOR_URL}"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if (( WAITED >= MAX_WAIT )); then
    echo "[tunnel] ⚠️  Executor NOT reachable after ${MAX_WAIT}s."
    echo "[tunnel]    Is brane_executor.py running on your local machine?"
    echo "[tunnel]    Is start_tunnel.sh running on your local machine?"
    echo "[tunnel]    Workflow will be generated and saved, but NOT executed."
    export BRANE_EXECUTOR_URL=""
else
    export BRANE_EXECUTOR_URL="${EXECUTOR_URL}"
fi
