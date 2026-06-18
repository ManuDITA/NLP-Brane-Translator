#!/usr/bin/env bash
# setup_local.sh — One-time setup for the local Brane executor.
#
# What this does
# --------------
#   1. Generates a dedicated SSH key pair for the tunnel (brane_tunnel_key).
#   2. Generates a random Bearer token for HTTP authentication.
#   3. Creates / updates the project .env file with these values.
#   4. Prints the public key and instructions for adding it to Snellius.
#
# Run this ONCE on your local machine before using the remote execution system.
#
# Usage
# -----
#   bash scripts/remote_execution/local/setup_local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

echo "═══════════════════════════════════════════════════════"
echo " Brane Remote Execution — Local Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Step 1: SSH key ───────────────────────────────────────────────────────
KEY_PATH="${HOME}/.ssh/brane_tunnel_key"

if [[ -f "${KEY_PATH}" ]]; then
    echo "✅ SSH key already exists: ${KEY_PATH}"
else
    echo "🔑 Generating SSH key pair for the tunnel..."
    ssh-keygen -t ed25519 -f "${KEY_PATH}" -N "" \
        -C "brane-tunnel-$(hostname)-$(date +%Y%m%d)"
    echo "✅ SSH key generated: ${KEY_PATH}"
fi

echo ""
echo "Public key (add this to ~/.ssh/authorized_keys on Snellius):"
echo "────────────────────────────────────────────────────────────"
# Restrict the key: no PTY, no agent forwarding, only allow the port-forward
echo -n 'no-pty,no-agent-forwarding,no-X11-forwarding,permitopen="localhost:9753" '
cat "${KEY_PATH}.pub"
echo "────────────────────────────────────────────────────────────"
echo ""

# ── Step 2: Bearer token ──────────────────────────────────────────────────
TOKEN_VAR="BRANE_EXECUTOR_TOKEN"

# Check if token already in .env
if [[ -f "${ENV_FILE}" ]] && grep -q "^${TOKEN_VAR}=" "${ENV_FILE}"; then
    EXISTING_TOKEN=$(grep "^${TOKEN_VAR}=" "${ENV_FILE}" | cut -d= -f2-)
    echo "✅ ${TOKEN_VAR} already set in .env"
else
    # Generate a 32-byte (64 hex char) random token
    if command -v openssl &>/dev/null; then
        NEW_TOKEN=$(openssl rand -hex 32)
    else
        NEW_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    fi
    echo "🔐 Generated Bearer token: ${NEW_TOKEN}"
    EXISTING_TOKEN="${NEW_TOKEN}"

    # Append to .env (create if missing)
    {
        echo ""
        echo "# Brane remote execution"
        echo "${TOKEN_VAR}=${NEW_TOKEN}"
        echo "SNELLIUS_USER=${USER}"
        echo "SNELLIUS_HOST=snellius.surf.nl"
        echo "TUNNEL_PORT=9753"
        echo "TUNNEL_SSH_KEY=${KEY_PATH}"
        echo "BRANE_INSTANCE=local-instance"
        echo "BRANE_EXECUTOR_TIMEOUT=300"
    } >> "${ENV_FILE}"
    echo "✅ Values written to ${ENV_FILE}"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Next steps"
echo "═══════════════════════════════════════════════════════"
echo ""
echo " 1. Copy the public key above and add it to Snellius:"
echo ""
echo "       ssh ${USER}@snellius.surf.nl"
echo "       nano ~/.ssh/authorized_keys   # paste the key line (with restrictions)"
echo ""
echo " 2. On Snellius, set the shared token in your environment:"
echo ""
echo "       echo 'export BRANE_EXECUTOR_TOKEN=${EXISTING_TOKEN}' >> ~/.bashrc"
echo "       source ~/.bashrc"
echo ""
echo " 3. On this machine, start the executor (keep it running in a terminal / tmux):"
echo ""
echo "       source .env && python scripts/remote_execution/local/brane_executor.py"
echo ""
echo " 4. On this machine, start the tunnel (keep it running in another terminal / tmux):"
echo ""
echo "       source .env && bash scripts/remote_execution/local/start_tunnel.sh"
echo ""
echo " 5. On Snellius, test the connection:"
echo ""
echo "       curl -s -H 'Authorization: Bearer ${EXISTING_TOKEN}' \\"
echo "            http://localhost:9753/health"
echo "       # Expected: {\"status\": \"ok\", ...}"
echo ""
echo " 6. Submit a SLURM job with the --execute flag:"
echo ""
echo "       sbatch sbatch.sh"
echo ""
echo "═══════════════════════════════════════════════════════"
