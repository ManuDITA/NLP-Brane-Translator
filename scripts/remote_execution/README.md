# Remote Execution System

Runs LLM inference on **Snellius** (GPU cluster) and executes the generated BraneScript on your **local machine** (where Docker and Brane are available).

## Why this architecture

| Constraint | Solution |
|---|---|
| Docker not allowed on Snellius | Execute `brane workflow run` only on the local machine |
| LLM needs GPU (H100) | Run inference on Snellius via SLURM |
| Local machine is behind NAT | Reverse SSH tunnel: local machine connects outbound to Snellius |
| Secure personal machine | Traffic encrypted by SSH; executor binds to loopback only; Bearer token auth |

## Traffic flow

```
[Local machine]
  brane_executor.py             ← runs brane workflow run
  listens on 127.0.0.1:9753     ← only reachable via the tunnel

       ↑  SSH reverse tunnel (local initiates)
       |  ssh -NR 9753:localhost:9753  user@snellius.surf.nl

[Snellius login node]
  localhost:9753                ← reverse tunnel endpoint

       ↑  SSH local-forward (from inside SLURM job)
       |  ssh -NL 9753:localhost:9753  login_node

[Snellius compute node]
  localhost:9753  →  POST /run  {workflow: "..."}
  pipeline.py   (LLM inference + HTTP call)
```

---

## One-time setup

### 1. Local machine

```bash
# From the project root
bash scripts/remote_execution/local/setup_local.sh
```

This generates:
- `~/.ssh/brane_tunnel_key` — dedicated Ed25519 SSH key for the tunnel
- Appends to `.env` — BRANE_EXECUTOR_TOKEN, SNELLIUS_USER, and other settings

It prints the **restricted public key** to copy to Snellius.

### 2. Snellius — add the public key

```bash
ssh your_username@snellius.surf.nl
nano ~/.ssh/authorized_keys
```

Paste the key line printed by `setup_local.sh`. It looks like:

```
no-pty,no-agent-forwarding,no-X11-forwarding,permitopen="localhost:9753" ssh-ed25519 AAAA... brane-tunnel-...
```

The restrictions prevent the key from being used for anything other than the port forward, even if it is ever compromised.

### 3. Snellius — set the Bearer token

```bash
# On Snellius, add to ~/.bashrc:
echo 'export BRANE_EXECUTOR_TOKEN=<the_token_from_setup_output>' >> ~/.bashrc
source ~/.bashrc
```

The token must match the one in your local `.env`.

---

## Running the system

### On your local machine (two persistent terminal sessions / tmux panes)

**Terminal 1 — executor server:**
```bash
source .env
python scripts/remote_execution/local/brane_executor.py
```

**Terminal 2 — SSH tunnel:**
```bash
source .env
bash scripts/remote_execution/local/start_tunnel.sh
```

Both must be running whenever you submit jobs to Snellius.

> **Tip**: Use `tmux new-session -s brane` and split the window (`Ctrl-B %`) so both run side by side.

### On Snellius — submit a job

```bash
sbatch sbatch.sh
```

The `sbatch.sh` script:
1. Sources `setup_compute_tunnel.sh` to forward the port from the compute node through the login node
2. Calls `pipeline.py` with LLM inference
3. If `--execute` is enabled and the tunnel is up, POSTs the BraneScript to the local executor
4. Prints the Brane execution output to the SLURM log

---

## Verifying the connection

From an **interactive session on Snellius** (login node):
```bash
curl -s \
  -H "Authorization: Bearer ${BRANE_EXECUTOR_TOKEN}" \
  http://localhost:9753/health
# Expected: {"status": "ok", "timestamp": "...", "brane_instance": "local-instance"}
```

From a **compute node** (after sourcing setup_compute_tunnel.sh inside a job):
```bash
curl -s \
  -H "Authorization: Bearer ${BRANE_EXECUTOR_TOKEN}" \
  http://localhost:9753/health
```

---

## Configuration reference

All settings can be placed in the project `.env` file or exported as environment variables.

| Variable | Default | Description |
|---|---|---|
| `BRANE_EXECUTOR_TOKEN` | *(required)* | Shared Bearer token for HTTP auth |
| `SNELLIUS_USER` | `$USER` | Your Snellius username |
| `SNELLIUS_HOST` | `snellius.surf.nl` | Snellius login node hostname |
| `TUNNEL_PORT` | `9753` | Port for the reverse tunnel and executor |
| `TUNNEL_SSH_KEY` | `~/.ssh/brane_tunnel_key` | Dedicated SSH key path |
| `BRANE_INSTANCE` | `local-instance` | Brane instance name on local machine |
| `BRANE_EXECUTOR_TIMEOUT` | `300` | Max seconds per workflow execution |
| `BRANE_EXECUTOR_URL` | *(set by setup_compute_tunnel.sh)* | Overrides the executor URL if set |

---

## Security notes

| Measure | Purpose |
|---|---|
| SSH reverse tunnel | All traffic is encrypted (AES-256-CTR or ChaCha20 via OpenSSH) |
| Executor binds to `127.0.0.1` | Never directly reachable from the network |
| Bearer token | Authenticates every HTTP request; guards against other users on the cluster accessing the tunnel port |
| Restricted authorized_keys | `no-pty,no-agent-forwarding,permitopen="localhost:9753"` limits what the tunnel key can do |
| Dedicated key pair | Compromise of the tunnel key cannot be used to log in or do anything other than the port forward |

---

## File reference

```
scripts/remote_execution/
  local/
    brane_executor.py           HTTP server — receives workflows, runs brane
    start_tunnel.sh             Maintains reverse SSH tunnel (local → Snellius)
    setup_local.sh              One-time setup: SSH key + token generation
  snellius/
    setup_compute_tunnel.sh     Sourced by sbatch.sh; forwards port from compute → login node
  shared/
    config.example.env          Configuration template
  README.md                     This file
```

Modified project files:
```
src/pipeline.py                 execute_workflow() now POSTs to the local executor
sbatch.sh                       Sources setup_compute_tunnel.sh; passes --execute flag
```

---

## Troubleshooting

**`[tunnel] ❌ Cannot SSH to login node`**  
The compute node cannot reach the login node via SSH. On Snellius this usually works by default. Check with the SURF support team if it doesn't.

**`[tunnel] ⚠️ Executor NOT reachable after 30s`**  
Either `brane_executor.py` is not running on your local machine, or `start_tunnel.sh` is not running (or has disconnected). Check both terminal sessions.

**`401 Unauthorized`**  
The `BRANE_EXECUTOR_TOKEN` on Snellius does not match the one in your local `.env`. Re-check both values.

**`'brane' not found in PATH`**  
The executor runs as your local user but cannot find the `brane` binary. Make sure the `brane` CLI is on your PATH in the shell where you start `brane_executor.py`.

**Tunnel disconnects frequently**  
Install `autossh` (`sudo apt install autossh`) — `start_tunnel.sh` uses it automatically when available for much more reliable reconnection.
