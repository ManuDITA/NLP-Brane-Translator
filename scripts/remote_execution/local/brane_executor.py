#!/usr/bin/env python3
"""
brane_executor.py — Local Brane workflow execution server.

Receives BraneScript workflow strings from the Snellius pipeline via an SSH
reverse tunnel, runs them with `brane workflow run`, and returns stdout/stderr.

Security model
--------------
- Binds ONLY to 127.0.0.1 (loopback). Never reachable from the network directly.
- All traffic arrives through the SSH reverse tunnel (encrypted end-to-end).
- Every request is authenticated with a shared Bearer token.

Usage
-----
    # Load token from .env, then start:
    source .env
    python scripts/remote_execution/local/brane_executor.py

    # Or with explicit overrides:
    BRANE_EXECUTOR_TOKEN=mytoken python brane_executor.py --port 9753

Required environment variable
------------------------------
    BRANE_EXECUTOR_TOKEN   Shared secret token; must match the one set on Snellius.

Optional environment variables
-------------------------------
    BRANE_EXECUTOR_PORT    Listening port (default: 9753)
    BRANE_EXECUTOR_TIMEOUT Max seconds to wait for a single workflow (default: 300)
    BRANE_INSTANCE         `brane --instance` value (default: local-instance)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# ---------------------------------------------------------------------------
# Configuration (read from env, overridable via CLI args)
# ---------------------------------------------------------------------------
TOKEN = os.environ.get("BRANE_EXECUTOR_TOKEN", "")
DEFAULT_PORT = int(os.environ.get("BRANE_EXECUTOR_PORT", "9753"))
DEFAULT_TIMEOUT = int(os.environ.get("BRANE_EXECUTOR_TIMEOUT", "300"))
BRANE_INSTANCE = os.environ.get("BRANE_INSTANCE", "local-instance")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class ExecutorHandler(BaseHTTPRequestHandler):
    """Handle /health (GET) and /run (POST)."""

    # ── Auth ────────────────────────────────────────────────────────────────
    def _authorized(self) -> bool:
        if not TOKEN:
            _log("WARNING: BRANE_EXECUTOR_TOKEN is not set — accepting all requests!")
            return True
        return self.headers.get("Authorization", "") == f"Bearer {TOKEN}"

    # ── Response helpers ────────────────────────────────────────────────────
    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    # ── Routes ──────────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "timestamp": _ts(),
                "brane_instance": BRANE_INSTANCE,
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            _log(f"REJECTED unauthorised request from {self.client_address[0]}")
            self._send_json(401, {"error": "Unauthorized — wrong or missing Bearer token"})
            return

        if self.path == "/run":
            self._handle_run()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_run(self) -> None:
        body = self._read_body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        workflow: str = payload.get("workflow", "").strip()
        if not workflow:
            self._send_json(400, {"error": "'workflow' field is missing or empty"})
            return

        timeout: int = int(payload.get("timeout", DEFAULT_TIMEOUT))
        query: str = payload.get("query", "")  # optional: for logging

        query_preview = repr(query[:80])
        _log(f"RUN request — query: {query_preview} — workflow length: {len(workflow)} chars")

        # Write workflow to a temp file; brane needs a real path
        suffix = ".bs"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(workflow)
            tmp_path = tmp.name

        try:
            cmd = [
                "brane",
                "--instance", BRANE_INSTANCE,
                "workflow", "run", "a", tmp_path,
            ]
            _log(f"Executing: {' '.join(cmd)}")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )

            success = proc.returncode == 0
            _log(f"Finished — exit_code={proc.returncode}  success={success}")

            self._send_json(200, {
                "success": success,
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "workflow": workflow,
            })

        except subprocess.TimeoutExpired:
            _log(f"TIMEOUT after {timeout}s")
            self._send_json(200, {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Workflow execution timed out after {timeout} seconds.",
                "workflow": workflow,
            })
        except FileNotFoundError:
            _log("ERROR: 'brane' command not found in PATH")
            self._send_json(500, {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "'brane' not found in PATH. Is Brane installed and on your PATH?",
                "workflow": workflow,
            })
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── Suppress default access log (we do our own) ────────────────────────
    def log_message(self, fmt: str, *args) -> None:
        pass


# ---------------------------------------------------------------------------
# Threaded HTTP server (handle multiple requests without blocking)
# ---------------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brane local workflow executor — receives BraneScript via SSH tunnel and runs it."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Workflow execution timeout in seconds (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()

    if not TOKEN:
        print("⚠️  WARNING: BRANE_EXECUTOR_TOKEN is not set.", file=sys.stderr)
        print("   Set it in .env and source it before starting, or export it directly.", file=sys.stderr)
        print("   Example: export BRANE_EXECUTOR_TOKEN=$(openssl rand -hex 32)", file=sys.stderr)
        print("   Continuing without authentication — do not expose the port!", file=sys.stderr)

    # Bind to loopback only — the SSH tunnel delivers traffic here
    bind_addr = ("127.0.0.1", args.port)
    server = ThreadedHTTPServer(bind_addr, ExecutorHandler)

    # Handle Ctrl-C / SIGTERM gracefully
    def _shutdown(signum, frame):
        _log("Shutting down executor...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _log(f"Brane executor listening on {bind_addr[0]}:{bind_addr[1]}")
    _log(f"Brane instance: {BRANE_INSTANCE}")
    _log(f"Auth: {'enabled (Bearer token set)' if TOKEN else 'DISABLED — no token'}")
    _log(f"Workflow timeout: {args.timeout}s")
    _log("Waiting for connections through SSH tunnel...")

    server.serve_forever()


if __name__ == "__main__":
    main()
