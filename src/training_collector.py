"""
training_collector.py

Collects (intent, generated_code, verdict) tuples from every pipeline run and
appends them as JSONL records to a log file on the Snellius filesystem.

The log accumulates both positive examples (pass) and negative examples (fail)
so the training loop has access to:
  - Correct BraneScript for a given intent           → reward signal
  - Incorrect code with a labelled error type        → penalty signal / DPO pairs

Error type taxonomy (ordered by detection stage)
-------------------------------------------------
  non_code        Model produced no recognisable code (blank, prose, etc.)
  python_code     Model generated Python instead of BraneScript
  json_string     Model used escaped JSON strings (\"key\": \"val\") instead of classes
  syntax          Local heuristic syntax check failed (unbalanced braces, wrong :=, ...)
  semantic        Package/function name not found in retrieved context
  compilation     brane CLI rejected the script (exit_code=2)
  runtime         brane ran but a task container failed (exit_code=1)
  timeout         Execution exceeded the timeout
  pass            All checks passed and execution succeeded

Usage
-----
    from training_collector import TrainingCollector
    collector = TrainingCollector()          # uses TRAINING_DATA_DIR env var
    collector.log_attempt(
        intent="analyze heart-disease risk",
        generated_code="import healthcare; ...",
        verdict="fail",
        error_type="compilation",
        error_message="Unexpected token ';' at line 3",
        stdout="",
        stderr="error: unexpected token",
        exit_code=2,
        attempt_number=1,
        model="Qwen/Qwen3-27B",
    )
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRAINING_DIR = os.path.expanduser(
    os.environ.get("TRAINING_DATA_DIR", "~/brane_training_data")
)

VALID_VERDICTS = {"pass", "fail"}

VALID_ERROR_TYPES = {
    None,           # verdict=pass, no error
    "non_code",     # model output is not code at all
    "python_code",  # model generated Python
    "json_string",  # escaped-JSON-string antipattern
    "syntax",       # local heuristic check failed
    "semantic",     # unknown package/function
    "compilation",  # brane exit_code=2
    "runtime",      # brane exit_code=1
    "timeout",      # execution timed out
}


# ---------------------------------------------------------------------------
# TrainingCollector
# ---------------------------------------------------------------------------

class TrainingCollector:
    """
    Appends JSONL training records to <log_dir>/training_log.jsonl.

    Thread-safe for single-process use (file opened in append mode per write).
    For parallel workers, use separate log files and merge offline.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir or DEFAULT_TRAINING_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.jsonl"

    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------

    def log_attempt(
        self,
        *,
        intent: str,
        generated_code: str,
        verdict: str,
        error_type: Optional[str],
        error_message: str = "",
        stdout: str = "",
        stderr: str = "",
        exit_code: Optional[int] = None,
        attempt_number: int = 1,
        model: str = "",
    ) -> str:
        """
        Append one training record and return its UUID.

        Parameters
        ----------
        intent          : the original user query / natural-language intent
        generated_code  : the BraneScript text produced by the model
                          (may be empty / invalid for non_code errors)
        verdict         : "pass" or "fail"
        error_type      : one of VALID_ERROR_TYPES (None when verdict=pass)
        error_message   : human-readable error detail
        stdout          : captured brane workflow stdout (empty if not executed)
        stderr          : captured brane workflow stderr (empty if not executed)
        exit_code       : brane CLI exit code (None if not executed)
        attempt_number  : which retry attempt this was (1-indexed)
        model           : model identifier string (e.g. "Qwen/Qwen3-27B")
        """
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got {verdict!r}")
        if error_type not in VALID_ERROR_TYPES:
            raise ValueError(f"error_type must be one of {VALID_ERROR_TYPES}, got {error_type!r}")

        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent": intent,
            "generated_code": generated_code,
            "verdict": verdict,
            "error_type": error_type,
            "error_message": error_message,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "attempt_number": attempt_number,
            "model": model,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record_id

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def log_pass(
        self,
        *,
        intent: str,
        generated_code: str,
        stdout: str = "",
        attempt_number: int = 1,
        model: str = "",
    ) -> str:
        """Log a fully successful run."""
        return self.log_attempt(
            intent=intent,
            generated_code=generated_code,
            verdict="pass",
            error_type=None,
            stdout=stdout,
            attempt_number=attempt_number,
            model=model,
        )

    def log_fail(
        self,
        *,
        intent: str,
        generated_code: str,
        error_type: str,
        error_message: str = "",
        stdout: str = "",
        stderr: str = "",
        exit_code: Optional[int] = None,
        attempt_number: int = 1,
        model: str = "",
    ) -> str:
        """Log a failed attempt."""
        return self.log_attempt(
            intent=intent,
            generated_code=generated_code,
            verdict="fail",
            error_type=error_type,
            error_message=error_message,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            attempt_number=attempt_number,
            model=model,
        )

    # ------------------------------------------------------------------
    # Stats helper
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary dict of the current log file."""
        if not self.log_file.exists():
            return {"total": 0}

        total = passes = fails = 0
        error_counts: dict[str, int] = {}

        with open(self.log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if rec.get("verdict") == "pass":
                    passes += 1
                else:
                    fails += 1
                    et = rec.get("error_type") or "unknown"
                    error_counts[et] = error_counts.get(et, 0) + 1

        return {
            "total": total,
            "passes": passes,
            "fails": fails,
            "pass_rate": round(passes / total, 3) if total else 0.0,
            "error_breakdown": error_counts,
            "log_file": str(self.log_file),
        }
