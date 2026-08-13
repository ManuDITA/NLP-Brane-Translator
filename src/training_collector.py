"""
training_collector.py

Collects every pipeline run as a human-readable directory of files on the
Snellius filesystem.  Each run gets its own folder under
<TRAINING_DATA_DIR>/runs/ named by timestamp + short UUID so it is easy to
browse chronologically.

Directory layout
----------------
~/Thesis/NLP-Brane-Translator/
  data/training/
    index.jsonl                          <- lightweight index (one line per run)
  runs/
    2026-06-24_221905_a3b7c2d1/
      execution_result.json            <- COMPLETE record - every field in one file
      intent.txt                       <- original natural-language request
      generated.bs                     <- BraneScript produced by the model
      verdict.txt                      <- "pass" or "fail"
      error_type.txt                   <- e.g. "compilation"  (absent on pass)
      error_message.txt                <- human-readable detail (absent if empty)
      stdout.txt                       <- brane stdout        (absent if empty)
      stderr.txt                       <- brane stderr        (absent if empty)
      exit_code.txt                    <- numeric exit code   (absent if not run)
      meta.json                        <- id, timestamp, attempt_number, model
      committed/                       <- output from commit_result() in the script
        healthcare_reports3/
          summary.html
          reports/
            PAT001_report.json

execution_result.json is always written and contains every field:
  id, timestamp, model, attempt_number,
  intent, generated_code,
  verdict, error_type, error_message,
  stdout, stderr, exit_code,
  committed_data,
  execution  - raw executor payload when Brane was invoked, else null

index.jsonl has one JSON line per run with lightweight fields (id, timestamp,
verdict, error_type, intent excerpt, run_dir) so you can grep / query without
reading every file.

Error type taxonomy (ordered by detection stage)
-------------------------------------------------
  non_code        Model produced no recognisable code (blank, prose, etc.)
  python_code     Model generated Python instead of BraneScript
  json_string     Model used escaped JSON strings instead of classes
  syntax          Local heuristic syntax check failed
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
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRAINING_DIR = str(
    Path(os.environ["TRAINING_DATA_DIR"])
    if "TRAINING_DATA_DIR" in os.environ
    else Path(__file__).resolve().parent.parent / "data" / "training"
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
    Writes one directory per run under <log_dir>/runs/ and keeps a lightweight
    index.jsonl for fast querying.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir or DEFAULT_TRAINING_DIR)
        self.runs_dir = self.log_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        # log_file points at the index so external callers that print
        # collector.log_file still show a useful path.
        self.log_file = self.log_dir / "index.jsonl"
        # Separate append-only log for cache hits: these do not go through
        # generation or execution, so they do not fit the runs/ schema above
        # (no generated_code produced by *this* request, no verdict). Kept
        # here rather than folded into index.jsonl so existing readers of
        # index.jsonl (stats(), dashboard) do not need to special-case them.
        self.cache_hits_file = self.log_dir / "cache_hits.jsonl"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_dir_name(self, ts: datetime, short_id: str) -> str:
        return ts.strftime("%Y-%m-%d_%H%M%S") + "_" + short_id

    def _write(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

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
        committed_data: Optional[dict] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        attempt_number: int = 1,
        model: str = "",
        timing: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Write one run directory and append a line to index.jsonl.
        Returns the run UUID.

        Parameters
        ----------
        intent          : the original user query / natural-language intent
        generated_code  : the BraneScript text produced by the model
        verdict         : "pass" or "fail"
        error_type      : one of VALID_ERROR_TYPES (None when verdict=pass)
        error_message   : human-readable error detail
        stdout          : captured brane workflow stdout
        stderr          : captured brane workflow stderr
        exit_code       : brane CLI exit code (None if not executed)
        committed_data  : dict from collect_committed_outputs() - dataset files
        execution_result: full result payload returned by execute_workflow/run_workflow
        attempt_number  : which retry attempt this was (1-indexed)
        model           : model identifier string (e.g. "Qwen/Qwen3-27B")
        """
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got {verdict!r}")
        if error_type not in VALID_ERROR_TYPES:
            raise ValueError(f"error_type must be one of {VALID_ERROR_TYPES}, got {error_type!r}")

        run_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)
        short_id = run_id[:8]
        run_dir = self.runs_dir / self._run_dir_name(ts, short_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── Core readable files ──────────────────────────────────────────
        self._write(run_dir / "intent.txt", intent)
        self._write(run_dir / "generated.bs", generated_code)
        self._write(run_dir / "verdict.txt", verdict)

        if error_type:
            self._write(run_dir / "error_type.txt", error_type)
        if error_message.strip():
            self._write(run_dir / "error_message.txt", error_message)
        if stdout.strip():
            self._write(run_dir / "stdout.txt", stdout)
        if stderr.strip():
            self._write(run_dir / "stderr.txt", stderr)
        if exit_code is not None:
            self._write(run_dir / "exit_code.txt", str(exit_code))

        # ── Meta (id, timestamp, attempt, model) ────────────────────────
        meta = {
            "id": run_id,
            "timestamp": ts.isoformat(),
            "attempt_number": attempt_number,
            "model": model,
        }
        self._write(run_dir / "meta.json",
                    json.dumps(meta, indent=2, ensure_ascii=False))

        # ── Comprehensive record - single source of truth ────────────────
        # Contains every field so callers never need to parse multiple files.
        record = {
            # Identity
            "id":             run_id,
            "timestamp":      ts.isoformat(),
            "model":          model,
            "attempt_number": attempt_number,
            # Input / output
            "intent":         intent,
            "generated_code": generated_code,
            # Verdict
            "verdict":        verdict,
            "error_type":     error_type,
            "error_message":  error_message,
            # Execution output
            "stdout":         stdout,
            "stderr":         stderr,
            "exit_code":      exit_code,
            # Committed datasets (keys -> dataset info dicts)
            "committed_data": committed_data or {},
            # Raw executor payload (present only when Brane was invoked)
            "execution":      execution_result,
            # Wall-clock timings in seconds (None when not measured)
            "timing":         timing,
        }
        self._write(
            run_dir / "execution_result.json",
            json.dumps(record, indent=2, ensure_ascii=False),
        )

        # ── Committed output files ───────────────────────────────────────
        if committed_data:
            committed_dir = run_dir / "committed"
            for dataset_name, dataset_info in committed_data.items():
                files = dataset_info.get("files", {}) if isinstance(dataset_info, dict) else {}
                for rel_path, content in files.items():
                    out_path = committed_dir / dataset_name / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    self._write(out_path, content if isinstance(content, str) else str(content))

        # ── Lightweight index entry ──────────────────────────────────────
        index_entry = {
            "id": run_id,
            "timestamp": ts.isoformat(),
            "verdict": verdict,
            "error_type": error_type,
            "intent": intent[:120],
            "run_dir": str(run_dir),
            "model": model,
            "attempt_number": attempt_number,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

        return run_id

    # ------------------------------------------------------------------
    # Cache hits
    # ------------------------------------------------------------------

    def log_cache_hit(
        self,
        *,
        intent: str,
        matched_intent: str,
        matched_job_id: str,
        branescript: str,
        similarity: float,
        model: str = "",
    ) -> str:
        """
        Record that *intent* was served from the semantic cache instead of
        being generated, so this event has a persisted trace of its own
        (needed to reconstruct the cache-derivation edge in the PROV export,
        Section~4.6/4.7 in the thesis; see prov_export.py).

        Unlike log_attempt(), this does not create a runs/ directory: no
        generation or execution happened for this specific request, only a
        lookup against the entry originally produced by run
        *matched_job_id*.
        """
        hit_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)
        entry = {
            "id": hit_id,
            "timestamp": ts.isoformat(),
            "intent": intent,
            "matched_intent": matched_intent,
            "matched_job_id": matched_job_id,
            "branescript": branescript,
            "similarity": similarity,
            "model": model,
        }
        with open(self.cache_hits_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return hit_id

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def log_pass(
        self,
        *,
        intent: str,
        generated_code: str,
        stdout: str = "",
        committed_data: Optional[dict] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        attempt_number: int = 1,
        model: str = "",
        timing: Optional[Dict[str, float]] = None,
    ) -> str:
        """Log a fully successful run."""
        return self.log_attempt(
            intent=intent,
            generated_code=generated_code,
            verdict="pass",
            error_type=None,
            stdout=stdout,
            committed_data=committed_data,
            execution_result=execution_result,
            attempt_number=attempt_number,
            model=model,
            timing=timing,
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
        committed_data: Optional[dict] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        attempt_number: int = 1,
        model: str = "",
        timing: Optional[Dict[str, float]] = None,
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
            committed_data=committed_data,
            execution_result=execution_result,
            attempt_number=attempt_number,
            model=model,
            timing=timing,
        )

    # ------------------------------------------------------------------
    # Stats helper
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary dict by reading index.jsonl."""
        if not self.log_file.exists():
            return {"total": 0}

        total = passes = fails = 0
        error_counts = {}

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
            "index_file": str(self.log_file),
            "runs_dir": str(self.runs_dir),
        }
