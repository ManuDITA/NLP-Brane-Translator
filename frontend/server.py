#!/usr/bin/env python3
"""
frontend/server.py — Brane Pipeline Dashboard backend.

Serves the dashboard HTML and exposes a REST API over the
outputs/pipeline/ directory populated by job_watcher.py.

Usage:
    source .env
    python frontend/server.py

Env vars:
    DASHBOARD_DATA_DIR   Path to pipeline outputs dir
                         (default: <project_root>/outputs/pipeline)
    DASHBOARD_PORT       Port to listen on (default: 5001)
    DASHBOARD_HOST       Host to bind to   (default: 127.0.0.1)
"""

import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from flask import Flask, jsonify, send_from_directory, request

# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

class _PermissiveLoader(yaml.SafeLoader):
    """SafeLoader extended to silently accept unknown tags (e.g. !file)."""

def _ignore_unknown_tag(loader, tag_suffix, node):
    """Fallback constructor: parse the node value without the unknown tag."""
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)

_PermissiveLoader.add_multi_constructor("", _ignore_unknown_tag)


def _yaml_load(text: str) -> dict:
    """Load YAML tolerating custom tags like !file."""
    return yaml.load(text, Loader=_PermissiveLoader)  # noqa: S506

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

DASHBOARD_DATA_DIR = Path(
    os.environ.get("DASHBOARD_DATA_DIR", str(_PROJECT_ROOT / "outputs" / "pipeline"))
)
RESULTS_DIR          = DASHBOARD_DATA_DIR / "results"
PACKAGES_DIR         = _PROJECT_ROOT / "packages"
DATASETS_DIR         = _PROJECT_ROOT / "datasets"
EXEC_RESULTS_FILE    = _PROJECT_ROOT / "data" / "training" / "execution_results.jsonl"
EVAL_DIR             = _PROJECT_ROOT / "outputs" / "eval"

PORT = int(os.environ.get("DASHBOARD_PORT", "5001"))
HOST = os.environ.get("DASHBOARD_HOST", "127.0.0.1")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(_HERE), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(str(_HERE), "index.html")


# ---------------------------------------------------------------------------
# Semantic cache (lazy init — avoids loading ChromaDB on startup if unused)
# ---------------------------------------------------------------------------
_semantic_cache = None

def _get_cache():
    global _semantic_cache
    if _semantic_cache is None:
        try:
            import sys as _sys
            if str(_PROJECT_ROOT / "src") not in _sys.path:
                _sys.path.insert(0, str(_PROJECT_ROOT / "src"))
            from semantic_cache import SemanticCache
            _semantic_cache = SemanticCache()
        except Exception as e:
            print(f"⚠️  Semantic cache unavailable: {e}")
            _semantic_cache = False   # sentinel: don't retry
    return _semantic_cache if _semantic_cache is not False else None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _load_results() -> list[dict]:
    """Read all result JSON files, sorted newest-first."""
    if not RESULTS_DIR.exists():
        return []
    results = []
    for f in RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append(data)
        except Exception:
            pass
    # Sort newest first by submitted_at, fall back to file mtime
    def _sort_key(r):
        ts = r.get("submitted_at") or r.get("executed_at") or ""
        return ts
    results.sort(key=_sort_key, reverse=True)
    return results


@app.route("/api/results")
def api_results():
    """
    GET /api/results
    Returns all results as a JSON array, newest first.
    Query params:
        verdict=pass|fail   filter by verdict
        search=<text>       substring match on intent (case-insensitive)
        since=<iso8601>     only results submitted after this timestamp
        limit=<int>         max results to return (default 500)
    """
    results = _load_results()

    verdict_filter = request.args.get("verdict", "").lower()
    search = request.args.get("search", "").lower()
    since = request.args.get("since", "")
    limit = int(request.args.get("limit", "500"))

    if verdict_filter in ("pass", "fail"):
        results = [r for r in results if r.get("verdict") == verdict_filter]
    if search:
        results = [r for r in results if search in (r.get("intent") or "").lower()]
    if since:
        results = [r for r in results if (r.get("submitted_at") or "") > since]

    return jsonify(results[:limit])


@app.route("/api/results/<job_id>")
def api_result_detail(job_id):
    """GET /api/results/<job_id> — full record including generated_code."""
    path = RESULTS_DIR / f"{job_id}.json"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.route("/api/stats")
def api_stats():
    """GET /api/stats — aggregate counts and averages."""
    results = _load_results()
    total = len(results)
    passed = sum(1 for r in results if r.get("verdict") == "pass")
    failed = total - passed

    durations = [
        r["timing"]["total_s"]
        for r in results
        if r.get("timing") and r["timing"].get("total_s") is not None
    ]
    avg_duration = round(sum(durations) / len(durations), 1) if durations else None

    models = {}
    for r in results:
        m = r.get("model") or "unknown"
        short = m.split("/")[-1] if "/" in m else m
        models[short] = models.get(short, 0) + 1

    last_run = results[0].get("submitted_at") if results else None

    return jsonify({
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "avg_duration_s": avg_duration,
        "models": models,
        "last_run": last_run,
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok",
                    "results_dir": str(RESULTS_DIR),
                    "results_dir_exists": RESULTS_DIR.exists()})


# ---------------------------------------------------------------------------
# Packages API
# ---------------------------------------------------------------------------

def _parse_container_yml(path: Path) -> dict:
    """Parse a container.yml and return a structured dict."""
    try:
        raw = _yaml_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}

    actions = []
    for fn_name, fn_data in (raw.get("actions") or {}).items():
        inputs  = [{"name": p["name"], "type": p["type"]}
                   for p in (fn_data.get("input")  or [])]
        outputs = [{"name": p["name"], "type": p["type"]}
                   for p in (fn_data.get("output") or [])]
        actions.append({
            "name":        fn_name,
            "description": (fn_data.get("description") or "").strip(),
            "inputs":      inputs,
            "outputs":     outputs,
        })

    types = []
    for type_name, type_data in (raw.get("types") or {}).items():
        props = [{"name": p["name"], "type": p["type"]}
                 for p in (type_data.get("properties") or [])]
        types.append({"name": type_name, "properties": props})

    readme_path = path.parent / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    return {
        "name":        raw.get("name", path.parent.name),
        "version":     raw.get("version", ""),
        "kind":        raw.get("kind", ""),
        "actions":     actions,
        "types":       types,
        "readme":      readme,
        "source_file": str(path.relative_to(_PROJECT_ROOT)),
    }


@app.route("/api/packages")
def api_packages():
    """GET /api/packages — list all packages with their functions and types."""
    if not PACKAGES_DIR.exists():
        return jsonify([])
    result = []
    for pkg_dir in sorted(PACKAGES_DIR.iterdir()):
        if not pkg_dir.is_dir():
            continue
        yml_path = pkg_dir / "container.yml"
        if not yml_path.exists():
            continue
        result.append(_parse_container_yml(yml_path))
    return jsonify(result)


@app.route("/api/packages/<name>")
def api_package_detail(name):
    """GET /api/packages/<name> — single package detail."""
    yml_path = PACKAGES_DIR / name / "container.yml"
    if not yml_path.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(_parse_container_yml(yml_path))


# ---------------------------------------------------------------------------
# Datasets API
# ---------------------------------------------------------------------------

def _parse_data_yml(path: Path) -> dict:
    """Parse a data.yml and return a structured dict."""
    try:
        raw = _yaml_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}

    dataset_dir = path.parent
    readme_path = dataset_dir / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    # Collect data files (non-yml, non-readme, non-workflow)
    files = []
    for f in sorted(dataset_dir.rglob("*")):
        if f.is_file() and f.suffix not in (".yml", ".yaml") \
                and f.name.lower() != "readme.md":
            rel = str(f.relative_to(dataset_dir))
            size = f.stat().st_size
            files.append({"path": rel, "size_bytes": size})

    # Collect example workflows if present
    workflows = []
    wf_dir = dataset_dir / "workflows"
    if wf_dir.exists():
        for wf in sorted(wf_dir.glob("*.bs")):
            workflows.append({
                "name":    wf.name,
                "content": wf.read_text(encoding="utf-8"),
            })

    # access field can be a complex YAML object
    access = raw.get("access", {})
    if isinstance(access, dict):
        access_str = next(iter(access)) + ": " + str(next(iter(access.values())))
    else:
        access_str = str(access)

    return {
        "name":      raw.get("name", dataset_dir.name),
        "access":    access_str,
        "readme":    readme,
        "files":     files,
        "workflows": workflows,
        "source_file": str(path.relative_to(_PROJECT_ROOT)),
    }


@app.route("/api/datasets")
def api_datasets():
    """GET /api/datasets — list all datasets with metadata."""
    if not DATASETS_DIR.exists():
        return jsonify([])
    result = []
    for ds_dir in sorted(DATASETS_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        yml_path = ds_dir / "data.yml"
        if not yml_path.exists():
            continue
        result.append(_parse_data_yml(yml_path))
    return jsonify(result)


@app.route("/api/datasets/<name>")
def api_dataset_detail(name):
    """GET /api/datasets/<name> — single dataset detail."""
    yml_path = DATASETS_DIR / name / "data.yml"
    if not yml_path.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(_parse_data_yml(yml_path))


# ---------------------------------------------------------------------------
# Execution Results API
# ---------------------------------------------------------------------------

def _load_exec_results() -> list[dict]:
    """Read data/training/execution_results.jsonl, newest-timestamp first."""
    if not EXEC_RESULTS_FILE.exists():
        return []
    rows = []
    with open(EXEC_RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return rows


@app.route("/api/execution-results")
def api_exec_results():
    """
    GET /api/execution-results
    Query params:
        success=true|false   filter by success status
        search=<text>        substring match on intent (case-insensitive)
        source=<filename>    filter by source_file name
        limit=<int>          max rows to return (default 1000)
    """
    rows = _load_exec_results()

    success_filter = request.args.get("success", "")
    search         = request.args.get("search", "").lower()
    source         = request.args.get("source", "")
    limit          = int(request.args.get("limit", "1000"))

    if success_filter == "true":
        rows = [r for r in rows if r.get("success")]
    elif success_filter == "false":
        rows = [r for r in rows if not r.get("success")]
    if search:
        rows = [r for r in rows if search in (r.get("intent") or "").lower()]
    if source:
        rows = [r for r in rows if r.get("source_file") == source]

    # Strip heavy fields for list view; keep full data for detail
    lite = []
    for r in rows[:limit]:
        lite.append({
            "id":               r.get("id"),
            "source_file":      r.get("source_file"),
            "intent":           r.get("intent"),
            "exit_code":        r.get("exit_code"),
            "success":          r.get("success"),
            "timed_out":        r.get("timed_out"),
            "execution_time_s": r.get("execution_time_s"),
            "timestamp":        r.get("timestamp"),
            "stdout_preview":   (r.get("stdout") or "")[:200],
            "stderr_preview":   (r.get("stderr") or "")[:200],
        })
    return jsonify(lite)


@app.route("/api/execution-results/<path:exec_id>")
def api_exec_result_detail(exec_id):
    """GET /api/execution-results/<id> — full record including code + full output."""
    rows = _load_exec_results()
    for r in rows:
        if r.get("id") == exec_id:
            return jsonify(r)
    return jsonify({"error": "not found"}), 404


@app.route("/api/execution-results/stats")
def api_exec_results_stats():
    """GET /api/execution-results/stats — aggregate counts."""
    rows  = _load_exec_results()
    total = len(rows)
    passed  = sum(1 for r in rows if r.get("success"))
    failed  = total - passed
    timed   = sum(1 for r in rows if r.get("timed_out"))
    times   = [r["execution_time_s"] for r in rows if r.get("execution_time_s") is not None]
    avg_t   = round(sum(times) / len(times), 3) if times else None

    sources: dict[str, dict] = {}
    for r in rows:
        s = r.get("source_file") or "unknown"
        if s not in sources:
            sources[s] = {"total": 0, "passed": 0}
        sources[s]["total"] += 1
        if r.get("success"):
            sources[s]["passed"] += 1

    return jsonify({
        "total":      total,
        "passed":     passed,
        "failed":     failed,
        "timed_out":  timed,
        "pass_rate":  round(passed / total * 100, 1) if total else 0,
        "avg_time_s": avg_t,
        "sources":    sources,
    })


# ---------------------------------------------------------------------------
# Evaluation Results API
# ---------------------------------------------------------------------------

def _load_training_config(model_path: str | None) -> dict | None:
    """
    Given a model_path (e.g. outputs/models/qwen3.5-9b),
    look for training_config.json in the corresponding adapter directory
    (outputs/models/qwen3.5-9b-adapter or qwen3.5-9b-grpo-adapter).
    Returns the config dict or None if not found.
    """
    if not model_path:
        return None
    mp = Path(model_path)
    slug = mp.name  # e.g. qwen3.5-9b or qwen3.5-9b-ep5-r32
    candidates = [
        mp.parent / f"{slug}-adapter",
        mp.parent / f"{slug}-grpo-adapter",
        mp,  # config may also be saved in merged dir
    ]
    for candidate in candidates:
        cfg = candidate / "training_config.json"
        if cfg.exists():
            try:
                return json.loads(cfg.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def _eval_run_summary(path: Path) -> dict | None:
    """Return lightweight metadata for one eval run JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        model_path = data.get("model_path")
        # Prefer training_config embedded in the eval file (newer runs);
        # fall back to loading from disk for older eval files
        training_config = data.get("training_config") or _load_training_config(model_path)
        return {
            "file":                  path.name,
            "slug":                  path.stem,
            "model":                 data.get("model"),
            "model_path":            model_path,
            "total":                 data.get("total"),
            "compile_rate":          data.get("compile_rate"),
            "execution_rate":        data.get("execution_rate"),
            "output_match_rate":     data.get("output_match_rate"),
            "output_match_n":        data.get("output_match_n"),
            "committed_match_rate":  data.get("committed_match_rate"),
            "committed_match_n":     data.get("committed_match_n"),
            "timestamp":             data.get("timestamp"),
            "training_config":       training_config,
        }
    except Exception:
        return None


@app.route("/api/eval/runs")
def api_eval_runs():
    """GET /api/eval/runs — list all evaluation run summaries, newest first."""
    if not EVAL_DIR.exists():
        return jsonify([])
    runs = [s for f in sorted(EVAL_DIR.glob("*.json"))
            if (s := _eval_run_summary(f)) is not None]
    runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return jsonify(runs)


@app.route("/api/eval/runs/<path:slug>")
def api_eval_run_detail(slug):
    """GET /api/eval/runs/<slug> — full eval run with per-example details."""
    path = EVAL_DIR / f"{slug}.json"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.route("/api/eval/ground-truth")
def api_eval_ground_truth():
    """GET /api/eval/ground-truth — all execution_results entries keyed by id."""
    if not EXEC_RESULTS_FILE.exists():
        return jsonify({})
    refs = {}
    for line in EXEC_RESULTS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            eid = r.get("id")
            if eid:
                refs[eid] = {
                    "id":                eid,
                    "intent":            r.get("intent", ""),
                    "branescript":       r.get("branescript", ""),
                    "stdout":            (r.get("stdout") or "").strip(),
                    "stderr":            (r.get("stderr") or "").strip(),
                    "exit_code":         r.get("exit_code"),
                    "success":           r.get("success", False),
                    "committed_results": r.get("committed_results") or {},
                    "source_file":       r.get("source_file", ""),
                }
        except Exception:
            pass
    return jsonify(refs)


# ---------------------------------------------------------------------------
# Generate API — submit intent → Snellius SLURM job → BraneScript → execute
# ---------------------------------------------------------------------------

# In-memory store: req_id → {status, slurm_id, intent, model, submitted_at}
_generate_jobs: dict[str, dict] = {}


def _ssh_args() -> list[str]:
    """Build base SSH args from env vars."""
    user = os.environ.get("SNELLIUS_USER", "")
    host = os.environ.get("SNELLIUS_HOST", "snellius.surf.nl")
    key  = os.environ.get("SNELLIUS_SSH_KEY", "")
    remote = f"{user}@{host}" if user else host
    args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if key:
        args += ["-i", key]
    args.append(remote)
    return args


def _check_slurm_status(slurm_id: str) -> str:
    """Query squeue for SLURM job state. Returns 'PENDING','RUNNING','COMPLETED','FAILED', or ''."""
    try:
        result = subprocess.run(
            _ssh_args() + [f"squeue -j {slurm_id} -h -o '%T' 2>/dev/null"],
            capture_output=True, text=True, timeout=10,
        )
        state = result.stdout.strip()
        return state if state else "COMPLETED"
    except Exception:
        return ""


def _auto_label(model_path: str) -> str:
    """Derive a human-readable label from a model path or HF ID."""
    p = model_path.rstrip("/")
    name = Path(p).name
    # Detect local fine-tuned model: directory under outputs/models/ that isn't a HF-style "org/model"
    if (Path(p).exists() or p.startswith("outputs/models/")) and "/" not in name:
        if name.endswith("-grpo-adapter") or name.endswith("-adapter"):
            return f"{name} (adapter)"
        # It's a merged model dir — determine mode from whether grpo-adapter sibling exists
        models_dir = Path(p).parent
        if (models_dir / f"{name}-grpo-adapter").exists():
            suffix = "GRPO"
        else:
            suffix = "SFT"
        label = name.replace("-", ".").replace("_", ".").title()
        return f"{label} ({suffix})"
    if "/" in p and not p.startswith("/"):
        return f"{p.split('/')[-1]} (base)"
    return f"{name} (local)"


@app.route("/api/models")
def api_models():
    """GET /api/models — list available models (merged SFT/GRPO + env-configured HF IDs)."""
    models = []

    models_dir = _PROJECT_ROOT / "outputs" / "models"
    if models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            # Show merged model dirs (exclude adapter dirs)
            if d.is_dir() and not d.name.endswith("-adapter"):
                models.append({
                    "id":    str(d.relative_to(_PROJECT_ROOT)),
                    "label": _auto_label(str(d.relative_to(_PROJECT_ROOT))),
                    "type":  "local",
                })

    hf_models_str = os.environ.get("FRONTEND_MODELS", "")
    for m in (m.strip() for m in hf_models_str.split(",") if m.strip()):
        models.append({
            "id":    m,
            "label": f"{m.split('/')[-1]} (base)",
            "type":  "hf",
        })

    return jsonify(models)


@app.route("/api/generate", methods=["GET"])
def api_generate_list():
    """GET /api/generate — return only currently active (in-memory) jobs.
    Completed historical jobs are visible in the Runs tab via /api/results.
    """
    items = []
    for req_id, job in _generate_jobs.items():
        items.append({
            "req_id":       req_id,
            "intent":       job["intent"],
            "model":        job["model"],
            "status":       job.get("status", "submitted"),
            "slurm_id":     job.get("slurm_id"),
            "attempt":      job.get("attempt", 1),
            "max_attempts": job.get("max_attempts", MAX_GENERATE_RETRIES),
            "submitted_at": job.get("submitted_at", ""),
            "history":      job.get("history", []),
            "success":      None,
            "script":       "",
            "stdout":       "",
            "stderr":       "",
        })
    # Newest first
    items.sort(key=lambda x: x["submitted_at"], reverse=True)
    return jsonify(items)


MAX_GENERATE_RETRIES = 3   # max LLM attempts per intent (initial + retries)


def _submit_snellius_generate(req_id: str, intent: str, model: str,
                               context_file: str = "") -> tuple[str | None, str | None]:
    """
    SSH to Snellius and submit sbatch_generate.sh.
    context_file: optional remote path to a JSON file with {prev_script, error_feedback, attempt}.
    Returns (slurm_id, error_message).
    """
    user        = os.environ.get("SNELLIUS_USER", "")
    host        = os.environ.get("SNELLIUS_HOST", "snellius.surf.nl")
    project_dir = os.environ.get("SNELLIUS_PROJECT_DIR", "")

    if not user or not host or not project_dir:
        return None, "SNELLIUS_USER, SNELLIUS_HOST, SNELLIUS_PROJECT_DIR must be set in .env"

    intent_esc = intent.replace("'", "'\\''")
    model_esc  = model.replace("'", "'\\''")
    ctx_export = f",CONTEXT_FILE='{context_file}'" if context_file else ""

    cmd = (
        f"cd '{project_dir}' && "
        f"sbatch --export=ALL,"
        f"INTENT='{intent_esc}',"
        f"EVAL_MODEL='{model_esc}',"
        f"REQ_ID='{req_id}'"
        f"{ctx_export} "
        f"sbatch_generate.sh"
    )

    try:
        result = subprocess.run(
            _ssh_args() + [cmd],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        return None, f"SSH failed: {e}"

    if result.returncode != 0:
        return None, result.stderr.strip() or "sbatch failed"

    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    return (m.group(1) if m else None), None


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """POST /api/generate — submit an intent + model to Snellius for generation."""
    body   = request.get_json(silent=True) or {}
    intent = (body.get("intent") or "").strip()
    model  = (body.get("model")  or "").strip()

    if not intent:
        return jsonify({"error": "intent is required"}), 400
    if not model:
        return jsonify({"error": "model is required"}), 400

    # ── Semantic cache lookup ─────────────────────────────────────────────────
    cache = _get_cache()
    if cache:
        try:
            hit = cache.lookup(intent)
            if hit:
                req_id = str(uuid.uuid4())
                print(f"🎯 Cache hit (similarity={hit['similarity']}) for: {intent[:60]}…", flush=True)
                _generate_jobs[req_id] = {
                    "status":       "done",
                    "intent":       intent,
                    "model":        model,
                    "attempt":      1,
                    "max_attempts": MAX_GENERATE_RETRIES,
                    "history":      [],
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "cache_hit":    True,
                    "cache_similarity": hit["similarity"],
                    "original_intent":  hit["intent"],
                }
                return jsonify({
                    "req_id":       req_id,
                    "status":       "cache_hit",
                    "cache_hit":    True,
                    "similarity":   hit["similarity"],
                    "original_intent": hit["intent"],
                    "branescript":  hit["branescript"],
                    "execution":    hit["execution"],
                    "hits":         hit["hits"],
                })
        except Exception as e:
            print(f"⚠️  Cache lookup error: {e}", flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    req_id = str(uuid.uuid4())
    slurm_id, err = _submit_snellius_generate(req_id, intent, model)
    if err:
        return jsonify({"error": err}), 500

    _generate_jobs[req_id] = {
        "status":       "submitted",
        "slurm_id":     slurm_id,
        "intent":       intent,
        "model":        model,
        "attempt":      1,
        "max_attempts": MAX_GENERATE_RETRIES,
        "history":      [],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    return jsonify({"req_id": req_id, "slurm_id": slurm_id, "status": "submitted",
                    "attempt": 1, "max_attempts": MAX_GENERATE_RETRIES})


@app.route("/api/generate/<req_id>")
def api_generate_status(req_id):
    """GET /api/generate/<req_id> — poll for generation + execution result (with auto-retry)."""
    result_path = DASHBOARD_DATA_DIR / "results" / f"{req_id}.json"
    job = _generate_jobs.get(req_id, {})

    if result_path.exists():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            attempt      = job.get("attempt", 1)
            max_attempts = job.get("max_attempts", MAX_GENERATE_RETRIES)
            verdict_pass = data.get("verdict") == "pass"

            # Auto-retry on failure if attempts remain
            if not verdict_pass and attempt < max_attempts and req_id in _generate_jobs:
                prev_script    = data.get("generated_code", "")
                error_feedback = (data.get("stderr") or "").strip()
                error_type     = data.get("error_type", "")

                # Record this failed attempt in history
                job["history"] = job.get("history", [])
                job["history"].append({
                    "attempt":    attempt,
                    "script":     prev_script,
                    "error":      error_feedback,
                    "error_type": error_type,
                })

                # Write retry context file to Snellius so generate_single.py can read it
                jobs_dir   = os.environ.get("SNELLIUS_JOBS_DIR", "~/brane_jobs")
                ctx_remote = f"{jobs_dir}/context/{req_id}.json"
                ctx_json   = json.dumps({
                    "prev_script":    prev_script,
                    "error_feedback": error_feedback,
                    "attempt":        attempt + 1,
                }, ensure_ascii=False).replace("'", "'\\''")

                try:
                    subprocess.run(
                        _ssh_args() + [
                            f"mkdir -p {jobs_dir}/context && "
                            f"printf '%s' '{ctx_json}' > {ctx_remote}"
                        ],
                        capture_output=True, timeout=15,
                    )
                except Exception:
                    pass  # retry submission will still work without context

                # Delete failed result so the next poll won't re-trigger another retry
                result_path.unlink(missing_ok=True)

                # Submit new SLURM job (same req_id, with context)
                slurm_id, err = _submit_snellius_generate(
                    req_id, job["intent"], job["model"], context_file=ctx_remote
                )

                job["attempt"]  = attempt + 1
                job["slurm_id"] = slurm_id
                job["status"]   = "retrying"

                return jsonify({
                    "status":       "retrying",
                    "req_id":       req_id,
                    "attempt":      attempt + 1,
                    "max_attempts": max_attempts,
                    "slurm_id":     slurm_id,
                    "intent":       job["intent"],
                    "model":        job["model"],
                    "history":      job["history"],
                    "prev_error":   error_feedback[:300],
                })

            # Final result (pass or max retries exhausted)
            # ── Store in semantic cache on success ────────────────────────────
            if verdict_pass:
                try:
                    cache = _get_cache()
                    if cache:
                        cache.store(
                            intent      = data.get("intent") or job.get("intent", ""),
                            branescript = data.get("generated_code", ""),
                            execution   = {
                                "stdout":    (data.get("stdout") or "").strip(),
                                "stderr":    (data.get("stderr") or "").strip(),
                                "exit_code": data.get("exit_code"),
                                "success":   True,
                                "committed_data": data.get("committed_data") or {},
                            },
                            job_id = req_id,
                        )
                except Exception as _ce:
                    print(f"⚠️  Cache store error: {_ce}", flush=True)
            # ─────────────────────────────────────────────────────────────────
            return jsonify({
                "status":         "done",
                "req_id":         req_id,
                "attempt":        attempt,
                "max_attempts":   max_attempts,
                "intent":         data.get("intent") or job.get("intent", ""),
                "model":          data.get("model")  or job.get("model", ""),
                "script":         data.get("generated_code", ""),
                "success":        verdict_pass,
                "stdout":         (data.get("stdout") or "").strip(),
                "stderr":         (data.get("stderr") or "").strip(),
                "exit_code":      data.get("exit_code"),
                "error_type":     data.get("error_type"),
                "committed_data": data.get("committed_data") or {},
                "history":        job.get("history", []),
            })
        except Exception as e:
            return jsonify({"status": "error", "error": str(e), "req_id": req_id})

    job = _generate_jobs.get(req_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    # Query SLURM for a more precise status while we wait for the result
    slurm_id = job.get("slurm_id")
    if slurm_id:
        slurm_state = _check_slurm_status(slurm_id)
        if slurm_state == "PENDING":
            job["status"] = "queued"
        elif slurm_state == "RUNNING":
            job["status"] = "generating"
        elif slurm_state in ("COMPLETED", ""):
            job["status"] = "executing"   # job done on Snellius; job_watcher picking up
        elif slurm_state in ("FAILED", "CANCELLED", "TIMEOUT"):
            job["status"] = "slurm_failed"

    return jsonify({
        "status":   job.get("status", "submitted"),
        "req_id":   req_id,
        "slurm_id": slurm_id,
        "intent":   job["intent"],
        "model":    job["model"],
    })


@app.route("/api/cache/stats")
def api_cache_stats():
    """GET /api/cache/stats — semantic cache statistics."""
    cache = _get_cache()
    if not cache:
        return jsonify({"available": False})
    try:
        stats = cache.stats()
        return jsonify({"available": True, **stats})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/cache/entries")
def api_cache_entries():
    """GET /api/cache/entries — list cached intents (newest first, limit 100)."""
    cache = _get_cache()
    if not cache:
        return jsonify([])
    try:
        return jsonify(cache.list_entries(limit=100))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/invalidate", methods=["POST"])
def api_cache_invalidate():
    """POST /api/cache/invalidate — remove entries similar to a given intent."""
    body   = request.get_json(silent=True) or {}
    intent = (body.get("intent") or "").strip()
    if not intent:
        return jsonify({"error": "intent is required"}), 400
    cache = _get_cache()
    if not cache:
        return jsonify({"error": "cache unavailable"}), 503
    removed = cache.invalidate(intent)
    return jsonify({"removed": removed})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"📊 Brane Dashboard — http://{HOST}:{PORT}")
    print(f"   Reading results from: {RESULTS_DIR}")
    if not RESULTS_DIR.exists():
        print(f"   ⚠️  results dir does not exist yet — it will be created by job_watcher.py")
    app.run(host=HOST, port=PORT, debug=False)
