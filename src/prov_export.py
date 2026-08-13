"""
prov_export.py

Serialises the per-request provenance records already written by
TrainingCollector (src/training_collector.py, command-line entry point) and
save_to_dashboard (scripts/remote_execution/local/job_watcher.py, dashboard
entry point) as a W3C PROV-DM document, instead of the flat, per-request JSON
they are stored as today.

This does not add anything to PROV-DM itself: it defines a small vocabulary
of PROV-DM specialisations for this system (an Intent entity, a Translation
activity, a BraneScriptWorkflow entity, an Execution activity, an
ExecutionResult entity, and a ModelAgent) the same way CWLProv and ProvONE
specialise PROV-DM's entities and activities for their own domains (see
Section 2.4 and Section 4.6 of the thesis).

Two kinds of source record are handled.

Generation record (a TrainingCollector run directory's execution_result.json,
or a dashboard results/<id>.json file) -- records that an intent was
translated by a specific model into a BraneScript workflow and, if executed,
what the outcome was:

    Intent --used--> Translation --wasGeneratedBy--> Workflow
    Translation --wasAssociatedWith--> ModelAgent
    Workflow --wasAttributedTo--> ModelAgent
    Workflow --used--> Execution --wasGeneratedBy--> ExecutionResult
    Execution --wasInformedBy--> Translation

Cache-hit record (TrainingCollector.log_cache_hit, cache_hits.jsonl) --
records that a repeated or paraphrased intent was served from the semantic
cache instead of triggering a new translation:

    NewIntent --alternateOf--> MatchedIntent
    NewIntent --used--> CacheLookup --wasGeneratedBy--> ServedWorkflow
    ServedWorkflow --wasDerivedFrom--> Workflow (of matched_job_id)

Usage
-----
    # One run directory
    python -m src.prov_export --run-dir data/training/runs/<run> -o run.json

    # Every run + every cache hit under TRAINING_DATA_DIR, merged into one
    # document
    python -m src.prov_export --all -o combined.json
    python -m src.prov_export --all -o combined.provn --format provn
    python -m src.prov_export --all -o combined.dot   --format dot
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from prov.model import ProvDocument

NAMESPACE_URI = "https://github.com/manudita-nov/NLP-Brane-Translator#"
NS = "bt"

DEFAULT_TRAINING_DIR = Path(
    os.environ.get("TRAINING_DATA_DIR")
    or (Path(__file__).resolve().parent.parent / "data" / "training")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_document() -> ProvDocument:
    doc = ProvDocument()
    doc.add_namespace(NS, NAMESPACE_URI)
    return doc


def _qname(kind: str, ident: str) -> str:
    return f"{NS}:{kind}-{ident}"


def _model_agent_id(model: str) -> str:
    slug = (model or "unknown-model").strip().replace("/", "_").replace(" ", "_")
    return _qname("agent", slug)


# ---------------------------------------------------------------------------
# Generation records (TrainingCollector runs/, dashboard results/)
# ---------------------------------------------------------------------------

def generation_record_to_document(record: dict) -> ProvDocument:
    """Build a PROV document for one generation record (pass or fail)."""
    doc = _new_document()

    run_id = str(record["id"])
    intent_text = record.get("intent", "")
    model = record.get("model", "")
    timestamp = (
        record.get("timestamp")
        or record.get("executed_at")
        or record.get("submitted_at")
    )
    verdict = record.get("verdict")
    error_type = record.get("error_type")
    exit_code = record.get("exit_code")
    attempt_number = record.get("attempt_number", 1)
    # An execution stage exists whenever Brane was actually invoked, which is
    # whenever an exit code was captured, regardless of whether it succeeded.
    executed = exit_code is not None

    intent_id = _qname("intent", run_id)
    translation_id = _qname("translation", run_id)
    workflow_id = _qname("workflow", run_id)
    agent_id = _model_agent_id(model)

    intent_e = doc.entity(
        intent_id,
        {"prov:type": f"{NS}:Intent", f"{NS}:text": intent_text},
    )
    agent = doc.agent(
        agent_id,
        {"prov:type": "prov:SoftwareAgent", f"{NS}:model": model},
    )
    translation = doc.activity(
        translation_id,
        startTime=timestamp,
        other_attributes={
            f"{NS}:type": f"{NS}:TranslationActivity",
            f"{NS}:attemptNumber": attempt_number,
        },
    )
    workflow_e = doc.entity(
        workflow_id,
        {"prov:type": f"{NS}:BraneScriptWorkflow"},
    )

    doc.used(translation, intent_e)
    doc.wasGeneratedBy(workflow_e, translation)
    doc.wasAssociatedWith(translation, agent)
    doc.wasAttributedTo(workflow_e, agent)

    if executed:
        execution_id = _qname("execution", run_id)
        result_id = _qname("result", run_id)
        execution = doc.activity(
            execution_id,
            other_attributes={f"{NS}:type": f"{NS}:ExecutionActivity"},
        )
        doc.used(execution, workflow_e)
        doc.wasInformedBy(execution, translation)
        result_e = doc.entity(
            result_id,
            {
                "prov:type": f"{NS}:ExecutionResult",
                f"{NS}:exitCode": exit_code,
                f"{NS}:verdict": verdict,
                f"{NS}:errorType": error_type,
            },
        )
        doc.wasGeneratedBy(result_e, execution)

    return doc


# ---------------------------------------------------------------------------
# Cache-hit records (TrainingCollector.log_cache_hit, cache_hits.jsonl)
# ---------------------------------------------------------------------------

def cache_hit_record_to_document(record: dict) -> ProvDocument:
    """Build a PROV document for one cache-hit record."""
    doc = _new_document()

    hit_id = str(record["id"])
    matched_job_id = str(record.get("matched_job_id") or "unknown")
    timestamp = record.get("timestamp")
    model = record.get("model", "")

    new_intent_id = _qname("intent", hit_id)
    matched_intent_id = _qname("intent", matched_job_id)
    lookup_id = _qname("cachelookup", hit_id)
    served_workflow_id = _qname("workflow", hit_id)
    original_workflow_id = _qname("workflow", matched_job_id)

    new_intent_e = doc.entity(
        new_intent_id,
        {"prov:type": f"{NS}:Intent", f"{NS}:text": record.get("intent", "")},
    )
    # The matched entity is declared here only as a reference; when this
    # document is merged with the one produced for matched_job_id (see
    # export_all below), the two declarations resolve to the same node
    # because both use the id scheme "bt:intent-<job id>".
    matched_intent_e = doc.entity(
        matched_intent_id, {"prov:type": f"{NS}:Intent"}
    )
    doc.alternateOf(new_intent_e, matched_intent_e)

    lookup = doc.activity(
        lookup_id,
        startTime=timestamp,
        other_attributes={
            f"{NS}:type": f"{NS}:CacheLookupActivity",
            f"{NS}:similarity": record.get("similarity"),
            f"{NS}:model": model,
        },
    )
    doc.used(lookup, new_intent_e)

    served_e = doc.entity(
        served_workflow_id, {"prov:type": f"{NS}:BraneScriptWorkflow"}
    )
    doc.wasGeneratedBy(served_e, lookup)

    original_e = doc.entity(
        original_workflow_id, {"prov:type": f"{NS}:BraneScriptWorkflow"}
    )
    doc.wasDerivedFrom(served_e, original_e)

    return doc


# ---------------------------------------------------------------------------
# Directory-level export
# ---------------------------------------------------------------------------

def export_run(run_dir: Path) -> ProvDocument:
    """Read one TrainingCollector run directory and build its document."""
    record_path = run_dir / "execution_result.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return generation_record_to_document(record)


def export_dashboard_result(result_path: Path) -> ProvDocument:
    """Read one dashboard results/<id>.json file and build its document."""
    record = json.loads(result_path.read_text(encoding="utf-8"))
    return generation_record_to_document(record)


def export_all(
    training_data_dir: Path = DEFAULT_TRAINING_DIR,
    dashboard_results_dir: Optional[Path] = None,
) -> ProvDocument:
    """
    Merge every generation record and every cache-hit record found under
    *training_data_dir* (and, optionally, *dashboard_results_dir*) into a
    single PROV document.

    Merging is done by re-adding every statement into one document rather
    than by ProvDocument.update(), so that entities sharing the same
    qualified name (e.g. bt:workflow-<job id>, referenced both by the run
    that generated it and by a later cache hit that derived from it) are
    folded into a single node instead of duplicated.
    """
    combined = _new_document()

    runs_dir = training_data_dir / "runs"
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            record_path = run_dir / "execution_result.json"
            if not record_path.exists():
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            _merge_into(combined, generation_record_to_document(record))

    cache_hits_path = training_data_dir / "cache_hits.jsonl"
    if cache_hits_path.exists():
        with open(cache_hits_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                _merge_into(combined, cache_hit_record_to_document(record))

    if dashboard_results_dir and dashboard_results_dir.is_dir():
        for result_path in sorted(dashboard_results_dir.glob("*.json")):
            record = json.loads(result_path.read_text(encoding="utf-8"))
            _merge_into(combined, generation_record_to_document(record))

    return combined


def _merge_into(combined: ProvDocument, doc: ProvDocument) -> None:
    """Copy every record of *doc* into *combined*, deduplicating identical
    (same-identifier) entity/activity/agent declarations."""
    for record in doc.get_records():
        combined.add_record(record)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write(doc: ProvDocument, out_path: Path, fmt: str) -> None:
    if fmt == "dot":
        from prov.dot import prov_to_dot
        dot = prov_to_dot(doc)
        dot.write(str(out_path))
    else:
        doc.serialize(destination=str(out_path), format=fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path, help="a single TrainingCollector run directory")
    group.add_argument("--dashboard-result", type=Path, help="a single dashboard results/<id>.json file")
    group.add_argument("--all", action="store_true", help="every run + cache hit under TRAINING_DATA_DIR")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--format", choices=["json", "provn", "xml", "rdf", "dot"], default="json")
    parser.add_argument("--dashboard-results-dir", type=Path, default=None,
                        help="also fold in dashboard results/*.json when used with --all")
    args = parser.parse_args()

    if args.run_dir:
        doc = export_run(args.run_dir)
    elif args.dashboard_result:
        doc = export_dashboard_result(args.dashboard_result)
    else:
        doc = export_all(dashboard_results_dir=args.dashboard_results_dir)

    _write(doc, args.output, args.format)
    print(f"Wrote {args.format} PROV document to {args.output}")


if __name__ == "__main__":
    main()
