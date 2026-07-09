#!/usr/bin/env python3
"""
Data Masking Package for NLP-Brane-Translator

Brane entrypoint: the action name is read from sys.argv[1] (set via
container.yml command.args). Input arguments arrive as uppercase env vars
whose values are JSON-serialised by branelet (e.g. VALUE, STRATEGY).

This package provides privacy-preserving data masking utilities aligned with
GDPR pseudonymisation and anonymisation requirements. It is designed to sit
at the start of a Brane pipeline so that downstream analytics never receive
raw sensitive fields.

Masking strategies
------------------
redact          Replace the entire value with [REDACTED].
hash            SHA-256 hex digest — deterministic pseudonymisation.
                Same plaintext always produces the same hash, preserving
                linkability within a dataset while hiding the original value.
partial         Retain only the last 4 characters; mask the rest with *.
                Useful for IDs and account numbers (e.g. ****-****-1234).
generalise_date Extract the year component from a date string.
                e.g. "1985-03-15" → "1985", "15/03/1985" → "1985".
mask_email      Redact the local part of an email address.
                e.g. "alice@example.com" → "****@example.com".

Function categories
-------------------
String-based (inline value/record via env vars)
  mask_value          -- apply a strategy to a single string value
  detect_pii          -- return PIIResult for regex-based PII detection
  mask_json_record    -- apply per-field strategies to a JSON object string

Class-returning
  get_masking_summary -- return a MaskingResult summarising a configuration

Data-based (Brane dataset; path via CSV_FILE / JSON_FILE env var)
  mask_csv_file       -- mask columns in a CSV, write masked CSV + report
  mask_json_file      -- mask fields in a JSON file, write masked JSON + report

IntermediateResult chaining
  generate_masking_report -- read masking IR, write HTML audit report
"""
import csv
import hashlib
import io
import json
import os
import re
import sys
import yaml
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Masking core
# ---------------------------------------------------------------------------

# Regex patterns for PII detection
_PII_PATTERNS = {
    'email':        re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
    'phone_intl':   re.compile(r'\+?[1-9]\d{0,2}[\s\-.]?\(?\d{1,4}\)?[\s\-.]?\d{1,4}[\s\-.]?\d{1,9}'),
    'date_iso':     re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
    'date_eu':      re.compile(r'\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b'),
    'ssn_us':       re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'credit_card':  re.compile(r'\b(?:\d[ \-]?){13,16}\b'),
    'postcode_nl':  re.compile(r'\b[1-9]\d{3}\s?[A-Z]{2}\b'),
    'iban':         re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b'),
}

# Date extraction: try common formats to pull out the year
_DATE_YEAR_PATTERNS = [
    re.compile(r'^(\d{4})-\d{2}-\d{2}'),    # ISO: 1985-03-15
    re.compile(r'^\d{1,2}[/.\-]\d{1,2}[/.\-](\d{4})$'),  # EU: 15/03/1985
    re.compile(r'^\d{1,2}[/.\-]\d{1,2}[/.\-](\d{2})$'),  # short year: 15/03/85
    re.compile(r'^(\d{4})'),                  # leading year fallback
]


def _mask_single(value: str, strategy: str) -> str:
    """Apply one masking strategy to a string value. Returns the masked string."""
    strategy = strategy.strip().lower()
    v = str(value)

    if strategy == 'redact':
        return '[REDACTED]'

    if strategy == 'hash':
        return hashlib.sha256(v.encode('utf-8')).hexdigest()

    if strategy == 'partial':
        visible = v[-4:] if len(v) >= 4 else v
        hidden = '*' * max(0, len(v) - 4)
        return hidden + visible

    if strategy == 'generalise_date':
        for pat in _DATE_YEAR_PATTERNS:
            m = pat.match(v.strip())
            if m:
                year = m.group(1)
                # Normalise 2-digit years
                if len(year) == 2:
                    year = ('19' if int(year) >= 25 else '20') + year
                return year
        return '[DATE]'

    if strategy == 'mask_email':
        m = re.match(r'^([^@]+)(@.+)$', v)
        if m:
            return '****' + m.group(2)
        return '[MASKED_EMAIL]'

    raise ValueError(f'Unknown masking strategy: {strategy!r}. '
                     f'Valid strategies: redact, hash, partial, generalise_date, mask_email')


def _detect_pii_in_text(text: str) -> Dict[str, List[str]]:
    """Return a dict mapping PII pattern names to lists of found examples."""
    findings: Dict[str, List[str]] = {}
    for name, pat in _PII_PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            # Deduplicate and keep at most 3 examples
            seen = []
            for m in matches:
                if m not in seen:
                    seen.append(m)
                if len(seen) == 3:
                    break
            findings[name] = seen
    return findings


def _parse_fields_json(raw: str) -> List[Dict[str, str]]:
    """Parse a JSON array of {field/column, strategy} dicts from a raw string.

    Also accepts plain string items (e.g. ["name", "email"]) and normalises
    them to {"field": name, "strategy": "redact"} for convenience.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'fields_json is not valid JSON: {exc}') from exc
    if not isinstance(parsed, list):
        raise ValueError('fields_json must be a JSON array')
    normalized = []
    for item in parsed:
        if isinstance(item, str):
            normalized.append({'field': item, 'strategy': 'redact'})
        else:
            normalized.append(item)
    return normalized


def _apply_fields_to_record(
    record: Dict[str, Any],
    fields: List[Dict[str, str]],
    key_name: str = 'field',
) -> Tuple[Dict[str, Any], int]:
    """Apply masking to a dict record. Returns (masked_record, n_values_masked)."""
    masked = dict(record)
    count = 0
    for spec in fields:
        col = spec.get(key_name) or spec.get('column') or spec.get('field', '')
        strategy = spec.get('strategy', 'redact')
        if col in masked and masked[col] is not None:
            masked[col] = _mask_single(str(masked[col]), strategy)
            count += 1
    return masked, count


# ---------------------------------------------------------------------------
# Brane I/O helpers
# ---------------------------------------------------------------------------

def _env_str(name: str) -> str:
    raw = os.environ.get(name.upper(), '""')
    try:
        v = json.loads(raw)
        return str(v)
    except (json.JSONDecodeError, ValueError):
        return raw


def _resolve_env_path(var_name: str) -> str:
    raw = os.environ.get(var_name, '')
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, str):
            return decoded
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def _out_str(value: str) -> None:
    print(f'output: {json.dumps(str(value))}', flush=True)


def _out_class(class_name: str, fields: Dict[str, Any]) -> None:
    print(yaml.dump({'output': [class_name, fields]}), end='', flush=True)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def action_mask_value() -> None:
    value = _env_str('VALUE')
    strategy = _env_str('STRATEGY')
    _out_str(_mask_single(value, strategy))


def action_detect_pii() -> None:
    text = _env_str('TEXT')
    findings = _detect_pii_in_text(text)
    total = sum(len(v) for v in findings.values())
    _out_class('PIIResult', {
        'total_matches': total,
        'pattern_types_found': len(findings),
        'pii_found': total > 0,
    })


def action_mask_json_record() -> None:
    record_str = _env_str('RECORD')
    fields_raw = _env_str('FIELDS_JSON')
    try:
        record = json.loads(record_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f'record is not valid JSON: {exc}') from exc
    fields = _parse_fields_json(fields_raw)
    masked, _ = _apply_fields_to_record(record, fields, key_name='field')
    _out_str(json.dumps(masked))


def action_get_masking_summary() -> None:
    """Return a MaskingResult class instance for a given strategy + fields config."""
    strategy = _env_str('STRATEGY')
    fields_raw = _env_str('FIELDS_JSON')
    fields = _parse_fields_json(fields_raw)
    _out_class('MaskingResult', {
        'records_processed': 0,
        'fields_masked': len(fields),
        'strategy': strategy,
        'output_format': 'config_only',
    })


def action_mask_csv_file() -> None:
    """Data-based: mask specified columns in a CSV dataset."""
    path = _resolve_env_path('CSV_FILE')
    if not path:
        raise RuntimeError('CSV_FILE not set — invoke via a Data reference')
    fields_raw = _env_str('FIELDS_JSON')
    fields = _parse_fields_json(fields_raw)

    # Build a lookup: column name → strategy
    col_strategy: Dict[str, str] = {}
    for spec in fields:
        col = spec.get('column') or spec.get('field', '')
        col_strategy[col] = spec.get('strategy', 'redact')

    # Resolve single file or directory (use first CSV found)
    if os.path.isdir(path):
        csv_files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f'No CSV files found in directory {path}')
        csv_path = os.path.join(path, sorted(csv_files)[0])
    else:
        csv_path = path

    # Read and mask
    with open(csv_path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    masked_rows = []
    total_masked = 0
    col_counts: Dict[str, int] = {c: 0 for c in col_strategy}

    for row in rows:
        masked_row = dict(row)
        for col, strategy in col_strategy.items():
            if col in masked_row and masked_row[col]:
                masked_row[col] = _mask_single(str(masked_row[col]), strategy)
                col_counts[col] = col_counts.get(col, 0) + 1
                total_masked += 1
        masked_rows.append(masked_row)

    os.makedirs('/result', exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Write masked CSV
    with open('/result/masked_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(masked_rows)

    # Write masking report
    report = {
        'source_file': os.path.basename(csv_path),
        'records_processed': len(rows),
        'total_values_masked': total_masked,
        'fields_masked': col_counts,
        'masking_config': fields,
        'timestamp': ts,
    }
    with open('/result/masking_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def action_mask_json_file() -> None:
    """Data-based: mask specified fields in a JSON dataset."""
    path = _resolve_env_path('JSON_FILE')
    if not path:
        raise RuntimeError('JSON_FILE not set — invoke via a Data reference')
    fields_raw = _env_str('FIELDS_JSON')
    fields = _parse_fields_json(fields_raw)

    if os.path.isdir(path):
        json_files = [f for f in os.listdir(path) if f.lower().endswith('.json')]
        if not json_files:
            raise FileNotFoundError(f'No JSON files found in directory {path}')
        json_path = os.path.join(path, sorted(json_files)[0])
    else:
        json_path = path

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]
    masked_records = []
    total_masked = 0
    field_counts: Dict[str, int] = {}

    for rec in records:
        if not isinstance(rec, dict):
            masked_records.append(rec)
            continue
        masked_rec, n = _apply_fields_to_record(rec, fields, key_name='field')
        total_masked += n
        for spec in fields:
            col = spec.get('field') or spec.get('column', '')
            if col in rec and rec[col] is not None:
                field_counts[col] = field_counts.get(col, 0) + 1
        masked_records.append(masked_rec)

    os.makedirs('/result', exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    output_data = masked_records if isinstance(data, list) else masked_records[0]
    with open('/result/masked_data.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    report = {
        'source_file': os.path.basename(json_path),
        'records_processed': len(records),
        'total_values_masked': total_masked,
        'fields_masked': field_counts,
        'masking_config': fields,
        'timestamp': ts,
    }
    with open('/result/masking_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def action_generate_masking_report() -> None:
    """IntermediateResult chaining: read masking report JSON, write HTML audit report."""
    result_dir = _resolve_env_path('MASKING_RESULT')
    if not result_dir:
        raise RuntimeError('MASKING_RESULT not set — invoke via an IntermediateResult reference')

    report_path = None
    for root, _, files in os.walk(result_dir):
        if 'masking_report.json' in files:
            report_path = os.path.join(root, 'masking_report.json')
            break
    if report_path is None:
        raise FileNotFoundError(f'masking_report.json not found under {result_dir}')

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    os.makedirs('/result', exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Build field rows
    fields_masked = report.get('fields_masked', {})
    masking_config = {
        spec.get('column') or spec.get('field', ''): spec.get('strategy', 'redact')
        for spec in report.get('masking_config', [])
    }
    field_rows = ''.join(
        f'<tr><td style="padding:4px 12px">{col}</td>'
        f'<td style="padding:4px 12px"><code>{masking_config.get(col, "?")}</code></td>'
        f'<td style="padding:4px 12px">{count}</td></tr>'
        for col, count in fields_masked.items()
    )

    strategy_descriptions = {
        'redact':          'Replaced with [REDACTED]',
        'hash':            'SHA-256 pseudonymisation (deterministic)',
        'partial':         'Last 4 characters retained, rest masked with *',
        'generalise_date': 'Year extracted, day and month removed',
        'mask_email':      'Local part replaced with ****',
    }
    legend_rows = ''.join(
        f'<tr><td style="padding:4px 12px"><code>{k}</code></td>'
        f'<td style="padding:4px 12px">{v}</td></tr>'
        for k, v in strategy_descriptions.items()
    )

    html = (
        '<!DOCTYPE html><html><head>'
        '<meta charset="utf-8">'
        '<title>Data Masking Audit Report</title>'
        '<style>body{font-family:sans-serif;margin:40px;color:#222}'
        'h1{color:#b00020}h2{color:#333;border-bottom:1px solid #ccc}'
        'table{border-collapse:collapse}td,th{border:1px solid #ccc}'
        'th{background:#f5f5f5;padding:4px 12px}'
        '.badge{display:inline-block;padding:2px 8px;border-radius:4px;'
        'background:#e8f5e9;color:#2e7d32;font-weight:bold}</style>'
        '</head><body>'
        '<h1>&#128274; Data Masking Audit Report</h1>'
        f'<p>Generated: {ts}</p>'
        '<h2>Summary</h2>'
        '<table><tr><th>Metric</th><th>Value</th></tr>'
        f'<tr><td>Source file</td><td>{report.get("source_file", "—")}</td></tr>'
        f'<tr><td>Records processed</td><td>{report.get("records_processed", 0)}</td></tr>'
        f'<tr><td>Total values masked</td><td>{report.get("total_values_masked", 0)}</td></tr>'
        f'<tr><td>Fields targeted</td><td>{len(fields_masked)}</td></tr>'
        f'<tr><td>Masking timestamp</td><td>{report.get("timestamp", "—")}</td></tr>'
        '</table>'
        '<h2>Per-field Breakdown</h2>'
        '<table><tr><th>Field / Column</th><th>Strategy</th><th>Values Masked</th></tr>'
        f'{field_rows}</table>'
        '<h2>Strategy Reference</h2>'
        '<table><tr><th>Strategy</th><th>Description</th></tr>'
        f'{legend_rows}</table>'
        '<h2>Compliance Notes</h2>'
        '<ul>'
        '<li><strong>hash</strong> constitutes <em>pseudonymisation</em> under GDPR Art. 4(5) — '
        'the original value can be recovered if the hash input is known.</li>'
        '<li><strong>redact</strong>, <strong>generalise_date</strong>, and <strong>mask_email</strong> '
        'are forms of <em>anonymisation</em> where original values cannot be recovered from the output.</li>'
        '<li><strong>partial</strong> masking retains linkability for the visible suffix; '
        'treat as pseudonymisation.</li>'
        '</ul>'
        '</body></html>'
    )

    with open('/result/masking_audit_report.html', 'w', encoding='utf-8') as f:
        f.write(html)

    # Copy the original masking_report.json through to the new IR
    import shutil
    shutil.copy(report_path, '/result/masking_report.json')


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ACTIONS = {
    'mask_value':             action_mask_value,
    'detect_pii':             action_detect_pii,
    'mask_json_record':       action_mask_json_record,
    'get_masking_summary':    action_get_masking_summary,
    'mask_csv_file':          action_mask_csv_file,
    'mask_json_file':         action_mask_json_file,
    'generate_masking_report': action_generate_masking_report,
}


def main() -> None:
    if len(sys.argv) < 2:
        _out_str(json.dumps({'error': 'No action name in argv[1]', 'status': 'failed'}))
        sys.exit(1)
    action = sys.argv[1]
    handler = _ACTIONS.get(action)
    if handler is None:
        _out_str(json.dumps({'error': f'Unknown action: {action!r}', 'status': 'failed'}))
        sys.exit(1)
    handler()


if __name__ == '__main__':
    main()
