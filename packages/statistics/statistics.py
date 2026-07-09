#!/usr/bin/env python3
"""
Statistics Package for NLP-Brane-Translator

Generic CSV statistical analysis: summary stats, group aggregation,
correlation, outlier detection, filtering, sorting, and normalization.
All functions operate on a registered Brane dataset (CSV file).

Brane input convention:
  - Data inputs: path passed via uppercase env var (e.g. DATA_FILE)
  - String/numeric inputs: JSON-encoded via uppercase env var
  - String outputs: printed as  output: "<json-string>"
  - IntermediateResult outputs: files written to /result/
"""

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Env var helpers
# ---------------------------------------------------------------------------

def _resolve_env_path(var_name: str) -> str:
    raw = os.environ.get(var_name, '')
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, str):
            return decoded
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def _env_str(name: str) -> str:
    raw = os.environ.get(name.upper(), '""')
    try:
        v = json.loads(raw)
        return str(v)
    except (json.JSONDecodeError, ValueError):
        return raw


def _env_int(name: str, default: int = 10) -> int:
    raw = os.environ.get(name.upper(), str(default))
    try:
        return int(json.loads(raw))
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name.upper(), str(default))
    try:
        return float(json.loads(raw))
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


def _out_str(value: str) -> None:
    print(f'output: {json.dumps(str(value))}', flush=True)


def _out_class(class_name: str, fields: Dict[str, Any]) -> None:
    """Print a class instance as a YAML 2-element list ["ClassName", {…}]."""
    print(yaml.dump({'output': [class_name, fields]}), end='', flush=True)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (fieldnames, rows) from a CSV file."""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _numeric_values(rows: List[Dict[str, str]], column: str) -> List[float]:
    """Extract non-null numeric values for a given column."""
    values = []
    for row in rows:
        v = _parse_float(row.get(column, ''))
        if v is not None:
            values.append(v)
    return values


def _stats(values: List[float]) -> Dict[str, Any]:
    """Compute descriptive statistics."""
    if not values:
        return {'count': 0, 'mean': None, 'std_dev': None, 'min': None, 'max': None, 'median': None, 'q25': None, 'q75': None}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)
    sv = sorted(values)

    def _percentile(p: float) -> float:
        idx = (n - 1) * p
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)

    return {
        'count': n,
        'mean': round(mean, 4),
        'std_dev': round(std_dev, 4),
        'min': round(sv[0], 4),
        'max': round(sv[-1], 4),
        'median': round(_percentile(0.5), 4),
        'q25': round(_percentile(0.25), 4),
        'q75': round(_percentile(0.75), 4),
    }


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def action_compute_summary_stats() -> None:
    """Compute descriptive statistics for a numeric column in a CSV dataset."""
    path = _resolve_env_path('DATA_FILE')
    column = _env_str('COLUMN')
    if not path:
        _out_str(json.dumps({'error': 'DATA_FILE not set'}))
        return
    _, rows = _load_csv(path)
    values = _numeric_values(rows, column)
    if not values:
        _out_str(json.dumps({'error': f'Column {column!r} not found or has no numeric values'}))
        return
    result = _stats(values)
    result['column'] = column
    result['dataset'] = os.path.basename(path)
    _out_class('SummaryStats', {
        'column': result.get('column', column),
        'count': result.get('count', 0),
        'mean': result.get('mean') or 0.0,
        'std_dev': result.get('std_dev') or 0.0,
        'min_val': result.get('min') or 0.0,
        'max_val': result.get('max') or 0.0,
        'median': result.get('median') or 0.0,
    })


def action_count_by_category() -> None:
    """Count rows per unique value in a categorical column."""
    path = _resolve_env_path('DATA_FILE')
    column = _env_str('COLUMN')
    if not path:
        _out_str(json.dumps({'error': 'DATA_FILE not set'}))
        return
    _, rows = _load_csv(path)
    counts: Dict[str, int] = {}
    for row in rows:
        val = row.get(column, '').strip()
        counts[val] = counts.get(val, 0) + 1
    sorted_counts = dict(sorted(counts.items(), key=lambda x: -x[1]))
    _out_str(json.dumps({
        'column': column,
        'total_rows': len(rows),
        'category_counts': sorted_counts,
        'unique_categories': len(sorted_counts),
    }))


def action_compute_correlation() -> None:
    """Compute Pearson correlation coefficient between two numeric columns."""
    path = _resolve_env_path('DATA_FILE')
    col_a = _env_str('COL_A')
    col_b = _env_str('COL_B')
    if not path:
        _out_str(json.dumps({'error': 'DATA_FILE not set'}))
        return
    _, rows = _load_csv(path)

    pairs = []
    for row in rows:
        a = _parse_float(row.get(col_a, ''))
        b = _parse_float(row.get(col_b, ''))
        if a is not None and b is not None:
            pairs.append((a, b))

    if len(pairs) < 2:
        _out_str(json.dumps({'error': 'Not enough numeric pairs to compute correlation'}))
        return

    n = len(pairs)
    mean_a = sum(p[0] for p in pairs) / n
    mean_b = sum(p[1] for p in pairs) / n
    cov = sum((p[0] - mean_a) * (p[1] - mean_b) for p in pairs) / n
    std_a = math.sqrt(sum((p[0] - mean_a) ** 2 for p in pairs) / n)
    std_b = math.sqrt(sum((p[1] - mean_b) ** 2 for p in pairs) / n)

    if std_a == 0 or std_b == 0:
        _out_str(json.dumps({'error': 'Standard deviation is zero — correlation undefined'}))
        return

    r = round(cov / (std_a * std_b), 4)
    if abs(r) >= 0.7:
        interpretation = 'strong'
    elif abs(r) >= 0.4:
        interpretation = 'moderate'
    else:
        interpretation = 'weak'
    direction = 'positive' if r >= 0 else 'negative'

    _out_class('CorrelationResult', {
        'col_a': col_a,
        'col_b': col_b,
        'pearson_r': r,
        'interpretation': f'{interpretation} {direction} correlation',
        'n_pairs': n,
    })


def action_detect_outliers() -> None:
    """Detect outliers in a numeric column using IQR or z-score method.

    Input: data_file (Data), column (string), method (string: 'iqr' or 'zscore').
    Returns: JSON string with outlier rows and statistics.
    """
    path = _resolve_env_path('DATA_FILE')
    column = _env_str('COLUMN')
    method = _env_str('METHOD').lower().strip() or 'iqr'
    if not path:
        _out_str(json.dumps({'error': 'DATA_FILE not set'}))
        return
    _, rows = _load_csv(path)
    values = _numeric_values(rows, column)
    if not values:
        _out_str(json.dumps({'error': f'Column {column!r} has no numeric values'}))
        return

    outlier_indices = set()
    threshold_info = {}

    if method == 'iqr':
        sv = sorted(values)
        n = len(sv)

        def _pct(p):
            idx = (n - 1) * p
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)

        q1 = _pct(0.25)
        q3 = _pct(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        threshold_info = {'method': 'iqr', 'lower_fence': round(lower, 4), 'upper_fence': round(upper, 4), 'q1': round(q1, 4), 'q3': round(q3, 4)}
        for i, row in enumerate(rows):
            v = _parse_float(row.get(column, ''))
            if v is not None and (v < lower or v > upper):
                outlier_indices.add(i)
    else:
        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
        threshold_info = {'method': 'zscore', 'mean': round(mean, 4), 'std_dev': round(std, 4), 'threshold': 3.0}
        for i, row in enumerate(rows):
            v = _parse_float(row.get(column, ''))
            if v is not None and std > 0:
                z = abs((v - mean) / std)
                if z > 3.0:
                    outlier_indices.add(i)

    outliers = []
    for i in sorted(outlier_indices):
        row = rows[i]
        outliers.append({
            'row_index': i,
            'value': row.get(column, ''),
            'id': row.get('patient_id') or row.get('id') or str(i),
        })

    _out_str(json.dumps({
        'column': column,
        'total_rows': len(rows),
        'outlier_count': len(outliers),
        'outliers': outliers,
        'statistics': threshold_info,
    }))


def action_filter_by_threshold() -> None:
    """Filter CSV rows where column value satisfies operator vs threshold.

    Input: data_file (Data), column (string), operator (string: gt/gte/lt/lte/eq/neq), threshold (real).
    Output: /result/filtered.csv and /result/filter_summary.json
    """
    path = _resolve_env_path('DATA_FILE')
    column = _env_str('COLUMN')
    operator = _env_str('OPERATOR').lower().strip()
    threshold = _env_float('THRESHOLD')
    if not path:
        raise RuntimeError('DATA_FILE not set')

    ops = {
        'gt': lambda v, t: v > t,
        'gte': lambda v, t: v >= t,
        'lt': lambda v, t: v < t,
        'lte': lambda v, t: v <= t,
        'eq': lambda v, t: v == t,
        'neq': lambda v, t: v != t,
    }
    op_fn = ops.get(operator, ops['gt'])

    fieldnames, rows = _load_csv(path)
    matched = []
    for row in rows:
        v = _parse_float(row.get(column, ''))
        if v is not None and op_fn(v, threshold):
            matched.append(row)

    os.makedirs('/result', exist_ok=True)
    with open('/result/filtered.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(matched)

    with open('/result/filter_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'column': column,
            'operator': operator,
            'threshold': threshold,
            'total_rows': len(rows),
            'matched_rows': len(matched),
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        }, f, indent=2)


def action_sort_and_rank() -> None:
    """Sort CSV by a numeric column and add a rank column.

    Input: data_file (Data), column (string), descending (string: 'true'/'false').
    Output: /result/sorted.csv
    """
    path = _resolve_env_path('DATA_FILE')
    column = _env_str('COLUMN')
    descending = _env_str('DESCENDING').lower().strip() in ('true', '1', 'yes')
    if not path:
        raise RuntimeError('DATA_FILE not set')

    fieldnames, rows = _load_csv(path)

    def _sort_key(row):
        v = _parse_float(row.get(column, ''))
        return v if v is not None else (float('inf') if not descending else float('-inf'))

    sorted_rows = sorted(rows, key=_sort_key, reverse=descending)
    for rank, row in enumerate(sorted_rows, start=1):
        row['rank'] = str(rank)

    out_fields = fieldnames + ['rank'] if 'rank' not in fieldnames else fieldnames
    os.makedirs('/result', exist_ok=True)
    with open('/result/sorted.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted_rows)


def action_aggregate_by_group() -> None:
    """Group CSV by a categorical column and aggregate a numeric column.

    Input: data_file (Data), group_col (string), value_col (string), func (string: mean/sum/count/min/max).
    Returns: JSON string with aggregated result per group.
    """
    path = _resolve_env_path('DATA_FILE')
    group_col = _env_str('GROUP_COL')
    value_col = _env_str('VALUE_COL')
    func = _env_str('FUNC').lower().strip() or 'mean'
    if not path:
        _out_str(json.dumps({'error': 'DATA_FILE not set'}))
        return

    _, rows = _load_csv(path)
    groups: Dict[str, List[float]] = {}
    for row in rows:
        grp = row.get(group_col, '').strip()
        v = _parse_float(row.get(value_col, ''))
        if v is not None:
            groups.setdefault(grp, []).append(v)

    result: Dict[str, Any] = {}
    for grp, vals in sorted(groups.items()):
        if func == 'mean':
            result[grp] = round(sum(vals) / len(vals), 4) if vals else None
        elif func == 'sum':
            result[grp] = round(sum(vals), 4)
        elif func == 'count':
            result[grp] = len(vals)
        elif func == 'min':
            result[grp] = round(min(vals), 4)
        elif func == 'max':
            result[grp] = round(max(vals), 4)
        else:
            result[grp] = round(sum(vals) / len(vals), 4)

    _out_str(json.dumps({
        'group_col': group_col,
        'value_col': value_col,
        'func': func,
        'groups': result,
        'group_count': len(result),
    }))


def action_normalize_column() -> None:
    """Normalize a numeric column using min-max or z-score normalization.

    Input: data_file (Data), column (string), method (string: 'minmax'/'zscore').
    Output: /result/normalized.csv  (original CSV with extra column <column>_normalized)
    """
    path = _resolve_env_path('DATA_FILE')
    column = _env_str('COLUMN')
    method = _env_str('METHOD').lower().strip() or 'minmax'
    if not path:
        raise RuntimeError('DATA_FILE not set')

    fieldnames, rows = _load_csv(path)
    values = _numeric_values(rows, column)
    if not values:
        raise ValueError(f'Column {column!r} has no numeric values')

    if method == 'minmax':
        vmin = min(values)
        vmax = max(values)
        rng = vmax - vmin if vmax != vmin else 1.0

        def _norm(v):
            return round((v - vmin) / rng, 6)
    else:
        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)) or 1.0

        def _norm(v):
            return round((v - mean) / std, 6)

    new_col = f'{column}_normalized'
    out_fields = fieldnames + [new_col] if new_col not in fieldnames else fieldnames
    for row in rows:
        v = _parse_float(row.get(column, ''))
        row[new_col] = str(_norm(v)) if v is not None else ''

    os.makedirs('/result', exist_ok=True)
    with open('/result/normalized.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ACTIONS = {
    'compute_summary_stats':   action_compute_summary_stats,
    'count_by_category':       action_count_by_category,
    'compute_correlation':     action_compute_correlation,
    'detect_outliers':         action_detect_outliers,
    'filter_by_threshold':     action_filter_by_threshold,
    'sort_and_rank':           action_sort_and_rank,
    'aggregate_by_group':      action_aggregate_by_group,
    'normalize_column':        action_normalize_column,
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
