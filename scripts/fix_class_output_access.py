#!/usr/bin/env python3
"""Fix BraneScript examples to use field access for class-returning functions."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FILES = [
    REPO / "data/examples/packages.jsonl",
    REPO / "data/examples/training_500.jsonl",
    REPO / "data/training/execution_results.jsonl",
    REPO / "data/examples/advanced.jsonl",
]

# Map function_name -> fields to print from the returned class.
FUNC_FIELDS = {
    "compute_sentiment": ["polarity", "score"],
    "compute_readability": ["flesch_score", "grade_level"],
    "compute_summary_stats": ["mean", "std_dev", "count"],
    "compute_correlation": ["pearson_r", "interpretation"],
    "detect_pii": ["total_matches", "pii_found"],
    "compute_attack_rate": ["attack_rate_pct", "severity"],
    # Already-class functions to verify/fix too.
    "get_epidemic_status": ["stage", "alert_level"],
    "get_masking_summary": ["fields_masked", "strategy"],
    "get_text_stats": ["word_count", "sentiment"],
}

FUNC_PATTERN = "|".join(re.escape(name) for name in FUNC_FIELDS)
ASSIGNMENT_RE = re.compile(rf"\blet\s+(\w+)\s*:=\s*({FUNC_PATTERN})\s*\(")
PARALLEL_RE = re.compile(r"\blet\s+(\w+)\s*:=\s*parallel\b([\s\S]*?)\];")
RETURN_RE = re.compile(rf"\breturn\s+({FUNC_PATTERN})\s*\(")
RAW_PRINT_RE = re.compile(r"^([ \t]*)println\s*\(\s*(\w+)\s*\)\s*;\s*$")
RAW_INDEX_PRINT_RE = re.compile(r"^([ \t]*)println\s*\(\s*(\w+)\[(\d+)\]\s*\)\s*;\s*$")


def collect_assignments(bs: str) -> dict[str, str]:
    return {var: func for var, func in ASSIGNMENT_RE.findall(bs)}


def collect_parallel_returns(bs: str) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {}
    for var, body in PARALLEL_RE.findall(bs):
        indexed = {
            index: func
            for index, func in enumerate(RETURN_RE.findall(body))
            if func in FUNC_FIELDS
        }
        if indexed:
            result[var] = indexed
    return result


def projected_lines(expr: str, fields: list[str], indent: str) -> list[str]:
    return [f"{indent}println({expr}.{field});" for field in fields]


def fix_output_access(bs: str) -> tuple[str, int]:
    """Replace raw println() of class-returning results with field projection."""
    assignments = collect_assignments(bs)
    parallel_returns = collect_parallel_returns(bs)

    changed = 0
    new_lines: list[str] = []

    for line in bs.splitlines():
        match = RAW_PRINT_RE.match(line)
        if match:
            indent, var = match.groups()
            func = assignments.get(var)
            if func:
                new_lines.extend(projected_lines(var, FUNC_FIELDS[func], indent))
                changed += 1
                continue

        match = RAW_INDEX_PRINT_RE.match(line)
        if match:
            indent, var, index_text = match.groups()
            func = parallel_returns.get(var, {}).get(int(index_text))
            if func:
                expr = f"{var}[{index_text}]"
                new_lines.extend(projected_lines(expr, FUNC_FIELDS[func], indent))
                changed += 1
                continue

        new_lines.append(line)

    return "\n".join(new_lines), changed


def process_file(path: Path) -> tuple[int, int]:
    if not path.exists():
        print(f"  SKIP: {path.name}")
        return 0, 0

    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines = []
    total = 0
    changed = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            out_lines.append(line)
            continue

        bs = entry.get("branescript", "")
        if bs:
            new_bs, entry_changes = fix_output_access(bs)
            if entry_changes:
                entry["branescript"] = new_bs
                changed += entry_changes

        out_lines.append(json.dumps(entry, ensure_ascii=False))
        total += 1

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return total, changed


def main() -> None:
    for path in FILES:
        total, changed = process_file(path)
        print(f"  {path.name}: {total} entries, {changed} println() statements updated")


if __name__ == "__main__":
    main()
