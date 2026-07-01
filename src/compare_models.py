"""
compare_models.py

Side-by-side comparison of two model evaluation reports.

Usage
-----
    python src/compare_models.py \\
        evaluation_results/baseline/report.json \\
        evaluation_results/finetuned/report.json

    # Custom labels:
    python src/compare_models.py \\
        evaluation_results/qwen3-4b/report.json \\
        evaluation_results/qwen3-27b/report.json \\
        --labels "Qwen3-4B" "Qwen3-27B"
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v:.1%}"


def _delta(a: float, b: float) -> str:
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1%}"


def _col(s: str, width: int) -> str:
    return str(s).ljust(width)


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison(report_a: dict, report_b: dict, label_a: str, label_b: str) -> None:

    COL_W = 26

    def row(name: str, a_val: str, b_val: str, delta: str = "") -> None:
        print(f"  {_col(name, 30)} {_col(a_val, COL_W)} {_col(b_val, COL_W)} {delta}")

    header = f"  {'Metric':<30} {_col(label_a, COL_W)} {_col(label_b, COL_W)} Delta"
    sep    = "  " + "─" * (30 + COL_W * 2 + 4)

    print()
    print("=" * (30 + COL_W * 2 + 8))
    print("  EVALUATION COMPARISON")
    print("=" * (30 + COL_W * 2 + 8))
    print(header)
    print(sep)

    def metric(name: str, key: str, fmt=_pct) -> None:
        a = report_a.get(key)
        b = report_b.get(key)
        if a is None or b is None:
            av, bv, d = str(a), str(b), "—"
        else:
            av, bv = fmt(a), fmt(b)
            d = _delta(a, b)
        row(name, av, bv, d)

    def int_metric(name: str, key: str) -> None:
        a = report_a.get(key, "—")
        b = report_b.get(key, "—")
        row(name, str(a), str(b))

    int_metric("Total benchmark items",     "total")
    metric("Validation pass rate",          "validation_pass_rate")
    metric("Functional match rate",         "functional_match_rate")
    metric("First-attempt success rate",    "first_attempt_rate")

    def float_metric(name: str, key: str) -> None:
        a = report_a.get(key)
        b = report_b.get(key)
        av = f"{a:.2f}" if a is not None else "—"
        bv = f"{b:.2f}" if b is not None else "—"
        d  = f"+{b-a:.2f}" if (a is not None and b is not None) else "—"
        row(name, av, bv, d)

    float_metric("Mean generation attempts",    "mean_attempts")

    # Execution metrics (optional)
    if "execution_match_rate" in report_a or "execution_match_rate" in report_b:
        print(sep)
        metric("Execution match rate",  "execution_match_rate")
        int_metric("Execution run count", "execution_run_count")

    # By difficulty
    print(sep)
    print(f"\n  {'By difficulty':<30} {_col('functional_match', COL_W)} {_col('functional_match', COL_W)} Delta")
    print(sep)
    for diff in ("easy", "medium", "hard"):
        da = report_a.get("by_difficulty", {}).get(diff, {})
        db = report_b.get("by_difficulty", {}).get(diff, {})
        ra = da.get("functional_match_rate")
        rb = db.get("functional_match_rate")
        na = da.get("count", 0)
        nb = db.get("count", 0)
        av = f"{_pct(ra)} (n={na})" if ra is not None else "—"
        bv = f"{_pct(rb)} (n={nb})" if rb is not None else "—"
        d  = _delta(ra, rb) if (ra is not None and rb is not None) else "—"
        row(f"  {diff}", av, bv, d)

    # By tag (show tags present in either report)
    all_tags = sorted(
        set(report_a.get("by_tag", {}).keys()) |
        set(report_b.get("by_tag", {}).keys())
    )
    if all_tags:
        print(sep)
        print(f"\n  {'By tag':<30} {_col('functional_match', COL_W)} {_col('functional_match', COL_W)} Delta")
        print(sep)
        for tag in all_tags:
            da = report_a.get("by_tag", {}).get(tag, {})
            db = report_b.get("by_tag", {}).get(tag, {})
            ra = da.get("functional_match_rate")
            rb = db.get("functional_match_rate")
            na = da.get("count", 0)
            nb = db.get("count", 0)
            av = f"{_pct(ra)} (n={na})" if ra is not None else "—"
            bv = f"{_pct(rb)} (n={nb})" if rb is not None else "—"
            d  = _delta(ra, rb) if (ra is not None and rb is not None) else "—"
            row(f"  {tag}", av, bv, d)

    print()
    print(f"  Model A timestamp: {report_a.get('timestamp', '—')}")
    print(f"  Model B timestamp: {report_b.get('timestamp', '—')}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two model evaluation reports side by side")
    parser.add_argument("report_a", help="Path to first report.json")
    parser.add_argument("report_b", help="Path to second report.json")
    parser.add_argument(
        "--labels", nargs=2, metavar=("LABEL_A", "LABEL_B"), default=None,
        help="Human-readable labels for the two models (default: model names from reports)."
    )
    args = parser.parse_args()

    def load(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            print(f"❌ File not found: {p}")
            sys.exit(1)
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    report_a = load(args.report_a)
    report_b = load(args.report_b)

    if args.labels:
        label_a, label_b = args.labels
    else:
        label_a = report_a.get("model", Path(args.report_a).parent.name)
        label_b = report_b.get("model", Path(args.report_b).parent.name)

    print_comparison(report_a, report_b, label_a, label_b)


if __name__ == "__main__":
    main()
