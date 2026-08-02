#!/usr/bin/env python3
"""
plot_results.py

Visualise evaluation and cache benchmark results for the thesis.

Subcommands:
  cache-sweep   — threshold vs hit/correct/FP rate from cache_benchmark_sweep.json
  eval          — compile & execution rate comparison across all models
  sft-vs-base   — side-by-side SFT vs base for the same model family

Usage:
  python scripts/plot_results.py cache-sweep
  python scripts/plot_results.py eval
  python scripts/plot_results.py sft-vs-base
  python scripts/plot_results.py cache-sweep --save
  python scripts/plot_results.py eval --save
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
except ImportError:
    print("Install matplotlib and numpy: pip install matplotlib numpy")
    sys.exit(1)

EVAL_DIR  = ROOT / "outputs" / "eval"
FIG_DIR   = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "hit":     "#4C72B0",
    "correct": "#55A868",
    "fp":      "#C44E52",
    "compile": "#4C72B0",
    "exec":    "#55A868",
    "base":    "#4C72B0",
    "sft":     "#DD8452",
    "grpo":    "#8172B2",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cache sweep plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_cache_sweep(save: bool):
    sweep_file = EVAL_DIR / "cache_benchmark_sweep.json"
    if not sweep_file.exists():
        print(f"❌ Not found: {sweep_file}")
        print("   Run the sweep first: python scripts/test_cache_lookup.py --sweep --output outputs/eval/cache_benchmark_sweep.json")
        sys.exit(1)

    data = json.loads(sweep_file.read_text())
    rows = data["sweep"]

    thresholds  = [r["threshold"]        for r in rows]
    hit_rates   = [r["hit_rate_pct"]     for r in rows]
    correct     = [r["correct_rate_pct"] for r in rows]
    fp_rates    = [r["fp_rate_pct"]      for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds, hit_rates, "o-",  color=COLORS["hit"],     label="Hit rate %",     linewidth=2)
    ax.plot(thresholds, correct,   "s-",  color=COLORS["correct"], label="Correct rate %", linewidth=2)
    ax.plot(thresholds, fp_rates,  "^--", color=COLORS["fp"],      label="False positive % (of hits)", linewidth=1.5, alpha=0.8)

    # Mark the best threshold (highest correct rate with FP < 5%)
    candidates = [(r["correct_rate_pct"], r["threshold"]) for r in rows if r["fp_rate_pct"] < 5.0]
    if candidates:
        best_correct, best_t = max(candidates)
        ax.axvline(best_t, color="gray", linestyle=":", linewidth=1.5, label=f"Suggested threshold ({best_t})")
        ax.annotate(f"  t={best_t}", xy=(best_t, best_correct), fontsize=9, color="gray")

    ax.set_xlabel("Cosine Similarity Threshold", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Semantic Cache: Threshold vs Accuracy", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.set_xticks(thresholds)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        out = FIG_DIR / "cache_sweep.png"
        fig.savefig(out, dpi=150)
        print(f"💾 Saved → {out}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Eval comparison plot (all models)
# ─────────────────────────────────────────────────────────────────────────────

def _load_eval_results() -> list[dict]:
    """Load all *_eval.json files and return summary rows."""
    results = []
    for f in sorted(EVAL_DIR.glob("*_eval.json")):
        try:
            data = json.loads(f.read_text())
            summary = data.get("summary", {})
            model   = data.get("model", f.stem.replace("_eval", ""))
            label   = data.get("label", Path(model).name)
            results.append({
                "label":        label,
                "model":        model,
                "compile_rate": summary.get("compile_rate_pct", 0),
                "exec_rate":    summary.get("execution_rate_pct", 0),
                "file":         f.name,
                "is_sft":       "sft" in f.name.lower() or "merged" in model.lower()
                                 or ("output" not in model and "/" not in model and model != label),
            })
        except Exception:
            pass
    return results


def plot_eval(save: bool):
    rows = _load_eval_results()
    if not rows:
        print(f"❌ No *_eval.json files found in {EVAL_DIR}")
        sys.exit(1)

    labels       = [r["label"]        for r in rows]
    compile_rate = [r["compile_rate"] for r in rows]
    exec_rate    = [r["exec_rate"]    for r in rows]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))

    bars1 = ax.bar(x - w/2, compile_rate, w, label="Compile rate %", color=COLORS["compile"], alpha=0.85)
    bars2 = ax.bar(x + w/2, exec_rate,    w, label="Execution rate %", color=COLORS["exec"],    alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Model Evaluation: Compile & Execution Rate", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        out = FIG_DIR / "eval_comparison.png"
        fig.savefig(out, dpi=150)
        print(f"💾 Saved → {out}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SFT vs Base side-by-side
# ─────────────────────────────────────────────────────────────────────────────

def plot_sft_vs_base(save: bool):
    rows = _load_eval_results()
    if not rows:
        print(f"❌ No *_eval.json files found in {EVAL_DIR}")
        sys.exit(1)

    # Group by model family (e.g. "qwen3.5-9b")
    families: dict[str, dict] = {}
    for r in rows:
        label = r["label"].lower()
        for family in ["qwen3.5-9b", "qwen3.6-27b", "qwen3.5-4b"]:
            if family in label or family in r["model"].lower():
                if family not in families:
                    families[family] = {}
                role = "sft" if ("sft" in label or "ep" in label or "merged" in r["model"].lower()) else "base"
                families[family][role] = r
                break

    if not families:
        print("❌ Could not match any model to a known family (qwen3.5-9b, qwen3.6-27b, qwen3.5-4b)")
        sys.exit(1)

    fam_names = sorted(families.keys())
    n = len(fam_names)
    x = np.arange(n)
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax_i, (metric, title) in enumerate([("compile_rate", "Compile Rate %"), ("exec_rate", "Execution Rate %")]):
        ax = axes[ax_i]
        base_vals = [families[f].get("base", {}).get(metric, 0) for f in fam_names]
        sft_vals  = [families[f].get("sft",  {}).get(metric, 0) for f in fam_names]

        bars1 = ax.bar(x - w/2, base_vals, w*1.8, label="Base",  color=COLORS["base"], alpha=0.85)
        bars2 = ax.bar(x + w/2, sft_vals,  w*1.8, label="SFT",   color=COLORS["sft"],  alpha=0.85)

        for bar in bars1 + bars2:
            v = bar.get_height()
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(fam_names, rotation=15, ha="right", fontsize=10)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        ax.set_ylim(0, 110)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle("SFT vs Base Model Performance", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        out = FIG_DIR / "sft_vs_base.png"
        fig.savefig(out, dpi=150)
        print(f"💾 Saved → {out}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Plot thesis results")
    ap.add_argument("plot", choices=["cache-sweep", "eval", "sft-vs-base"],
                    help="Which plot to generate")
    ap.add_argument("--save", action="store_true",
                    help="Save to outputs/figures/ instead of showing interactively")
    args = ap.parse_args()

    if args.plot == "cache-sweep":
        plot_cache_sweep(args.save)
    elif args.plot == "eval":
        plot_eval(args.save)
    elif args.plot == "sft-vs-base":
        plot_sft_vs_base(args.save)


if __name__ == "__main__":
    main()
