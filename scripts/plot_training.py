"""
plot_training.py

Plot train and validation loss curves from training_metrics.jsonl files.

Each file is produced automatically during training (src/fine_tuning/train.py).
Multiple runs in the same file are plotted as separate series.

Usage
─────
  # Plot all runs for a specific model
  python scripts/plot_training.py --model qwen3.6-27b

  # Plot a specific metrics file
  python scripts/plot_training.py --file outputs/metrics/qwen3.6-27b_training_metrics.jsonl

  # Save figure instead of showing it interactively
  python scripts/plot_training.py --model qwen3.6-27b --save

Output
──────
  Shows an interactive matplotlib window, or saves to
  outputs/metrics/<model>_loss_curves.png when --save is used.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "outputs" / "metrics"


def load_metrics(path: Path) -> dict[str, list[dict]]:
    """Load a JSONL metrics file and group entries by run_id."""
    runs: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                run_id = entry.get("run_id", "unknown")
                runs[run_id].append(entry)
            except json.JSONDecodeError:
                continue
    return dict(runs)


def plot(runs: dict[str, list[dict]], title: str, save_path: Path | None = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("❌ matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (run_id, entries) in enumerate(sorted(runs.items())):
        color = colors[i % len(colors)]
        mode  = entries[0].get("mode", "sft")
        label = run_id

        # Train loss — logged at every LOGGING_STEPS
        train_steps = [e["step"] for e in entries if "loss" in e]
        train_loss  = [e["loss"] for e in entries if "loss" in e]

        # Eval loss — logged at every SAVE_STEPS (eval_loss key)
        eval_steps  = [e["step"] for e in entries if "eval_loss" in e]
        eval_loss   = [e["eval_loss"] for e in entries if "eval_loss" in e]

        # GRPO reward mean (if present)
        reward_steps = [e["step"] for e in entries if "rewards/mean" in e]
        reward_vals  = [e["rewards/mean"] for e in entries if "rewards/mean" in e]

        if train_steps:
            axes[0].plot(train_steps, train_loss, color=color,
                         linewidth=1.2, alpha=0.85, label=f"{label} (train)")
        if eval_steps:
            axes[0].plot(eval_steps, eval_loss, color=color,
                         linewidth=2.0, linestyle="--", marker="o", markersize=4,
                         alpha=0.9, label=f"{label} (val)")
        if reward_steps:
            axes[1].plot(reward_steps, reward_vals, color=color,
                         linewidth=1.5, label=f"{label} (reward mean)")

    # ── Left panel: loss ──────────────────────────────────────────────────────
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Right panel: reward (GRPO) or empty note ─────────────────────────────
    has_reward = any("rewards/mean" in e for run in runs.values() for e in run)
    if has_reward:
        axes[1].set_title("GRPO Reward (mean per step)")
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Mean reward")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color="gray", linewidth=0.8, linestyle=":")
    else:
        axes[1].set_title("GRPO Reward (not available for SFT)")
        axes[1].text(0.5, 0.5, "Run GRPO training to populate this panel",
                     ha="center", va="center", transform=axes[1].transAxes,
                     color="gray", fontsize=10)
        axes[1].set_axis_off()

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved → {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model slug, e.g. qwen3.6-27b")
    group.add_argument("--file",  help="Direct path to a _training_metrics.jsonl file")
    parser.add_argument("--save", action="store_true",
                        help="Save figure to PNG instead of showing interactively")
    args = parser.parse_args()

    if args.file:
        metrics_file = Path(args.file)
        title = metrics_file.stem.replace("_training_metrics", "").replace("_", " ").title()
    else:
        slug = args.model.lower().replace("/", "_")
        metrics_file = METRICS_DIR / f"{slug}_training_metrics.jsonl"
        title = f"{args.model} — Training Loss"

    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        print("   Training automatically creates this file. Run training first.")
        sys.exit(1)

    runs = load_metrics(metrics_file)
    if not runs:
        print("❌ No valid metric entries found in the file.")
        sys.exit(1)

    total = sum(len(v) for v in runs.values())
    print(f"📊 Loaded {total} log entries across {len(runs)} run(s)")
    for run_id, entries in sorted(runs.items()):
        train_n = sum(1 for e in entries if "loss" in e)
        eval_n  = sum(1 for e in entries if "eval_loss" in e)
        print(f"   {run_id}: {train_n} train points, {eval_n} eval points")

    save_path = None
    if args.save:
        stem = metrics_file.stem.replace("_training_metrics", "")
        save_path = METRICS_DIR / f"{stem}_loss_curves.png"

    plot(runs, title, save_path)


if __name__ == "__main__":
    main()
