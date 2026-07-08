#!/bin/bash
# submit_baselines.sh
#
# Submits one sbatch_baseline.sh job per model — they run in parallel on
# separate GPUs.  Edit the MODELS array to match your HuggingFace model IDs.
#
# Usage:
#   bash submit_baselines.sh
#
# After all jobs finish, copy outputs/eval/*_generated.json back locally and run:
#   python scripts/execute_generated.py outputs/eval/*_generated.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Model list ──────────────────────────────────────────────────────────────
# Format: "HuggingFace/model-id|human label"
MODELS=(
    "Qwen/Qwen3-4B|qwen3-4b (base)"
    "Qwen/Qwen3-8B|qwen3-8b (base)"
    "Qwen/Qwen3-32B|qwen3-32b (base)"
)

# ── Submit ───────────────────────────────────────────────────────────────────
echo "🚀 Submitting ${#MODELS[@]} baseline jobs..."
echo ""

for entry in "${MODELS[@]}"; do
    model="${entry%%|*}"
    label="${entry##*|}"
    job_id=$(
        EVAL_MODEL="${model}" \
        EVAL_LABEL="${label}" \
        sbatch --parsable "${SCRIPT_DIR}/sbatch_baseline.sh"
    )
    printf "  %-32s → job %s\n" "${model}" "${job_id}"
done

echo ""
echo "✅ All jobs submitted.  Track with: squeue -u \$USER"
echo ""
echo "When done, copy results and run:"
echo "  python scripts/execute_generated.py outputs/eval/*_generated.json"
