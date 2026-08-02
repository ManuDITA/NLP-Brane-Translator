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
    "Qwen/Qwen3.5-4B|qwen3.5-4b (base)"
    "Qwen/Qwen3.5-9B|qwen3.5-9b (base)"
    "Qwen/Qwen3.6-27B|qwen3.6-27b (base)"
)

# ── Submit ───────────────────────────────────────────────────────────────────
echo "🚀 Submitting ${#MODELS[@]} baseline jobs..."
echo ""

for entry in "${MODELS[@]}"; do
    model="${entry%%|*}"
    label="${entry##*|}"
    test_file="${EVAL_TEST_FILE:-${SCRIPT_DIR}/data/training/train.jsonl}"
    job_id=$(
        sbatch --parsable \
            --export="EVAL_MODEL=${model},EVAL_LABEL=${label},EVAL_TEST_FILE=${test_file}" \
            "${SCRIPT_DIR}/sbatch_baseline.sh"
    )
    printf "  %-32s → job %s\n" "${model}" "${job_id}"
done

echo ""
echo "✅ All jobs submitted.  Track with: squeue -u \$USER"
echo ""
echo "When all jobs finish, run this on your LOCAL machine to fetch and process results:"
echo "  bash fetch_baselines.sh"
echo ""
echo "Or manually:"
echo "  scp ${SNELLIUS_USER:-\$SNELLIUS_USER}@${SNELLIUS_HOST:-snellius.surf.nl}:${SNELLIUS_PROJECT_DIR:-~/NLP-Brane-Translator}/outputs/eval/*_generated.json outputs/snellius/"
echo "  python scripts/process_snellius.py"
