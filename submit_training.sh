#!/bin/bash
# submit_training.sh
#
# Submits SFT fine-tuning jobs for all configured models, then queues
# a merge job for each that depends on the training job finishing.
#
# Usage:
#   bash submit_training.sh
#
# Edit the RUNS array to add/remove models or change hyperparams.
# Each entry is a comma-separated list of KEY=VALUE pairs passed via --export.
# Any omitted hyperparameter uses the default from train.py.
#
# After all merges complete, evaluate with:
#   bash submit_baselines.sh   (after editing it to list the merged model paths)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Runs to submit ────────────────────────────────────────────────────────────
# Add one line per training run. Use the same env var names as train.py.
RUNS=(
    "FINETUNE_MODEL=Qwen/Qwen3.5-9B"
    "FINETUNE_MODEL=Qwen/Qwen3.6-27B"
)

echo "Submitting ${#RUNS[@]} training job(s)..."
echo ""

for RUN_VARS in "${RUNS[@]}"; do
    # Submit training
    TRAIN_JOB=$(sbatch --parsable \
        --export="ALL,TRAIN_MODE=sft,${RUN_VARS}" \
        "${PROJECT_DIR}/sbatch_finetune.sh")
    echo "✅ Training job ${TRAIN_JOB} submitted  [${RUN_VARS}]"

    # Submit merge — depends on training finishing successfully
    MERGE_JOB=$(sbatch --parsable \
        --dependency="afterok:${TRAIN_JOB}" \
        --partition=gpu_h100 \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --gpus=1 --mem=60G --time=00:30:00 \
        --job-name="brane-merge" \
        --output="${PROJECT_DIR}/outputs/slurm/merge-%j.out" \
        --export="ALL,${RUN_VARS}" \
        --wrap="cd '${PROJECT_DIR}' && source .venv/bin/activate && python src/fine_tuning/train.py --merge")
    echo "   └─ Merge job  ${MERGE_JOB} queued    (depends on ${TRAIN_JOB})"
    echo ""
done

echo "Done. Check queue with: squeue -u \$USER"
echo "Slurm logs: outputs/slurm/"
