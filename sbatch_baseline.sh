#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=brane-baseline
#SBATCH --output=outputs/slurm/baseline-%j.out

echo "Job started on $(hostname) at $(date)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"

echo "Project dir: ${PROJECT_DIR}"

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load 2025
module load CUDA/12.9.1

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
    echo "✅ Activated .venv"
else
    echo "❌ .venv not found at ${PROJECT_DIR}/.venv"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

# ── Model configuration ───────────────────────────────────────────────────────
# Override via: EVAL_MODEL=Qwen/Qwen3-4B sbatch sbatch_baseline.sh
#
# Placeholder model IDs — fill in with the correct HuggingFace repo names:
#   SMALL  (4B)  → e.g. Qwen/Qwen3-4B
#   MEDIUM (9B)  → e.g. Qwen/Qwen3-8B
#   LARGE  (27B) → e.g. Qwen/Qwen3-32B
export EVAL_MODEL="${EVAL_MODEL:-PLACEHOLDER/model-name-4b}"
export EVAL_LABEL="${EVAL_LABEL:-${EVAL_MODEL##*/} (base)}"

echo "Model  : ${EVAL_MODEL}"
echo "Label  : ${EVAL_LABEL}"

mkdir -p "${PROJECT_DIR}/outputs/slurm" "${PROJECT_DIR}/outputs/eval"

# ── GPU info ──────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# ── Run baseline evaluation ───────────────────────────────────────────────────
echo ""
echo "🔍 Starting baseline evaluation at $(date)"
echo "──────────────────────────────────────────────────────"

python "${PROJECT_DIR}/src/fine_tuning/evaluate.py" \
    --model  "${EVAL_MODEL}" \
    --label  "${EVAL_LABEL}"

echo ""
echo "✅ Baseline evaluation finished at $(date)"
