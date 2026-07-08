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
#SBATCH --error=outputs/slurm/baseline-%j.err

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
# Override via environment variables before calling sbatch, e.g.:
#   EVAL_MODEL=Qwen/Qwen3-4B EVAL_LABEL="qwen3-4b (base)" sbatch sbatch_baseline.sh
#
# Qwen3 model IDs on HuggingFace:
#   Small  (4B)  → Qwen/Qwen3-4B
#   Medium (8B)  → Qwen/Qwen3-8B
#   Large  (32B) → Qwen/Qwen3-32B
export EVAL_MODEL="${EVAL_MODEL:-Qwen/Qwen3-4B}"
export EVAL_LABEL="${EVAL_LABEL:-${EVAL_MODEL##*/} (base)}"
# Full training suite (607 examples) for the baseline benchmark
export EVAL_TEST_FILE="${EVAL_TEST_FILE:-${PROJECT_DIR}/data/training/train.jsonl}"

echo "Model    : ${EVAL_MODEL}"
echo "Label    : ${EVAL_LABEL}"
echo "Test file: ${EVAL_TEST_FILE}"

mkdir -p "${PROJECT_DIR}/outputs/slurm" "${PROJECT_DIR}/outputs/eval"

# ── GPU info ──────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# ── Run baseline evaluation ───────────────────────────────────────────────────
echo ""
echo "🔍 Starting baseline evaluation at $(date)"
echo "──────────────────────────────────────────────────────"

python "${PROJECT_DIR}/src/fine_tuning/evaluate.py" \
    --model         "${EVAL_MODEL}" \
    --label         "${EVAL_LABEL}" \
    --test-file     "${EVAL_TEST_FILE}" \
    --generate-only \
    --resume

echo ""
echo "✅ Baseline evaluation finished at $(date)"
