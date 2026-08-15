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

# ── HuggingFace cache → scratch (avoids home dir quota exhaustion) ────────────
# Pre-download models here before submitting jobs:
#   export HF_HOME=/scratch-shared/$USER/hf_cache
#   python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('<model>')"
export HF_HOME="${HF_HOME:-/scratch-shared/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"
echo "HF cache : ${HF_HOME}"

# ── Model configuration ───────────────────────────────────────────────────────
# Set EVAL_MODEL before calling sbatch, e.g.:
#   export EVAL_MODEL=Qwen/Qwen3-4B && sbatch sbatch_baseline.sh
#   export EVAL_MODEL=outputs/models/qwen3.5-9b && sbatch sbatch_baseline.sh
#   export EVAL_MODEL=outputs/models/qwen3.5-9b-ep5-r32 && sbatch sbatch_baseline.sh
#   sbatch --export="ALL,EVAL_MODEL=outputs/models/qwen3.5-9b-ep5-r32" sbatch_baseline.sh
#
# The label is derived automatically from the model path:
#   Qwen/Qwen3-4B                           → Qwen3-4B (base)
#   outputs/models/qwen3.5-9b → Qwen3.5.9B (SFT)
#   (override with EVAL_LABEL if needed)
#
# Qwen3 model IDs on HuggingFace:
#   Small  (4B)  → Qwen/Qwen3-4B
#   Medium (8B)  → Qwen/Qwen3-8B
#   Large  (32B) → Qwen/Qwen3-32B
export EVAL_MODEL="${EVAL_MODEL:?ERROR: EVAL_MODEL not set. Example: EVAL_MODEL=Qwen/Qwen3-4B sbatch sbatch_baseline.sh}"
export EVAL_LABEL="${EVAL_LABEL:-}"
export EVAL_TEST_FILE="${EVAL_TEST_FILE:-${PROJECT_DIR}/data/training/train.jsonl}"
export EVAL_SKIP_DECOMPOSITION="${EVAL_SKIP_DECOMPOSITION:-0}"

echo "Model    : ${EVAL_MODEL}"
echo "Label    : ${EVAL_LABEL:-<auto>}"
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
    ${EVAL_LABEL:+--label "${EVAL_LABEL}"} \
    --test-file     "${EVAL_TEST_FILE}" \
    --generate-only \
    --resume \
    $([[ "${EVAL_SKIP_DECOMPOSITION}" == "1" ]] && echo "--skip-decomposition")

echo ""
echo "✅ Baseline evaluation finished at $(date)"
