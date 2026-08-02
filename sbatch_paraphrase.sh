#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --job-name=brane-paraphrase
#SBATCH --output=outputs/slurm/paraphrase-%j.out
#SBATCH --error=outputs/slurm/paraphrase-%j.err

echo "Job started on $(hostname) at $(date)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"

module purge
module load 2025
module load CUDA/12.9.1

source "${PROJECT_DIR}/.venv/bin/activate"
echo "✅ Activated .venv"

export HF_HOME="${HF_HOME:-/scratch-shared/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

# ── Config ────────────────────────────────────────────────────────────────────
# Override with env vars:
#   PARAPHRASE_MODEL=Qwen/Qwen3.5-9B PARAPHRASE_N=3 sbatch sbatch_paraphrase.sh
export PARAPHRASE_MODEL="${PARAPHRASE_MODEL:-Qwen/Qwen3.5-4B}"
export PARAPHRASE_N="${PARAPHRASE_N:-3}"

echo "Model  : ${PARAPHRASE_MODEL}"
echo "N/intent: ${PARAPHRASE_N}"
echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
echo ""

mkdir -p "${PROJECT_DIR}/outputs/slurm"

python "${PROJECT_DIR}/scripts/generate_paraphrases.py" \
    --model "${PARAPHRASE_MODEL}" \
    --n     "${PARAPHRASE_N}"     \
    --resume

echo ""
echo "✅ Paraphrase job finished at $(date)"
echo "Output: ${PROJECT_DIR}/data/training/paraphrases.jsonl"
