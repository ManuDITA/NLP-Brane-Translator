#!/bin/bash
# sbatch_generate.sh — Generate a single BraneScript from a natural-language
# intent and enqueue it for local execution via job_watcher.py.
#
# Submitted automatically by the frontend "Generate" tab through server.py.
# Can also be submitted manually:
#
#   sbatch --export=ALL,\
#     INTENT="analyze diabetes risk for all patients in heal_pa_2",\
#     EVAL_MODEL="outputs/models/output_merged_qwen3.6-27b",\
#     REQ_ID="$(uuidgen)" \
#     sbatch_generate.sh
#
# Required --export vars:
#   INTENT       Natural-language intent to translate
#   EVAL_MODEL   Model path (relative to project dir) or HuggingFace ID
#   REQ_ID       Request UUID — used as the job ID in ~/brane_jobs/pending/
#
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --job-name=brane-generate
#SBATCH --output=outputs/slurm/generate-%j.out
#SBATCH --error=outputs/slurm/generate-%j.err

echo "Job started on $(hostname) at $(date)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"

echo "Project dir : ${PROJECT_DIR}"
echo "Intent      : ${INTENT}"
echo "Model       : ${EVAL_MODEL}"
echo "Req ID      : ${REQ_ID}"

# ── Validate required vars ────────────────────────────────────────────────────
if [[ -z "${INTENT}" || -z "${EVAL_MODEL}" || -z "${REQ_ID}" ]]; then
    echo "❌ INTENT, EVAL_MODEL, and REQ_ID must all be set via --export."
    exit 1
fi

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

# ── HuggingFace cache → scratch ───────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/scratch-shared/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"
echo "HF cache : ${HF_HOME}"

mkdir -p "${PROJECT_DIR}/outputs/slurm"

# ── GPU info ──────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# ── Generate BraneScript and enqueue for execution ───────────────────────────
echo ""
echo "🚀 Starting inference at $(date)"
echo "──────────────────────────────────────────────────────"

python "${PROJECT_DIR}/src/fine_tuning/generate_single.py" \
    --intent  "${INTENT}" \
    --model   "${EVAL_MODEL}" \
    --req-id  "${REQ_ID}"

echo ""
echo "✅ generate_single.py finished at $(date)"
echo "   job_watcher.py will execute the script and return results to the dashboard."
