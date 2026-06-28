#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --job-name=brane-single-intent
#SBATCH --output=logs/slurm_out/slurm-%j.out

set -eo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 \"<single intent>\""
    echo "Example: sbatch $0 \"Analyze all patients in CSV and commit results\""
    exit 1
fi

INTENT="$*"

echo "Job started on $(hostname) at $(date)"
echo "Intent: ${INTENT}"

# ---- modules ----
module purge
module load 2025
module load CUDA/12.9.1
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

export PATH=$HOME/tools/mmseqs/bin:$PATH

# ---- activate venv ----
source ~/Thesis/NLP-Brane-Translator/.venv/bin/activate

# ---- folders ----
mkdir -p runs
mkdir -p logs/slurm_out

# ---- useful HF/cache settings ----
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="runs/output_single_${TIMESTAMP}.txt"

ln -sfn "$(basename "$LOGFILE")" runs/latest_output.txt

echo "Logging to $LOGFILE"
echo "Latest log symlink: runs/latest_output.txt"

# ---- Remote execution setup -----------------------------------------------
# The file-based job queue lives at ~/brane_jobs/ on the Snellius filesystem.
# job_watcher.py on your local machine polls this via SSH.
export SNELLIUS_JOBS_DIR="${HOME}/brane_jobs"
mkdir -p "${SNELLIUS_JOBS_DIR}/pending" "${SNELLIUS_JOBS_DIR}/done"
echo "Job queue: ${SNELLIUS_JOBS_DIR}"

# ---- Training data collection ----------------------------------------------
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export TRAINING_DATA_DIR="${PROJECT_DIR}/training_data"
mkdir -p "${TRAINING_DATA_DIR}"
echo "Training data: ${TRAINING_DATA_DIR}/index.jsonl"

python -u src/pipeline.py \
    --model Qwen/Qwen3.6-27B \
    --temperature 0.2 \
    --execute \
    --collect \
    --query "${INTENT}" \
    > "$LOGFILE" 2>&1

echo "Job finished at $(date)"
