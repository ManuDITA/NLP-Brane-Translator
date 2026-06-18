#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --job-name=brane-pipeline
#SBATCH --output=slurm_out/slurm-%j.out

echo "Job started on $(hostname) at $(date)"

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
mkdir -p slurm_out

# ---- useful HF/cache settings ----
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false

# Optional: only needed if you have a Hugging Face token
# export HF_TOKEN=your_token_here

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="runs/output_${TIMESTAMP}.txt"

# Make latest_output.txt point to this run's live log
ln -sfn "$(basename "$LOGFILE")" runs/latest_output.txt

echo "Logging to $LOGFILE"
echo "Latest log symlink: runs/latest_output.txt"

# ---- Remote execution setup -----------------------------------------------
# The file-based job queue lives at ~/brane_jobs/ on the Snellius filesystem.
# job_watcher.py on your local machine polls this via SSH — no port forwarding needed.
# Make sure job_watcher.py is running before submitting:
#   source .env && python scripts/remote_execution/local/job_watcher.py
export SNELLIUS_JOBS_DIR="${HOME}/brane_jobs"
mkdir -p "${SNELLIUS_JOBS_DIR}/pending" "${SNELLIUS_JOBS_DIR}/done"
echo "Job queue: ${SNELLIUS_JOBS_DIR}"
# ---------------------------------------------------------------------------

python -u src/pipeline.py \
    --model Qwen/Qwen3.6-27B \
    --temperature 0.2 \
    --execute \
    > "$LOGFILE" 2>&1

echo "Job finished at $(date)"