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

# ---- Remote execution tunnel setup ----------------------------------------
# Set up port forwarding from this compute node through the Snellius login node
# to the local machine's brane_executor.py (port 9753).
#
# Requires:
#   - brane_executor.py running on the local machine
#   - start_tunnel.sh running on the local machine (reverse SSH tunnel active)
#   - BRANE_EXECUTOR_TOKEN exported in your Snellius ~/.bashrc
#
# If the tunnel cannot be established, the pipeline still runs but skips execution.
source ~/Thesis/NLP-Brane-Translator/scripts/remote_execution/snellius/setup_compute_tunnel.sh

echo "Executor URL: ${BRANE_EXECUTOR_URL:-not set — execution disabled}"
# ---------------------------------------------------------------------------

python -u src/pipeline.py \
    --model Qwen/Qwen3.6-27B \
    --temperature 0.2 \
    --execute \
    > "$LOGFILE" 2>&1

echo "Job finished at $(date)"