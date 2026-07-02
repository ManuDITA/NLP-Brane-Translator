#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --job-name=brane-finetune
#SBATCH --output=logs/slurm_out/finetune-%j.out

echo "Job started on $(hostname) at $(date)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
umask 077

if [[ ! -w "${PROJECT_DIR}" ]]; then
    echo "❌ Project directory is not writable: ${PROJECT_DIR}"
    exit 1
fi

echo "Project dir: ${PROJECT_DIR}"

# ── Modules ──────────────────────────────────────────────────────────────────
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

# ── Install training deps if not already present ─────────────────────────────
python -c "import trl, peft, bitsandbytes, datasets" 2>/dev/null || {
    echo "📦 Installing training dependencies..."
    pip install -q trl peft accelerate bitsandbytes datasets
}

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

# Allow overriding the base model via environment variable
export FINETUNE_MODEL="${FINETUNE_MODEL:-Qwen/Qwen3-8B}"
echo "Base model: ${FINETUNE_MODEL}"

mkdir -p "${PROJECT_DIR}/logs/slurm_out"

# ── GPU info ──────────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# ── Run fine-tuning ───────────────────────────────────────────────────────────
# Training auto-resumes from the latest checkpoint in src/fine_tuning/output/.
# Pass --restart to ignore existing checkpoints and start fresh.
echo ""
echo "🏋️  Starting fine-tuning at $(date)"
echo "──────────────────────────────────────────────────────"

python "${PROJECT_DIR}/src/fine_tuning/train.py" ${EXTRA_ARGS:-}

echo ""
echo "✅ Fine-tuning finished at $(date)"
