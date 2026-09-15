#!/bin/bash
#SBATCH --job-name=escape-sps
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=0:30:00
#SBATCH --output=/home/knguyen2/madrona_escape_room/logs/escape_%j.out
#SBATCH --error=/home/knguyen2/madrona_escape_room/logs/escape_%j.err

set -euo pipefail

REPO_ROOT="/home/knguyen2/madrona_escape_room"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

# Memory-safe short-training defaults for an A100-SXM4-40GB.  The full mjwarp
# scene is much heavier per environment than the kinematic benchmark.
NUM_ENVS="${NUM_ENVS:-512}"
NUM_UPDATES="${NUM_UPDATES:-30}"
STEPS_PER_UPDATE="${STEPS_PER_UPDATE:-16}"

echo "Start: $(date)"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "JobId: ${SLURM_JOB_ID:-unknown}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "NUM_ENVS=$NUM_ENVS NUM_UPDATES=$NUM_UPDATES STEPS_PER_UPDATE=$STEPS_PER_UPDATE"

module purge
module load 2024
module load CUDA/12.6.0

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.torch}"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/ckpts" "$UV_CACHE_DIR" "$TORCH_HOME"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH; expected at ~/.local/bin/uv"
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv
uv --version

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Creating venv at $VENV_DIR"
  uv venv --python 3.11 "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Python: $(which python) ($(python --version 2>&1))"

# CUDA torch first (Snellius A100), then project deps.
uv pip install --upgrade pip setuptools wheel
uv pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1

# Install the primary mjlab + MuJoCo Warp + rsl-rl runtime.
uv pip install -e .

echo "---- nvidia-smi ----"
nvidia-smi || true
echo "---- torch cuda ----"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

TRAIN_ARGS=(
  --num-envs "$NUM_ENVS"
  --num-updates "$NUM_UPDATES"
  --steps-per-update "$STEPS_PER_UPDATE"
  --device cuda:0
  --ckpt-dir "$REPO_ROOT/ckpts/job_${SLURM_JOB_ID:-local}"
  --save-interval 10
)

echo "Launching: python scripts/train.py ${TRAIN_ARGS[*]}"
python scripts/train.py "${TRAIN_ARGS[@]}"

echo "Done: $(date)"
