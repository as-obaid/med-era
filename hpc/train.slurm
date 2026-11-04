#!/bin/bash
#SBATCH --job-name=medera_train
#SBATCH --account=ostadabbas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/med-era/runs/logs/%x_%j.out
# One adapter (arm x window), one pass at the equal-token budget, then the adapter audit.
#   sbatch hpc/train.slurm <arm> <window>        # what submit_all.sh does, capped to the QOS
#   sbatch --array=0-2 hpc/train.slurm <arm>      # per-arm array: task 0,1,2 -> w1,w2,w3
# The run checkpoints every save_every_steps and stops at wall_guard_hours (exit 3); resubmitting
# the same arm/window resumes from runs/checkpoints/<arm>_<window>/. Logs s/step and an ETA.
set -euo pipefail
ARM=${1:?usage: sbatch hpc/train.slurm <arm> [window]}
WINDOWS=(w1 w2 w3)
WINDOW=${2:-${WINDOWS[${SLURM_ARRAY_TASK_ID:?pass a window or use --array=0-2}]}}
source "/scratch/$USER/med-era/hpc/env.sh"
cd "$MEDERA_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline   # compute nodes have no internet

echo "[train] job=$SLURM_JOB_ID arm=$ARM window=$WINDOW node=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -u -m medera.train.lora_cpt --arm "$ARM" --window "$WINDOW"
