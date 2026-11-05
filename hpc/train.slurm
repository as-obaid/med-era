#!/bin/bash
#SBATCH --job-name=medera_train
#SBATCH --account=ostadabbas
#SBATCH --partition=sharing,gpu-short,gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/med-era/runs/logs/%x_%j.out
# One adapter (arm x window): `epochs` passes over the equal-token budget, then the adapter audit.
# Normally submitted by hpc/dispatch.sh, which adds --exclude=<GPU nodes under 80 GB> (hpc/_dispatch_lib.sh):
#   sbatch --exclude=... hpc/train.slurm <arm> <window>
# The job starts on whichever allowed GPU frees first (H100 in sharing, H200 in gpu-short/gpu-interactive or
# sharing, A100 80 GB). It checkpoints every save_every_steps and right before the job's end time
# (SLURM_JOB_END_TIME minus wall_margin_s), then exits 3; the next attempt resumes from
# runs/checkpoints/<arm>_<window>/. Exit 4 = landed on a GPU under 79 GiB (the dispatcher resubmits).
set -euo pipefail
ARM=${1:?usage: sbatch hpc/train.slurm <arm> [window]}
WINDOWS=(w1 w2 w3)
WINDOW=${2:-${WINDOWS[${SLURM_ARRAY_TASK_ID:?pass a window or use --array=0-2}]}}
source "/scratch/$USER/med-era/hpc/env.sh"
cd "$MEDERA_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline   # compute nodes have no internet

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | tr -d ' ')
echo "[train] job=$SLURM_JOB_ID attempt=${MEDERA_ATTEMPT:-?} arm=$ARM window=$WINDOW node=$(hostname)" \
     "partition=${SLURM_JOB_PARTITION:-?} gpu=$GPU mem=${MEM_MIB}MiB commit=$(git rev-parse --short HEAD)"
if [ "${MEM_MIB%%.*}" -lt $((79 * 1024)) ]; then
  echo "[train] ERROR: $GPU has ${MEM_MIB} MiB (< 79 GiB): bf16 LoRA needs an 80 GB card. Exit 4 so the" \
       "dispatcher resubmits it elsewhere."
  exit 4
fi
# Slurm 23.11 exports SLURM_JOB_END_TIME; fall back to squeue's end time.
export MEDERA_JOB_END_EPOCH=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
python -u -m medera.train.lora_cpt --arm "$ARM" --window "$WINDOW"
