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
# Normally submitted by hpc/dispatch.sh, which adds --exclude=<every non-H100/H200 GPU node> (hpc/_dispatch_lib.sh):
#   sbatch --exclude=... hpc/train.slurm <arm> <window>     (`bash hpc/dispatch.sh --dry-run` prints the exact call)
# The job starts on whichever allowed GPU frees first (H100 in sharing, H200 in gpu-short/gpu-interactive or
# sharing). It checkpoints every save_every_steps and right before the job's end time
# (SLURM_JOB_END_TIME minus wall_margin_s), or on the SIGTERM Slurm sends at the limit, then exits 3; the next
# attempt resumes exactly from runs/checkpoints/<arm>_<window>/. Exit 4 = landed on a GPU under 79 GB (bf16 arms;
# the dispatcher resubmits and excludes that node).
set -euo pipefail
ARM=${1:?usage: sbatch hpc/train.slurm <arm> [window]}
WINDOWS=(w1 w2 w3)
WINDOW=${2:-${WINDOWS[${SLURM_ARRAY_TASK_ID:?pass a window or use --array=0-2}]}}
# shellcheck source=hpc/env.sh disable=SC1091
source "/scratch/$USER/med-era/hpc/env.sh"
cd "$MEDERA_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline   # compute nodes have no internet

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | tr -d ' ')
echo "[train] job=$SLURM_JOB_ID attempt=${MEDERA_ATTEMPT:-?} arm=$ARM window=$WINDOW node=$(hostname)" \
     "partition=${SLURM_JOB_PARTITION:-?} gpu=$GPU mem=${MEM_MIB}MiB commit=$(git rev-parse --short HEAD)"
MIN_MIB=$((79 * 1000 * 1000 * 1000 / 1048576))   # 79 GB; every 80 GB card reports >= 81,000 MiB
case $ARM in
  *_q4) ;;   # the qlora4 switch arms fit 40 GB cards
  *) if [ "${MEM_MIB%%.*}" -lt "$MIN_MIB" ]; then
       echo "[train] ERROR: $GPU has ${MEM_MIB} MiB (< 79 GB): bf16 LoRA needs an 80 GB card. Exit 4 so the" \
            "dispatcher resubmits it elsewhere."
       exit 4
     fi ;;
esac
# Slurm 23.11 exports SLURM_JOB_END_TIME; fall back to squeue's end time.
export MEDERA_JOB_END_EPOCH=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
# exec: Python becomes the batch process, so Slurm's SIGTERM at the time limit reaches its checkpoint handler.
exec python -u -m medera.train.lora_cpt --arm "$ARM" --window "$WINDOW"
