#!/bin/bash
#SBATCH --job-name=medera_smoke
#SBATCH --account=ostadabbas
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/%u/med-era/runs/logs/%x_%A_%a.out
# FIRST job. One array task per core arm: 0 qwen, 1 gemma_al, 2 gemma_ex (3-5 = the *_q4 switch arms,
# e.g. `sbatch --array=5 hpc/smoke.slurm`). Each task loads the pinned base model in bf16, trains
# `smoke_steps` real optimizer steps (grad_accum 8, 2048-token blocks) on ~20 HRT articles from
# data/smoke (hpc/prep_smoke.sh), runs the adapter audit, and reports s/step, tokens/s, peak memory,
# the fla fast-path flag and the token count that fits in the wall-time guard.
# Result: runs/results/smoke/<arm>.json; the log line starts with "[smoke] PASS" or "[smoke] FAIL".
set -euo pipefail
source "/scratch/$USER/med-era/hpc/env.sh"
cd "$MEDERA_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline   # compute nodes have no internet

ARMS=(qwen gemma_al gemma_ex qwen_q4 gemma_al_q4 gemma_ex_q4)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "[smoke] job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID arm=$ARM node=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -u -m medera.train.lora_cpt --arm "$ARM" --window w1 --data-root data/smoke --smoke
