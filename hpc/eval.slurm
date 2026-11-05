#!/bin/bash
#SBATCH --job-name=medera_eval
#SBATCH --account=ostadabbas
#SBATCH --partition=sharing,gpu-short,gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/med-era/runs/logs/%x_%j.out
# One evaluation unit on one GPU: <phase> <variant>, where phase is
#   ppl    held-out perplexity of the variant on all three eras (medera.measure.perplexity; T3 manipulation check)
#   probe  the probe battery on the variant (medera.probe.generate; refuses adapters without the prereg tag)
# and variant is one of qwen_base gemma_base <arm>_<w1|w2|w3> (arm: qwen gemma_al gemma_ex).
# Normally submitted by `bash hpc/dispatch.sh --phase <phase>`, which adds --exclude=<every non-H100/H200 node>:
#   sbatch --exclude=... hpc/eval.slurm ppl gemma_ex_w2      (`bash hpc/dispatch.sh --phase ppl --dry-run` prints it)
# Like training: the job stops before its end time (SLURM_JOB_END_TIME minus wall_margin_s) or on SIGTERM after the
# current cell/chunk and exits 3; the next attempt resumes from the cells/chunks already written. Exit 4 = landed on
# a GPU under 79 GB (the dispatcher resubmits and excludes that node); exit 2 = refused (never resubmitted).
set -euo pipefail
PHASE=${1:?usage: sbatch hpc/eval.slurm <ppl|probe> <variant>}
VARIANT=${2:?usage: sbatch hpc/eval.slurm <ppl|probe> <variant>}
case $PHASE in
  ppl) MODULE=medera.measure.perplexity ;;
  probe) MODULE=medera.probe.generate ;;
  *) echo "[eval] unknown phase $PHASE (ppl | probe)"; exit 2 ;;
esac
# shellcheck source=hpc/env.sh disable=SC1091
source "/scratch/$USER/med-era/hpc/env.sh"
cd "$MEDERA_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled   # compute nodes have no internet; no tracking
export CUBLAS_WORKSPACE_CONFIG=:4096:8                               # deterministic cuBLAS reductions

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | tr -d ' ')
echo "[eval] job=$SLURM_JOB_ID attempt=${MEDERA_ATTEMPT:-?} phase=$PHASE variant=$VARIANT node=$(hostname)" \
     "partition=${SLURM_JOB_PARTITION:-?} gpu=$GPU mem=${MEM_MIB}MiB commit=$(git rev-parse --short HEAD)"
MIN_MIB=$((79 * 1000 * 1000 * 1000 / 1048576))   # 79 GB: the bf16 bases are 52-54 GB
if [ "${MEM_MIB%%.*}" -lt "$MIN_MIB" ]; then
  echo "[eval] ERROR: $GPU has ${MEM_MIB} MiB (< 79 GB). Exit 4 so the dispatcher resubmits it elsewhere."
  exit 4
fi
export MEDERA_JOB_END_EPOCH=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
# exec: Python becomes the batch process, so Slurm's SIGTERM reaches its stop handler.
exec python -u -m "$MODULE" --variant "$VARIANT"
