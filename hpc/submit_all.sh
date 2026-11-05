#!/usr/bin/env bash
# SUPERSEDED for the real launch by hpc/dispatch.sh (1-h jobs across sharing/gpu-short/gpu-interactive within
# the per-partition MaxSubmit limits). Kept for the 8-h `gpu` partition path.
# Submit the core training runs (3 arms x 3 windows = 9 adapters) within the gpu QOS limits:
# at most MEDERA_MAX_SUBMITTED (8) gpu jobs submitted per user; the QOS itself caps running jobs at 4.
# Idempotent: skips finished runs (adapter saved) and runs already queued, so rerun it to top up the
# queue and to resume runs that the wall-time guard stopped (they continue from their checkpoint).
#   bash hpc/submit_all.sh
#   MEDERA_ARMS="qwen_q4 gemma_al_q4 gemma_ex_q4" bash hpc/submit_all.sh   # qlora4 switch arms (off by default)
#   MEDERA_GPU=a100 bash hpc/submit_all.sh                                 # A100 80 GB fallback (off by default)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=hpc/env.sh disable=SC1091
source hpc/env.sh
# shellcheck source=hpc/_preflight.sh disable=SC1091
source hpc/_preflight.sh
read -r -a ARMS <<< "${MEDERA_ARMS:-qwen gemma_al gemma_ex}"
WINDOWS=(w1 w2 w3)
MAX_SUBMITTED=${MEDERA_MAX_SUBMITTED:-8}

models=$(for a in "${ARMS[@]}"; do arm_model "$a"; done | sort -u)
preflight_env
# shellcheck disable=SC2086
preflight_models $models
# shellcheck disable=SC2086
preflight_data data $models
GPU_ARGS=$(gpu_args)

queued=$(squeue -u "$USER" -h -o "%j")
count=$(squeue -u "$USER" -h -r -p gpu | wc -l)
for arm in "${ARMS[@]}"; do
  for w in "${WINDOWS[@]}"; do
    name="medera_${arm}_${w}"
    if [ -f "runs/adapters/${arm}_${w}/adapter_config.json" ]; then echo "[submit] $name done, skipping"; continue; fi
    if grep -qx "$name" <<< "$queued"; then echo "[submit] $name already queued"; continue; fi
    if (( count >= MAX_SUBMITTED )); then echo "[submit] cap of $MAX_SUBMITTED submitted gpu jobs reached; rerun later"; exit 0; fi
    # shellcheck disable=SC2086
    sbatch --job-name="$name" $GPU_ARGS hpc/train.slurm "$arm" "$w"
    count=$((count + 1))
  done
done
