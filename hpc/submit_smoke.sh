#!/usr/bin/env bash
# Preflight, then submit the FIRST job (hpc/smoke.slurm, array 0-2 = qwen, gemma_al, gemma_ex).
#   cd /scratch/$USER/med-era && bash hpc/submit_smoke.sh
#   SMOKE_ARRAY=0-5 bash hpc/submit_smoke.sh        # also smoke the qlora4 switch arms
#   MEDERA_GPU=a100 bash hpc/submit_smoke.sh        # A100 80 GB fallback (gpu-short has A100s too)
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
source hpc/_preflight.sh
preflight data/smoke qwen35_27b gemma4_26b_a4b
# shellcheck disable=SC2046
sbatch --array="${SMOKE_ARRAY:-0-2}" $(gpu_args) hpc/smoke.slurm
