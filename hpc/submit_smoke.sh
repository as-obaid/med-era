#!/usr/bin/env bash
# The FIRST jobs. Run on the login node after hpc/prep_smoke.sh:
#   cd /scratch/$USER/med-era && bash hpc/submit_smoke.sh
# Submits (1) hpc/build_data.slurm data/smoke merge (CPU: pytest env check, merge, clean, pack) unless
# the smoke data is already packed, then (2) hpc/smoke.slurm, array 0-2 = qwen, gemma_al, gemma_ex
# (bf16, one H200 each, gpu-short), which starts only if (1) succeeded.
#   SMOKE_ARRAY=0-5 bash hpc/submit_smoke.sh     # also smoke the qlora4 switch arms
#   MEDERA_GPU=a100 bash hpc/submit_smoke.sh     # A100 80 GB fallback (off by default)
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
source hpc/_preflight.sh
preflight_env
preflight_models qwen35_27b gemma4_26b_a4b
compgen -G "data/smoke/raw/hrt/w1/*.xml" >/dev/null \
  || { echo "[submit] no smoke articles: ssh discovery-xfer 'bash $MEDERA_ROOT/hpc/prep_smoke.sh'"; exit 1; }
dep=()
if preflight_data data/smoke qwen35_27b gemma4_26b_a4b 2>/dev/null; then
  echo "[submit] smoke data already packed"
else
  build=$(sbatch --parsable hpc/build_data.slurm data/smoke merge)
  echo "[submit] build job $build (CPU)"
  dep=(--dependency="afterok:$build")
fi
# shellcheck disable=SC2046
sbatch "${dep[@]}" --array="${SMOKE_ARRAY:-0-2}" $(gpu_args) hpc/smoke.slurm
