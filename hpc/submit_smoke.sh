#!/usr/bin/env bash
# The FIRST jobs. Run on the login node:
#   cd /scratch/$USER/med-era && SMOKE_ROOT=data bash hpc/submit_smoke.sh   # smoke on the real corpus (w1)
#   cd /scratch/$USER/med-era && bash hpc/submit_smoke.sh                   # or on data/smoke (hpc/prep_smoke.sh)
# Submits (1) hpc/build_data.slurm (CPU: pytest env check, then clean and pack; for data/smoke also merge)
# unless the data is already packed, then (2) hpc/smoke.slurm, array 0-2 = qwen, gemma_al, gemma_ex
# (bf16, one H200 each, gpu-short), which starts only if (1) succeeded.
#   SMOKE_ARRAY=0-5 bash hpc/submit_smoke.sh     # also smoke the qlora4 switch arms
#   MEDERA_GPU=a100 bash hpc/submit_smoke.sh     # A100 80 GB fallback (off by default)
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
source hpc/_preflight.sh
ROOT=${SMOKE_ROOT:-data/smoke}
preflight_env
preflight_models qwen35_27b gemma4_26b_a4b
if [ "$ROOT" = data/smoke ]; then
  compgen -G "data/smoke/raw/hrt/w1/*.xml" >/dev/null \
    || { echo "[submit] no smoke articles: ssh discovery-xfer 'bash $MEDERA_ROOT/hpc/prep_smoke.sh'"; exit 1; }
  BUILD=(data/smoke merge)
else
  compgen -G "$ROOT/raw/hrt/w1/*.xml" >/dev/null \
    || { echo "[submit] no XML under $ROOT/raw/hrt: rsync the corpus from the laptop first"; exit 1; }
  BUILD=("$ROOT")
fi
dep=()
if preflight_data "$ROOT" qwen35_27b gemma4_26b_a4b 2>/dev/null; then
  echo "[submit] $ROOT already packed"
else
  build=$(sbatch --parsable hpc/build_data.slurm "${BUILD[@]}")
  echo "[submit] build job $build (CPU, short partition) for $ROOT"
  dep=(--dependency="afterok:$build")
fi
# shellcheck disable=SC2046
sbatch "${dep[@]}" --export=ALL,SMOKE_ROOT="$ROOT" --array="${SMOKE_ARRAY:-0-2}" $(gpu_args) hpc/smoke.slurm
