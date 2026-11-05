#!/bin/bash
#SBATCH --job-name=medera_build
#SBATCH --account=ostadabbas
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/med-era/runs/logs/%x_%j.out
# CPU job on a compute node (RHEL 9, offline). Login nodes kill CPU-busy Python and the transfer node
# (CentOS 7) cannot load numpy or torch, so the env is checked and the data is built here:
#   1. pytest (env check on the compute-node OS, incl. the tiny Gemma-4 expert-LoRA recipe on CPU)
#   2. merge, only with the `merge` argument: smoke data, whose articles are not relevance-screened, so it
#      also passes --no-relevance-screen. The real corpus needs data/manifests/relevance_labels.csv.
#   3. preprocess text, then pack for both models (tokenizers from the local HF cache)
#   sbatch hpc/build_data.slurm data/smoke merge     # smoke data (submit_smoke.sh does this)
#   sbatch hpc/build_data.slurm data                 # real corpus, after the harvest
set -euo pipefail
ROOT=${1:-data}
source "/scratch/$USER/med-era/hpc/env.sh"
cd "$MEDERA_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[build] job=$SLURM_JOB_ID root=$ROOT node=$(hostname) commit=$(git rev-parse --short HEAD)"
python -m pytest -q -p no:cacheprovider tests
SCREEN=()
if [ "${2:-}" = merge ]; then python -m medera.collect merge --data-root "$ROOT"; SCREEN=(--no-relevance-screen); fi
python -m medera.preprocess text --data-root "$ROOT" "${SCREEN[@]}"
for m in qwen35_27b gemma4_26b_a4b; do
  python -m medera.preprocess pack --model "$m" --data-root "$ROOT"
done
echo "[build] DONE $ROOT"
