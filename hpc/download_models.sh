#!/usr/bin/env bash
# Pre-download the pinned checkpoints into $HF_HOME (inside the repo). Needs internet and Python, so
# run it on the transfer node (login nodes kill CPU-busy Python; compute nodes are offline).
#   ssh discovery-xfer 'bash /scratch/$USER/med-era/hpc/download_models.sh'      # all models
#   bash hpc/download_models.sh qwen35_27b                                         # one model
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=hpc/env.sh disable=SC1091
source hpc/env.sh
python -m medera.train.download --model "${1:-all}"
