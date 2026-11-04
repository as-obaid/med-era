#!/usr/bin/env bash
# Pre-download the pinned checkpoints into $HF_HOME (inside the repo). Needs internet: run it on the
# transfer node (preferred for bulk) or the login node, never in a job.
#   ssh discovery-xfer 'bash /scratch/$USER/med-era/hpc/download_models.sh'      # all models
#   bash hpc/download_models.sh qwen35_27b                                         # one model
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
python -m medera.train.download --model "${1:-all}"
