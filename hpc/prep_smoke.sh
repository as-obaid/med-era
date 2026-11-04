#!/usr/bin/env bash
# Build the smoke dataset once: 20 real HRT articles per window into data/smoke/ (gitignored), then
# merge, clean and pack for both models. Needs internet, so run it on the login node:
#   cd /scratch/$USER/med-era && bash hpc/prep_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
for w in w1 w2 w3; do
  python -m medera.collect harvest --window "$w" --query core --limit 20 --resume --data-root data/smoke
done
python -m medera.collect merge --data-root data/smoke
python -m medera.preprocess text --data-root data/smoke
for m in qwen35_27b gemma4_26b_a4b; do
  python -m medera.preprocess pack --model "$m" --data-root data/smoke
done
echo "[prep_smoke] DONE: data/smoke/processed/hrt/*/w1_train.npy"
