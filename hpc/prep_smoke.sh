#!/usr/bin/env bash
# Fetch the smoke-test articles once: 20 real HRT articles per window into data/smoke/ (gitignored).
# Needs internet and Python, so run it on the transfer node (login nodes kill CPU-busy Python):
#   ssh discovery-xfer 'bash /scratch/$USER/med-era/hpc/prep_smoke.sh'
# Cleaning and packing run in the CPU job that submit_smoke.sh submits (hpc/build_data.slurm).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=hpc/env.sh disable=SC1091
source hpc/env.sh
for w in w1 w2 w3; do
  python -m medera.collect harvest --window "$w" --query core --limit 20 --resume --data-root data/smoke
done
echo "[prep_smoke] DONE: $(find data/smoke/raw -name '*.xml' | wc -l) XML files in data/smoke/raw"
