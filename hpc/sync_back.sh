#!/usr/bin/env bash
# Run on the LAPTOP after every job: mirror the cluster's runs/ (results, adapter audits, smoke
# reports, trained adapters, logs) into this repo's runs/ (gitignored), so nothing is lost to the
# monthly /scratch purge. Checkpoints (resume state) are skipped; adapters are kept. Also pulls the evaluation
# outputs the eval jobs write into the cluster clone's results/ (answers/ and scores/: perplexity cells, probe
# answers); review and commit them from the laptop.
#   bash hpc/sync_back.sh
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${MEDERA_XFER_HOST:-discovery-xfer}
REMOTE_USER=${MEDERA_REMOTE_USER:-$(ssh -G "$HOST" | awk '$1 == "user" {print $2}')}
SRC="/scratch/$REMOTE_USER/med-era/runs/"
mkdir -p runs
rsync -avh --partial --exclude 'checkpoints/' "$HOST:$SRC" runs/
echo "[sync] $HOST:$SRC -> $(pwd)/runs/"
for d in answers scores; do
  mkdir -p "results/$d"
  rsync -avh --partial --exclude '*.tmp' "$HOST:/scratch/$REMOTE_USER/med-era/results/$d/" "results/$d/"
done
echo "[sync] $HOST:/scratch/$REMOTE_USER/med-era/results/{answers,scores}/ -> $(pwd)/results/"
