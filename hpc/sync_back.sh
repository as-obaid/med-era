#!/usr/bin/env bash
# Run on the LAPTOP after every job: mirror the cluster's runs/ (results, adapter audits, smoke
# reports, trained adapters, logs) into this repo's runs/ (gitignored), so nothing is lost to the
# monthly /scratch purge. Checkpoints (resume state) are skipped; adapters are kept.
#   bash hpc/sync_back.sh
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${MEDERA_XFER_HOST:-discovery-xfer}
REMOTE_USER=${MEDERA_REMOTE_USER:-$(ssh -G "$HOST" | awk '$1 == "user" {print $2}')}
SRC="/scratch/$REMOTE_USER/med-era/runs/"
mkdir -p runs
rsync -avh --partial --exclude 'checkpoints/' "$HOST:$SRC" runs/
echo "[sync] $HOST:$SRC -> $(pwd)/runs/"
