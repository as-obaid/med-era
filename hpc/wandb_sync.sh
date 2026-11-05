#!/usr/bin/env bash
# Upload the offline W&B runs that jobs wrote (compute nodes have no internet). Run it on the
# transfer node (internet; login nodes kill CPU-busy Python), and only when the owner asks for it:
#   ssh discovery-xfer 'bash /scratch/$USER/med-era/hpc/wandb_sync.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=hpc/env.sh disable=SC1091
source hpc/env.sh
[ -n "${WANDB_API_KEY:-}" ] || { echo "[wandb_sync] no key in .env; nothing to do"; exit 1; }
[ -d "$WANDB_DIR/wandb" ] || { echo "[wandb_sync] no offline runs in $WANDB_DIR/wandb"; exit 0; }
# wandb 0.31 removed --sync-all: with no paths it syncs every not-yet-synced run under $WANDB_DIR/wandb. The
# offline files of one adapter's attempts share a run id, so they go up into one run, in start order.
cd "$WANDB_DIR" && wandb sync --yes
