#!/usr/bin/env bash
# Upload the offline W&B runs that jobs wrote (compute nodes have no internet). Login node only,
# and only when the owner asks for it:
#   cd /scratch/$USER/med-era && bash hpc/wandb_sync.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
[ -n "${WANDB_API_KEY:-}" ] || { echo "[wandb_sync] no key in .env; nothing to do"; exit 1; }
[ -d "$WANDB_DIR/wandb" ] || { echo "[wandb_sync] no offline runs in $WANDB_DIR/wandb"; exit 0; }
cd "$WANDB_DIR" && wandb sync --sync-all
