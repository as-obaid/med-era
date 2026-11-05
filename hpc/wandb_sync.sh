#!/usr/bin/env bash
# Upload the offline W&B runs of the adapter trainings (compute nodes have no internet). Run it on a LOGIN node,
# and only when the owner asks for it:
#   ssh explorer 'bash /scratch/$USER/med-era/hpc/wandb_sync.sh'
# Not on discovery-xfer: it runs CentOS 7 (glibc 2.17), where the project env's numpy does not import.
# What it syncs: with no arguments, only the offline run dirs whose run id is listed in a
# runs/checkpoints/*/wandb_run_id file (the 9 adapters; all attempts of one adapter share that id and go up into one
# run, in start order). The Nov-4 smoke runs and any other offline dirs are left alone. Dirs already synced
# (run-<id>.wandb.synced) are skipped. Evaluation jobs do not log to W&B (hpc/eval.slurm sets WANDB_MODE=disabled).
#   bash hpc/wandb_sync.sh                         # the adapter runs not synced yet
#   bash hpc/wandb_sync.sh --dry-run               # list what would be synced; uploads nothing, needs no key
#   bash hpc/wandb_sync.sh <offline-run-dir> ...   # exactly these dirs (already synced ones are still skipped)
set -euo pipefail
DRY=0
if [ "${1:-}" = --dry-run ]; then DRY=1; shift; fi
explicit=()
for a in "$@"; do   # resolve explicit dirs against the caller's cwd before changing directory
  [ -d "$a" ] || { echo "[wandb_sync] not a directory: $a" >&2; exit 2; }
  explicit+=("$(cd "$a" && pwd)")
done
cd "$(dirname "$0")/.."
# shellcheck source=hpc/env.sh disable=SC1091
source hpc/env.sh
OFFLINE=$WANDB_DIR/wandb

run_id_of() { local b; b=$(basename "$1"); echo "${b#offline-run-*-}"; }   # offline-run-<date_time>-<id>
is_synced() { [ -f "$1/run-$(run_id_of "$1").wandb.synced" ]; }

targets=()
if [ "${#explicit[@]}" -gt 0 ]; then
  targets=("${explicit[@]}")
else
  ids=$(cat runs/checkpoints/*/wandb_run_id 2>/dev/null | sort -u || true)
  [ -n "$ids" ] || { echo "[wandb_sync] no runs/checkpoints/*/wandb_run_id files; nothing to sync"; exit 0; }
  for d in "$OFFLINE"/offline-run-*; do   # glob order = start-time order
    [ -d "$d" ] || continue
    if grep -qxF "$(run_id_of "$d")" <<< "$ids"; then targets+=("$d"); fi
  done
fi
todo=()
for d in ${targets[@]+"${targets[@]}"}; do
  if is_synced "$d"; then echo "[wandb_sync] already synced: $(basename "$d")"; else todo+=("$d"); fi
done
if [ "${#todo[@]}" -eq 0 ]; then echo "[wandb_sync] nothing to sync"; exit 0; fi
printf '[wandb_sync] to sync (%s): %s\n' "${#todo[@]}" "$(for d in "${todo[@]}"; do basename "$d"; done | paste -sd' ' -)"
if [ "$DRY" = 1 ]; then echo "[wandb_sync] dry run: nothing uploaded"; exit 0; fi
[ -n "${WANDB_API_KEY:-}" ] || { echo "[wandb_sync] no key in .env; nothing uploaded"; exit 1; }
wandb sync "${todo[@]}"
