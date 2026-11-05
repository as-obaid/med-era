#!/usr/bin/env bash
# One table for the training dispatcher: adapter | attempts | state | job | GPU | steps (login node, no Python).
#   cd /scratch/$USER/med-era && bash hpc/dispatch_status.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source hpc/_dispatch_lib.sh
# shellcheck disable=SC2034  # read by state_get in _dispatch_lib.sh
STATE=runs/logs/dispatch_state.tsv
if [ -f runs/logs/dispatch.pid ]; then read -r h p < runs/logs/dispatch.pid; echo "dispatcher: running on $h (pid $p)"
else echo "dispatcher: not running"; fi
printf '%-12s %-8s %-12s %-10s %-24s %s\n' adapter attempts state job gpu steps
for ad in $(adapters); do
  arm=${ad%%:*} w=${ad##*:}
  read -r attempts job status <<< "$(state_get "$ad")"
  state=$status
  if done_marker "$arm" "$w"; then state=DONE
  elif [ "$job" != - ]; then state=$(sacct -j "$job" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}'); state=${state:-$status}
  fi
  prog=runs/checkpoints/${arm}_${w}/progress.json gpu=- steps=-
  if [ -f "$prog" ]; then
    gpu=$(sed -n 's/.*"gpu": "\([^"]*\)".*/\1/p' "$prog" | sed 's/NVIDIA //')
    steps="$(sed -n 's/.*"step": \([0-9]*\).*/\1/p' "$prog")/$(sed -n 's/.*"total": \([0-9]*\).*/\1/p' "$prog")"
  fi
  printf '%-12s %-8s %-12s %-10s %-24s %s\n' "$ad" "$attempts" "$state" "$job" "${gpu:--}" "$steps"
done
