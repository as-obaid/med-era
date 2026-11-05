#!/usr/bin/env bash
# One table for the training dispatcher (login node, no Python):
#   adapter | attempts | state | job | partition/node/GPU | steps done
#   cd /scratch/$USER/med-era && bash hpc/dispatch_status.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=hpc/_dispatch_lib.sh disable=SC1091
source hpc/_dispatch_lib.sh
STATE=runs/logs/dispatch_state.tsv PIDFILE=runs/logs/dispatch.pid
if [ -f "$PIDFILE" ]; then
  read -r h p < "$PIDFILE"
  if [ "$h" != "$(hostname -s)" ]; then echo "dispatcher: pid file says $h pid $p (run this on $h to check it is alive)"
  elif ps -p "$p" -o args= 2>/dev/null | grep -q dispatch.sh; then echo "dispatcher: running on $h (pid $p)"
  else echo "dispatcher: NOT running (stale pid file: $h pid $p)"; fi
else
  echo "dispatcher: not running"
fi
row() { printf '%-12s %-8s %-20s %-9s %-36s %s\n' "$@"; }
row adapter attempts state job partition/node/gpu steps
for ad in $(adapters); do
  arm=${ad%%:*} w=${ad##*:}
  read -r attempts job status <<< "$(state_get "$ad")"
  jstate='' jpart='' jnode=''
  if [ "$job" != - ]; then
    IFS='|' read -r jstate jpart jnode <<< "$(sacct -j "$job" -X -n -P -o State,Partition,NodeList 2>/dev/null | head -1)"
  fi
  jstate=${jstate%% *}   # "CANCELLED by 123" -> CANCELLED
  if done_marker "$arm" "$w"; then state=DONE
  elif [ "$status" = submitted ] || [ "$status" = verify ]; then state=${jstate:-$status}
  else state=$status${jstate:+:$jstate}
  fi
  prog=runs/checkpoints/${arm}_${w}/progress.json
  gpu=$(json_field "$prog" gpu); gpu=${gpu#NVIDIA }
  [ "$(json_field "$prog" job_id)" = "$job" ] || gpu=-   # only the current job's GPU
  if [ "$jstate" = PENDING ]; then where=pending
  elif [ -n "$jpart" ]; then where=$jpart/$jnode/${gpu:--}
  else where=-
  fi
  steps=-
  [ -f "$prog" ] && steps="$(json_field "$prog" step)/$(json_field "$prog" total)"
  row "$ad" "$attempts" "$state" "$job" "$where" "$steps"
done
