#!/usr/bin/env bash
# One table per dispatcher phase (login node, no Python):
#   unit | attempts | state | job | partition/node/GPU | progress (train: optimizer steps; ppl: era cells; probe: chunks)
#   cd /scratch/$USER/med-era && bash hpc/dispatch_status.sh [--phase train|ppl|probe]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=hpc/_dispatch_lib.sh disable=SC1091
source hpc/_dispatch_lib.sh
PHASE=train
case "${1:-}" in
  "") ;;
  --phase) PHASE=${2:-} ;;
  --phase=*) PHASE=${1#*=} ;;
  *) echo "usage: bash hpc/dispatch_status.sh [--phase train|ppl|probe]" >&2; exit 2 ;;
esac
case " $PHASES " in *" $PHASE "*) ;; *) echo "unknown phase $PHASE (choose: $PHASES)" >&2; exit 2 ;; esac
if [ "$PHASE" = train ]; then STATE=runs/logs/dispatch_state.tsv; else STATE=runs/logs/dispatch_${PHASE}_state.tsv; fi
PIDFILE=runs/logs/dispatch.pid
if [ -f "$PIDFILE" ]; then
  read -r h p ph < "$PIDFILE"
  if [ "$h" != "$(hostname -s)" ]; then echo "dispatcher: pid file says $h pid $p (run this on $h to check it is alive)"
  elif ps -p "$p" -o args= 2>/dev/null | grep -q dispatch.sh; then echo "dispatcher: running on $h (pid $p, phase ${ph:-train})"
  else echo "dispatcher: NOT running (stale pid file: $h pid $p)"; fi
else
  echo "dispatcher: not running"
fi
row() { printf '%-12s %-8s %-20s %-9s %-36s %s\n' "$@"; }
if [ "$PHASE" = train ]; then row adapter attempts state job partition/node/gpu steps
else row "$PHASE" attempts state job partition/node/gpu progress; fi
for ad in $(units); do
  read -r attempts job status <<< "$(state_get "$ad")"
  jstate='' jpart='' jnode=''
  if [ "$job" != - ]; then
    IFS='|' read -r jstate jpart jnode <<< "$(sacct -j "$job" -X -n -P -o State,Partition,NodeList 2>/dev/null | head -1)"
  fi
  jstate=${jstate%% *}   # "CANCELLED by 123" -> CANCELLED
  if unit_done "$ad"; then state=DONE
  elif [ "$status" = submitted ] || [ "$status" = verify ]; then state=${jstate:-$status}
  else state=$status${jstate:+:$jstate}
  fi
  prog=$(unit_progress "$ad")
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
