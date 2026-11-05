#!/usr/bin/env bash
# Stop the dispatcher loop (any phase) by the PID in its pid file, and nothing else. Submitted jobs keep running
# (rerunning hpc/dispatch.sh later adopts them); the scancel command for them is printed, never run.
#   cd /scratch/$USER/med-era && bash hpc/dispatch_stop.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=hpc/_dispatch_lib.sh disable=SC1091
source hpc/_dispatch_lib.sh
PIDFILE=runs/logs/dispatch.pid
rc=0
if [ ! -f "$PIDFILE" ]; then
  echo "no pid file: the dispatcher is not running"
else
  read -r host pid _ < "$PIDFILE"
  if [ "$host" != "$(hostname -s)" ]; then
    echo "the dispatcher runs on $host; stop it there: ssh $host 'bash $(pwd)/hpc/dispatch_stop.sh'"; rc=1
  elif ps -p "$pid" -o args= 2>/dev/null | grep -q dispatch.sh; then
    kill "$pid" && echo "stopped dispatcher pid $pid on $host"
    for _ in 1 2 3 4 5; do ps -p "$pid" >/dev/null 2>&1 || break; sleep 1; done
  else
    echo "pid $pid on $host is not the dispatcher (stale pid file $PIDFILE); nothing killed"; rc=1
  fi
fi
ids=$(squeue -u "$USER" -h -o '%i %j' 2>/dev/null | awk -v re="$MEDERA_JOB_RE" '$2 ~ re {print $1}' | paste -sd' ' -)
if [ -n "$ids" ]; then
  echo "MedEra GPU jobs left queued or running: $ids"
  echo "to cancel them too (not done here): scancel $ids"
else
  echo "no MedEra GPU jobs in the queue"
fi
exit "$rc"
