#!/usr/bin/env bash
# Stop the training dispatcher loop by the PID in its pid file. Submitted jobs keep running; rerunning
# hpc/dispatch.sh later picks up where it left off.
#   cd /scratch/$USER/med-era && bash hpc/dispatch_stop.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PIDFILE=runs/logs/dispatch.pid
[ -f "$PIDFILE" ] || { echo "no pid file: the dispatcher is not running"; exit 0; }
read -r host pid < "$PIDFILE"
if [ "$host" != "$(hostname -s)" ]; then
  echo "the dispatcher runs on $host; run this there: ssh $host 'bash $(pwd)/hpc/dispatch_stop.sh'"; exit 1
fi
if ps -p "$pid" -o args= 2>/dev/null | grep -q dispatch.sh; then
  kill "$pid" && echo "stopped dispatcher pid $pid on $host"
else
  echo "pid $pid on $host is not the dispatcher (stale pid file $PIDFILE); nothing killed"; exit 1
fi
