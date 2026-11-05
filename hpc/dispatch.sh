#!/usr/bin/env bash
# Training dispatcher: a low-CPU bash loop for the LOGIN node (no Python there; it only calls Slurm tools).
# Keeps up to MEDERA_MAX_SUBMIT (4) adapter jobs submitted across sharing, gpu-short and gpu-interactive.
# A multi-partition job counts against every partition's MaxSubmit, and sharing allows 4. Each job asks for
# one >= 80 GB GPU (h100, h200 or a100@80g), 1 h. The loop submits the next adapter when a slot frees,
# resubmits TIMEOUT / NODE_FAIL / wall-guard (exit 3) / small-GPU (exit 4) jobs so they resume from their
# checkpoint (at most MEDERA_MAX_ATTEMPTS = 3 attempts per adapter), never resubmits a Python or audit failure,
# and exits when all 9 adapters are final. Finished = adapter + DONE marker, so a rerun skips them.
#   cd /scratch/$USER/med-era && nohup bash hpc/dispatch.sh >/dev/null 2>&1 </dev/null &   # start
#   bash hpc/dispatch.sh --test-only     # print allow/exclude lists + `sbatch --test-only` for the next adapter
#   bash hpc/dispatch_status.sh          # table: adapter | attempts | state | job | GPU | steps
#   bash hpc/dispatch_stop.sh            # stop the loop (by its pid file; running jobs keep running)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source hpc/env.sh 2>/dev/null
source hpc/_dispatch_lib.sh
LOGDIR=runs/logs; mkdir -p "$LOGDIR"
LOG=$LOGDIR/dispatch.log PIDFILE=$LOGDIR/dispatch.pid STATE=$LOGDIR/dispatch_state.tsv EXTRA=$LOGDIR/dispatch_exclude_extra.txt
PARTS=${MEDERA_PARTITIONS:-sharing,gpu-short,gpu-interactive}
MAX_SUBMIT=${MEDERA_MAX_SUBMIT:-4} MAX_ATTEMPTS=${MEDERA_MAX_ATTEMPTS:-3} POLL=${MEDERA_POLL_S:-150}
SBATCH_ARGS=(--account=ostadabbas --partition="$PARTS" --gres=gpu:1 --time=01:00:00 --cpus-per-task=8 --mem=96G)

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

exclude_list() {
  local excl extra
  excl=$(sinfo -h -N -p "$PARTS" -o "%N|%P|%G|%f" | exclude_csv)
  extra=$( [ -f "$EXTRA" ] && sort -u "$EXTRA" | paste -sd, - )
  echo "${excl}${extra:+,$extra}"
}

submit() {  # submit <arm:window>; returns 1 if Slurm refuses (e.g. a QOS submit limit)
  local ad=$1 arm=${1%%:*} w=${1##*:} attempts job status out n
  read -r attempts job status <<< "$(state_get "$ad")"
  n=$((attempts + 1))
  if out=$(sbatch --parsable "${SBATCH_ARGS[@]}" --exclude="$(exclude_list)" --job-name="medera_${arm}_${w}" \
             --export=ALL,MEDERA_ATTEMPT="$n" hpc/train.slurm "$arm" "$w" 2>&1); then
    state_set "$ad" "$n" "${out%%;*}" submitted
    log "$ad attempt $n submitted as job ${out%%;*}"
  else
    log "$ad submit refused (will retry next poll): $out"
    return 1
  fi
}

if [ "${1:-}" = --test-only ]; then
  echo "allowed (>= 80 GB):"; sinfo -h -N -p "$PARTS" -o "%N|%P|%G|%f" | gpu_nodes allow | sed 's/^/  /'
  echo "excluded GPU nodes:"; sinfo -h -N -p "$PARTS" -o "%N|%P|%G|%f" | gpu_nodes exclude | sed 's/^/  /'
  for ad in $(adapters); do done_marker "${ad%%:*}" "${ad##*:}" || break; done
  echo "next adapter: $ad"
  set -x
  sbatch --test-only "${SBATCH_ARGS[@]}" --exclude="$(exclude_list)" --job-name="medera_${ad%%:*}_${ad##*:}" \
    --export=ALL,MEDERA_ATTEMPT=1 hpc/train.slurm "${ad%%:*}" "${ad##*:}"
  exit $?
fi

if [ -f "$PIDFILE" ]; then
  read -r ohost opid < "$PIDFILE"
  if [ "$ohost" = "$(hostname -s)" ] && ps -p "$opid" -o args= 2>/dev/null | grep -q dispatch.sh; then
    echo "dispatcher already running on $ohost (pid $opid)"; exit 1
  fi
  [ "$ohost" = "$(hostname -s)" ] || { echo "pid file says the dispatcher runs on $ohost; check there first"; exit 1; }
fi
echo "$(hostname -s) $$" > "$PIDFILE"
trap 'rm -f "$PIDFILE"; log "dispatcher stopped (pid $$)"' EXIT
log "dispatcher started on $(hostname -s) pid $$: max_submit=$MAX_SUBMIT max_attempts=$MAX_ATTEMPTS poll=${POLL}s parts=$PARTS"

while :; do
  queue=() open=0
  for ad in $(adapters); do
    arm=${ad%%:*} w=${ad##*:}
    read -r attempts job status <<< "$(state_get "$ad")"
    if done_marker "$arm" "$w"; then
      [ "$status" = "done" ] || { state_set "$ad" "$attempts" "$job" "done"; log "$ad DONE"; }
      continue
    fi
    case "$status" in failed|gave_up|cancelled) continue ;; esac
    if [ "$status" = new ]; then  # adopt a job that is already queued under this name (e.g. after a restart)
      found=$(squeue -u "$USER" -h -n "medera_${arm}_${w}" -o %i | head -1)
      [ -n "$found" ] && { attempts=$((attempts > 0 ? attempts : 1)); job=$found; status=submitted
                           state_set "$ad" "$attempts" "$job" submitted; log "$ad adopted queued job $job"; }
    fi
    if [ "$status" = submitted ] || [ "$status" = verify ]; then
      IFS='|' read -r jstate jcode jnode <<< "$(sacct -j "$job" -X -n -P -o State,ExitCode,NodeList 2>/dev/null | head -1)"
      decision=active
      [ -n "${jstate:-}" ] && decision=$(classify "$jstate" "$jcode")
      case "$decision" in
        active) open=1; continue ;;
        completed)
          # The job writes DONE before it exits, but NFS can show COMPLETED here first: wait one poll.
          if [ "$status" = submitted ]; then state_set "$ad" "$attempts" "$job" verify; open=1; continue; fi
          state_set "$ad" "$attempts" "$job" failed
          log "$ad job $job COMPLETED without a DONE marker (audit failure?): $(tail -3 "$LOGDIR/medera_${arm}_${w}_$job.out" 2>/dev/null | tr '\n' ' ')"
          continue ;;
        failed) state_set "$ad" "$attempts" "$job" failed
                log "$ad job $job $jstate $jcode on $jnode: NOT resubmitted: $(tail -3 "$LOGDIR/medera_${arm}_${w}_$job.out" 2>/dev/null | tr '\n' ' ')"
                continue ;;
        cancelled) state_set "$ad" "$attempts" "$job" cancelled; log "$ad job $job cancelled: not resubmitted"; continue ;;
        resume)
          [ "${jcode%%:*}" = 4 ] && { echo "$jnode" >> "$EXTRA"; log "$ad job $job: GPU under 79 GiB on $jnode, node excluded"; }
          if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
            state_set "$ad" "$attempts" "$job" gave_up; log "$ad job $job $jstate after $attempts attempts: giving up"; continue
          fi
          state_set "$ad" "$attempts" "$job" retry; log "$ad job $job $jstate ($jcode) on $jnode: will resume" ;;
      esac
    fi
    open=1; queue+=("$ad")
  done
  if [ "${#queue[@]}" -gt 0 ]; then
    used=$(squeue -u "$USER" -h -o %P | grep -c -E 'sharing|gpu-short|gpu-interactive')
    slots=$((MAX_SUBMIT - used))
    for ad in "${queue[@]}"; do
      [ "$slots" -le 0 ] && break
      submit "$ad" || break
      slots=$((slots - 1))
    done
  fi
  if [ "$open" = 0 ]; then
    log "all adapters final: $(awk '{print $1 "=" $4}' "$STATE" | paste -sd' ' -)"
    exit 0
  fi
  sleep "$POLL"
done
