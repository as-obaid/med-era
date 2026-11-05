#!/usr/bin/env bash
# Training dispatcher: a low-CPU bash loop for the LOGIN node (no Python there; it only calls Slurm tools).
# Keeps up to MEDERA_MAX_SUBMIT (4) adapter jobs submitted across sharing, gpu-short and gpu-interactive: a pending
# multi-partition job counts against every partition's per-user MaxSubmit, and sharing and gpu-interactive allow 4
# (a 9-task array is refused). Each job asks for one GPU (--gres=gpu:1) with every GPU node under 80 GB excluded
# (h100, h200 and a100@80g remain; list built from sinfo at each submit), 1 h. Every MEDERA_POLL_S (150 s) it
# submits the next adapters in launch order (w1 qwen, gemma_al, gemma_ex, then w2, then w3) into free slots,
# resubmits TIMEOUT / NODE_FAIL / PREEMPTED / wall-guard (exit 3) / small-GPU (exit 4) jobs so they resume from
# their checkpoint (at most MEDERA_MAX_ATTEMPTS = 3 attempts per adapter), never resubmits a Python or audit
# failure (it logs it and moves on), skips adapters with a DONE marker, and exits when all 9 are done or given up.
# State: runs/logs/dispatch_state.tsv. A restarted dispatcher adopts jobs already queued under an adapter's name
# instead of submitting duplicates. Log: runs/logs/dispatch.log; pid file: runs/logs/dispatch.pid.
#   cd /scratch/$USER/med-era && nohup bash hpc/dispatch.sh >/dev/null 2>&1 </dev/null &   # start the loop
#   bash hpc/dispatch.sh --dry-run     # one poll that submits nothing and writes no state: the plan
#   bash hpc/dispatch.sh --test-only   # allow/exclude lists + `sbatch --test-only` for the next adapter
#   bash hpc/dispatch.sh --once        # one real poll, then exit (refused while the loop runs)
#   bash hpc/dispatch_status.sh        # table: adapter | attempts | state | job | partition/node/GPU | steps
#   bash hpc/dispatch_stop.sh          # stop the loop by its pid file; submitted jobs keep running
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=hpc/_dispatch_lib.sh disable=SC1091
source hpc/_dispatch_lib.sh
LOGDIR=runs/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/dispatch.log PIDFILE=$LOGDIR/dispatch.pid STATE=$LOGDIR/dispatch_state.tsv
EXTRA=$LOGDIR/dispatch_exclude_extra.txt   # nodes where a job found a GPU under 79 GB (exit 4)
PARTS=${MEDERA_PARTITIONS:-sharing,gpu-short,gpu-interactive}
MAX_SUBMIT=${MEDERA_MAX_SUBMIT:-4} MAX_ATTEMPTS=${MEDERA_MAX_ATTEMPTS:-3} POLL=${MEDERA_POLL_S:-150}
SBATCH_ARGS=(--account=ostadabbas --partition="$PARTS" --gres=gpu:1 --time=01:00:00 --cpus-per-task=8 --mem=96G)
SINFO_FMT='%N|%P|%G|%f'
HOST=$(hostname -s)
MODE=loop SLEEP_PID=
case "${1:-}" in
  "") ;;
  --dry-run) MODE='dry' ;;
  --test-only) MODE='test' ;;
  --once) MODE='once' ;;
  -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
  *) echo "usage: bash hpc/dispatch.sh [--dry-run | --test-only | --once]" >&2; exit 2 ;;
esac

log() {
  if [ "$MODE" = dry ]; then printf '  %s\n' "$*"; else printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; fi
}
last_lines() { tail -3 "$1" 2>/dev/null | tr '\n' ' ' | cut -c1-400; }

# Every GPU node under 80 GB in $PARTS (sinfo, now) plus nodes a job found too small. Fails when sinfo lists no
# allowed node, so a Slurm hiccup never leads to a submit with a missing exclude list.
exclude_list() {
  local nodes
  nodes=$(sinfo -h -N -p "$PARTS" -o "$SINFO_FMT" 2>/dev/null)
  [ -n "$(gpu_nodes allow <<< "$nodes")" ] || return 1
  { gpu_nodes exclude <<< "$nodes" | awk '{print $1}'; if [ -f "$EXTRA" ]; then cat "$EXTRA"; fi; } | sort -u | paste -sd, -
}

# sbatch_args <arm:window> <attempt>: sets CMD to everything after `sbatch` (submit adds --parsable).
sbatch_args() {
  local arm=${1%%:*} w=${1##*:} excl
  excl=$(exclude_list) || return 1
  CMD=("${SBATCH_ARGS[@]}" ${excl:+"--exclude=$excl"} --job-name="medera_${arm}_${w}"
       --comment="medera_attempt=$2" --export="ALL,MEDERA_ATTEMPT=$2" hpc/train.slurm "$arm" "$w")
}

submit() {  # submit <arm:window>; returns 1 if nothing was submitted (e.g. Slurm refused: a QOS submit limit)
  local ad=$1 attempts job status n out id
  read -r attempts job status <<< "$(state_get "$ad")"
  n=$((attempts + 1))
  if ! sbatch_args "$ad" "$n"; then log "$ad not submitted: sinfo lists no >= 80 GB GPU node in $PARTS"; return 1; fi
  if [ "$MODE" = dry ]; then
    printf '  WOULD SUBMIT %-12s attempt %s: sbatch --parsable %s\n' "$ad" "$n" \
      "$(printf '%s ' "${CMD[@]}" | sed 's/--exclude=[^ ]*/--exclude=<the excluded nodes above>/')"
    state_set "$ad" "$n" dry-run submitted
    return 0
  fi
  if out=$(sbatch --parsable "${CMD[@]}" 2>&1); then
    id=$(printf '%s\n' "$out" | tail -1); id=${id%%;*}
    case $id in
      ''|*[!0-9]*) log "$ad: unexpected sbatch reply ($out); the next poll adopts the job if it was queued"; return 1 ;;
    esac
    state_set "$ad" "$n" "$id" submitted
    log "$ad attempt $n submitted as job $id"
  else
    log "$ad submit refused (will retry next poll): $(printf '%s' "$out" | tr '\n' ' ')"
    return 1
  fi
}

# One pass over the 9 adapters: update each from Slurm, then submit the waiting ones into the free slots.
# Leaves OPEN=0 when every adapter is final (done, failed, gave_up or cancelled).
poll_once() {
  local ad arm w attempts job status found cmt n jstate jcode jnode decision out ours room queue=()
  OPEN=0
  for ad in $(adapters); do
    arm=${ad%%:*} w=${ad##*:}
    read -r attempts job status <<< "$(state_get "$ad")"
    if done_marker "$arm" "$w"; then
      [ "$status" = "done" ] || { state_set "$ad" "$attempts" "$job" "done"; log "$ad DONE (job $job, $attempts attempts)"; }
      continue
    fi
    case $status in done|failed|gave_up|cancelled) continue ;; esac
    if [ "$status" = new ] || [ "$status" = retry ]; then
      # Adopt a job already queued under this adapter's name (a restarted dispatcher, or a crash right after sbatch).
      found='' cmt=''
      read -r found cmt <<< "$(squeue -u "$USER" -h -n "medera_${arm}_${w}" -o '%i %k' 2>/dev/null | head -1)"
      if [ -n "$found" ] && [ "$found" != "$job" ]; then
        n=$(attempt_of_comment "$cmt"); [ -n "$n" ] || n=$((attempts + 1))
        attempts=$n job=$found status=submitted
        state_set "$ad" "$attempts" "$job" submitted
        log "$ad adopted queued job $job (attempt $attempts)"
      fi
    fi
    if [ "$status" = submitted ] || [ "$status" = verify ]; then
      jstate='' jcode='' jnode=''
      IFS='|' read -r jstate jcode jnode <<< "$(sacct -j "$job" -X -n -P -o State,ExitCode,NodeList 2>/dev/null | head -1)"
      decision=active
      [ -n "$jstate" ] && decision=$(classify "$jstate" "$jcode")
      out=$LOGDIR/medera_${arm}_${w}_$job.out
      case $decision in
        active) OPEN=1; continue ;;
        completed)
          # The job writes DONE before it exits, but NFS can show COMPLETED here first: look again next poll.
          if [ "$status" = submitted ]; then state_set "$ad" "$attempts" "$job" verify; OPEN=1; continue; fi
          state_set "$ad" "$attempts" "$job" failed
          log "$ad job $job COMPLETED without a DONE marker (audit failure?), NOT resubmitted: $(last_lines "$out")"
          continue ;;
        failed)
          state_set "$ad" "$attempts" "$job" failed
          log "$ad job $job $jstate $jcode on $jnode: NOT resubmitted (Python or audit error): $(last_lines "$out")"
          continue ;;
        cancelled)
          state_set "$ad" "$attempts" "$job" cancelled
          log "$ad job $job $jstate: cancelled by hand, NOT resubmitted"
          continue ;;
        resume)
          if [ "${jcode%%:*}" = 4 ]; then
            echo "$jnode" >> "$EXTRA"; log "$ad job $job: GPU under 79 GB on $jnode, node excluded from now on"
          fi
          if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
            state_set "$ad" "$attempts" "$job" gave_up
            log "$ad job $job $jstate after $attempts attempts: giving up"
            continue
          fi
          state_set "$ad" "$attempts" "$job" retry
          log "$ad job $job $jstate ($jcode) on $jnode: resubmitting to resume from its checkpoint" ;;
      esac
    fi
    OPEN=1; queue+=("$ad")
  done
  [ "${#queue[@]}" -gt 0 ] || return 0
  ours=$(squeue -u "$USER" -h -r -o %j 2>/dev/null | grep -c -E "$ADAPTER_JOB_RE")
  room=$(squeue -u "$USER" -h -r -o %P 2>/dev/null | submit_room "$MAX_SUBMIT" "$ours" "$PARTS")
  if [ "$MODE" = dry ]; then
    log "waiting to be submitted: ${queue[*]}"
    log "free submit slots: $room (our adapter jobs queued: $ours, cap $MAX_SUBMIT; QOS MaxSubmit: $QOS_MAXSUBMIT)"
  fi
  for ad in "${queue[@]}"; do
    [ "$room" -le 0 ] && break
    submit "$ad" || break
    room=$((room - 1))
  done
}

final_summary() { awk '{print $1 "=" $4}' "$STATE" 2>/dev/null | paste -sd' ' -; }

on_exit() {
  [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null
  if [ -f "$PIDFILE" ] && [ "$(cat "$PIDFILE")" = "$HOST $$" ]; then rm -f "$PIDFILE"; fi
  log "dispatcher stopped (pid $$)"
}

acquire_lock() {  # one dispatcher at a time, recorded as "<host> <pid>" in the pid file
  local ohost opid
  if [ -f "$PIDFILE" ]; then
    read -r ohost opid < "$PIDFILE"
    if [ "$ohost" != "$HOST" ]; then
      echo "pid file $PIDFILE says a dispatcher runs on $ohost (pid $opid); check there first: ssh $ohost ps -p $opid -o args="
      exit 1
    fi
    if ps -p "$opid" -o args= 2>/dev/null | grep -q dispatch.sh; then
      echo "dispatcher already running on $HOST (pid $opid)"; exit 1
    fi
    log "stale pid file ($ohost pid $opid) replaced"
  fi
  echo "$HOST $$" > "$PIDFILE"
  trap on_exit EXIT
  trap 'exit 0' TERM INT
}

print_nodes() {
  local nodes
  nodes=$(sinfo -h -N -p "$PARTS" -o "$SINFO_FMT" 2>/dev/null)
  echo "allowed GPU nodes (>= 80 GB) in $PARTS:"; gpu_nodes allow <<< "$nodes" | sed 's/^/  /'
  echo "excluded GPU nodes (passed as --exclude):"; gpu_nodes exclude <<< "$nodes" | sed 's/^/  /'
  if [ -s "$EXTRA" ]; then echo "also excluded (a job found a GPU under 79 GB):"; sort -u "$EXTRA" | sed 's/^/  /'; fi
}

case $MODE in
  test)
    print_nodes
    next=
    for ad in $(adapters); do done_marker "${ad%%:*}" "${ad##*:}" || { next=$ad; break; }; done
    [ -n "$next" ] || { echo "all adapters are done"; exit 0; }
    read -r attempts _ _ <<< "$(state_get "$next")"
    sbatch_args "$next" $((attempts + 1)) || { echo "sinfo lists no >= 80 GB GPU node in $PARTS"; exit 1; }
    echo "next adapter: $next (attempt $((attempts + 1)))"
    echo "+ sbatch --test-only ${CMD[*]}"
    sbatch --test-only "${CMD[@]}"
    exit $? ;;
  dry)
    # Work on copies so nothing real changes.
    DRY_STATE=$LOGDIR/dispatch_state.dry-run.tsv DRY_EXTRA=$LOGDIR/dispatch_exclude_extra.dry-run.txt
    rm -f "$DRY_STATE" "$DRY_EXTRA"
    if [ -f "$STATE" ]; then cp "$STATE" "$DRY_STATE"; fi
    if [ -f "$EXTRA" ]; then cp "$EXTRA" "$DRY_EXTRA"; fi
    STATE=$DRY_STATE EXTRA=$DRY_EXTRA
    trap 'rm -f "$DRY_STATE" "$DRY_STATE.tmp" "$DRY_EXTRA"' EXIT
    echo "DRY RUN on $HOST: nothing is submitted and no state is written."
    echo "partitions $PARTS; cap $MAX_SUBMIT jobs; max $MAX_ATTEMPTS attempts; poll ${POLL}s; order: $(adapters | paste -sd' ' -)"
    print_nodes
    echo "plan for this poll:"
    poll_once
    [ "$OPEN" = 0 ] && echo "  all adapters final: $(final_summary)"
    exit 0 ;;
  once)
    acquire_lock
    poll_once
    [ "$OPEN" = 0 ] && log "all adapters final: $(final_summary)"
    exit 0 ;;
esac

acquire_lock
log "dispatcher started on $HOST pid $$: max_submit=$MAX_SUBMIT max_attempts=$MAX_ATTEMPTS poll=${POLL}s parts=$PARTS"
while :; do
  poll_once
  if [ "$OPEN" = 0 ]; then log "all adapters final: $(final_summary)"; exit 0; fi
  sleep "$POLL" & SLEEP_PID=$!
  wait "$SLEEP_PID"; SLEEP_PID=
done
