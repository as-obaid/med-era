# shellcheck shell=bash disable=SC2153  # STATE is set by the caller
# Pure-bash helpers for hpc/dispatch.sh and dispatch_status.sh (Explorer login nodes kill CPU-busy Python).
# Bash 3.2 compatible. Callers set STATE (the dispatcher's state file) before state_get / state_set, and PHASE
# (train | ppl | probe; unset = train) before the unit_* helpers.

ARMS_ORDER="qwen gemma_al gemma_ex"
WINDOWS_ORDER="w1 w2 w3"
# The 11 evaluation variants, bases first, then each window's arms (must match medera.variants.core_variants).
VARIANTS_ORDER="qwen_base gemma_base qwen_w1 gemma_al_w1 gemma_ex_w1 qwen_w2 gemma_al_w2 gemma_ex_w2 qwen_w3 gemma_al_w3 gemma_ex_w3"
# shellcheck disable=SC2034  # used by dispatch.sh and dispatch_status.sh
PHASES="train ppl probe"
# shellcheck disable=SC2034  # used by dispatch.sh and dispatch_stop.sh
ADAPTER_JOB_RE='^medera_(qwen|gemma_al|gemma_ex)_w[123]$'   # Slurm job names of the 9 adapter jobs
# Every MedEra GPU job of any phase (adapter training, medera_ppl_<variant>, medera_probe_<variant>): the
# dispatcher's submit cap counts all of them, so a train and an eval phase never hold more than the cap together.
# shellcheck disable=SC2034
MEDERA_JOB_RE='^medera_((qwen|gemma_al|gemma_ex)_w[123]|(ppl|probe)_((qwen|gemma)_base|(qwen|gemma_al|gemma_ex)_w[123]))$'
# Per-user MaxSubmit of each partition's QOS (sacctmgr show qos, 2026-11-04). A pending job counts against every
# partition it is eligible for, so one multi-partition job uses a slot in all three.
QOS_MAXSUBMIT=${MEDERA_QOS_MAXSUBMIT:-"sharing:4 gpu-short:8 gpu-interactive:4"}

# Adapters in launch order: arms interleaved within each window, so partial results stay balanced.
adapters() {
  local w a
  for w in $WINDOWS_ORDER; do for a in $ARMS_ORDER; do printf '%s:%s\n' "$a" "$w"; done; done
}

# gpu_nodes allow|exclude  <  `sinfo -h -N -p <parts> -o "%N|%P|%G|%f"`   ->  "<node> <gpu type>" lines
# allow = h100 and h200 only (owner decision: sharing H100 + H200); exclude = every other GPU node, including
# every a100 (a100@80g too).
# CPU-only nodes (no gpu gres) are left out of both: they can never satisfy --gres=gpu:1.
gpu_nodes() {
  awk -F'|' -v want="$1" '
    $3 ~ /^gpu:/ {
      split($3, g, ":"); type = g[2]
      ok = (type == "h100" || type == "h200")
      if ((want == "allow") == ok) seen[$1] = type
    }
    END { for (n in seen) print n, seen[n] }' | sort -u
}

exclude_csv() { gpu_nodes exclude | awk '{print $1}' | paste -sd, -; }

# submit_room <max_ours> <ours_queued> <partitions_csv>  <  `squeue -u $USER -h -r -o %P`
# Free submit slots: our own cap minus our queued adapter jobs, and every partition's MaxSubmit minus the user's
# jobs (or array tasks) eligible for it, whichever is smallest. Never negative.
submit_room() {
  awk -v max_ours="$1" -v ours="$2" -v parts="$3" -v limits="$QOS_MAXSUBMIT" '
    BEGIN { n = split(limits, l, " "); for (i = 1; i <= n; i++) { split(l[i], kv, ":"); lim[kv[1]] = kv[2] } }
    NF { m = split($1, p, ","); for (i = 1; i <= m; i++) used[p[i]]++ }
    END {
      room = max_ours - ours
      k = split(parts, q, ",")
      for (i = 1; i <= k; i++) if (q[i] in lim) { r = lim[q[i]] - used[q[i]]; if (r < room) room = r }
      print (room > 0 ? room : 0)
    }'
}

# classify STATE EXITCODE -> active | completed | resume | failed | cancelled  (sacct State, ExitCode "N:S")
classify() {
  local state=$1 code=${2%%:*}
  case "$state" in
    PENDING*|RUNNING*|REQUEUED*|CONFIGURING*|COMPLETING*|SUSPENDED*|RESIZING*|STAGE_OUT*) echo active ;;
    COMPLETED*) echo completed ;;
    TIMEOUT*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*) echo resume ;;
    CANCELLED*) echo cancelled ;;
    FAILED*) if [ "$code" = 3 ] || [ "$code" = 4 ]; then echo resume; else echo failed; fi ;;
    *) echo failed ;;
  esac
}

# attempt_of_comment "<squeue %k>" -> N for a job submitted with --comment=medera_attempt=N (else empty).
attempt_of_comment() { printf '%s\n' "$1" | sed -n 's/.*medera_attempt=\([0-9][0-9]*\).*/\1/p'; }

# json_field <file> <key>: a scalar from the one-line progress.json that lora_cpt.py writes (empty if absent).
json_field() {
  [ -f "$1" ] || return 0
  sed -n "s/.*\"$2\": *\"\{0,1\}\([^\",}]*\)\"\{0,1\}[,}].*/\1/p" "$1" | head -1
}

# State file ($STATE): one line per adapter: "<arm:window> <attempts> <job_id> <status>".
# status: new | submitted | verify | retry | done | failed | gave_up | cancelled
state_get() {
  local line
  line=$( [ -f "$STATE" ] && awk -v k="$1" '$1 == k {print $2, $3, $4}' "$STATE" )
  echo "${line:-0 - new}"
}

state_set() {  # state_set <adapter> <attempts> <job_id> <status>
  local tmp="$STATE.tmp"
  { [ -f "$STATE" ] && awk -v k="$1" '$1 != k' "$STATE"; printf '%s %s %s %s\n' "$1" "$2" "$3" "$4"; } > "$tmp"
  mv "$tmp" "$STATE"
}

# done_marker <arm> <window>: the adapter and its DONE marker (written by lora_cpt.py after the audit passes).
done_marker() { [ -f "runs/adapters/$1_$2/DONE" ] && [ -f "runs/adapters/$1_$2/adapter_config.json" ]; }

# ---------------------------------------------------------------------------------------------- phases and units
# A unit is what one job works on: an adapter "<arm>:<window>" (train) or a variant name (ppl, probe).

units() {
  local v
  if [ "${PHASE:-train}" = train ]; then adapters; else for v in $VARIANTS_ORDER; do echo "$v"; done; fi
}

unit_job_name() { if [ "${PHASE:-train}" = train ]; then echo "medera_${1%%:*}_${1##*:}"; else echo "medera_${PHASE}_$1"; fi; }

unit_script() { if [ "${PHASE:-train}" = train ]; then echo "hpc/train.slurm ${1%%:*} ${1##*:}"; else echo "hpc/eval.slurm $PHASE $1"; fi; }

# unit_done <unit>: the marker the job writes after its last cell/chunk (train: the adapter's DONE marker).
unit_done() {
  case ${PHASE:-train} in
    train) done_marker "${1%%:*}" "${1##*:}" ;;
    ppl) [ -f "results/scores/perplexity/$1/DONE" ] ;;
    probe) [ -f "results/answers/$1/DONE" ] ;;
    *) return 1 ;;
  esac
}

unit_progress() {  # the progress.json the job rewrites after each step, cell or chunk
  if [ "${PHASE:-train}" = train ]; then echo "runs/checkpoints/${1%%:*}_${1##*:}/progress.json"
  else echo "runs/eval/$PHASE/$1/progress.json"; fi
}

prereg_tag() { sed -n 's/^prereg_tag:[[:space:]]*\([^[:space:]#]*\).*/\1/p' configs/probes.yaml 2>/dev/null | head -1; }

# unit_blocked <unit>: prints why the unit cannot be submitted yet and returns 0; returns 1 if it can be.
# ppl/probe of an adapter needs the trained adapter; probe also needs a frozen battery and the pre-registration tag
# (the job re-checks the full gate and exits 2 if it fails; this only avoids submitting jobs that would refuse).
unit_blocked() {
  local tag
  [ "${PHASE:-train}" = train ] && return 1
  case $1 in
    *_w[123]) if ! done_marker "${1%_w[123]}" "${1##*_}"; then echo "adapter runs/adapters/$1 is not trained"; return 0; fi ;;
  esac
  if [ "$PHASE" = probe ]; then
    if ! grep -q '^status:[[:space:]]*frozen' configs/probes.yaml 2>/dev/null; then
      echo "battery configs/probes.yaml is not frozen (status: draft)"; return 0
    fi
    tag=$(prereg_tag)
    if [ -z "$tag" ] || ! git rev-parse -q --verify "refs/tags/$tag^{commit}" >/dev/null 2>&1; then
      echo "pre-registration tag ${tag:-<none>} does not exist (git fetch --tags?)"; return 0
    fi
  fi
  return 1
}
