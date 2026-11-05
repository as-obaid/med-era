# shellcheck shell=bash disable=SC2153  # STATE is set by the caller
# Pure-bash helpers for hpc/dispatch.sh (Explorer login nodes kill CPU-busy Python). Bash 3.2 compatible.

ARMS_ORDER="qwen gemma_al gemma_ex"
WINDOWS_ORDER="w1 w2 w3"

# Adapters in launch order: arms interleaved within each window, so partial results stay balanced.
adapters() {
  local w a
  for w in $WINDOWS_ORDER; do for a in $ARMS_ORDER; do printf '%s:%s\n' "$a" "$w"; done; done
}

# gpu_nodes allow|exclude  <  `sinfo -h -N -p <parts> -o "%N|%P|%G|%f"`
# allow = h100, h200, and a100 carrying the a100@80g feature (all >= 80 GB); exclude = every other GPU node.
# CPU-only nodes (no gpu gres) are ignored: they can never satisfy --gres=gpu:1.
gpu_nodes() {
  awk -F'|' -v want="$1" '
    $3 ~ /^gpu:/ {
      split($3, g, ":"); type = g[2]
      ok = (type == "h100" || type == "h200" || (type == "a100" && $4 ~ /(^|,)a100@80g(,|$)/))
      label = (type == "a100" && ok) ? "a100@80g" : type
      if ((want == "allow") == ok) seen[$1] = label
    }
    END { for (n in seen) print n, seen[n] }' | sort -u
}

exclude_csv() { gpu_nodes exclude | awk '{print $1}' | paste -sd, -; }

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

# State file ($STATE): one line per adapter: "<arm:window> <attempts> <job_id> <status>".
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
