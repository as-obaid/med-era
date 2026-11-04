# shellcheck shell=bash
# Shared checks for the submit helpers. Pure shell: Explorer login nodes kill CPU-busy Python.

# Value of `<field>:` under top-level `<key>:` in configs/models.yaml.
model_field() { awk -v k="$1" -v f="$2" '$0 ~ "^"k":" {on=1; next} /^[^ #]/ {on=0} on && $1 == f":" {print $2}' configs/models.yaml; }

# Model key of an arm in configs/train.yaml.
arm_model() {
  awk -v a="$1" '$1 == a":" && match($0, /model: *[A-Za-z0-9_]+/) {s = substr($0, RSTART, RLENGTH); sub(/model: */, "", s); print s}' configs/train.yaml
}

# True if the pinned snapshot of <model-key> is fully in $HF_HOME (config + every safetensors shard).
snapshot_ok() {
  local id rev dir f
  id=$(model_field "$1" hf_id); rev=$(model_field "$1" revision)
  dir="$HF_HOME/hub/models--${id//\//--}/snapshots/$rev"
  [ -n "$id" ] && [ -f "$dir/config.json" ] || return 1
  if [ -f "$dir/model.safetensors.index.json" ]; then
    for f in $(grep -o '"[^"]*\.safetensors"' "$dir/model.safetensors.index.json" | tr -d '"' | sort -u); do
      [ -e "$dir/$f" ] || return 1
    done
  else
    compgen -G "$dir/*.safetensors" >/dev/null || return 1
  fi
}

preflight_env() {
  [ -x "$MEDERA_ENV/bin/python" ] || { echo "[preflight] env missing: ssh discovery-xfer 'bash $MEDERA_ROOT/hpc/setup_env.sh'"; return 1; }
}

preflight_models() {
  local m
  for m in "$@"; do
    snapshot_ok "$m" || { echo "[preflight] $m not fully in $HF_HOME: ssh discovery-xfer 'bash $MEDERA_ROOT/hpc/download_models.sh $m'"; return 1; }
  done
  echo "[preflight] model snapshots cached: $*"
}

preflight_data() {  # <data-root> <model-key>...
  local root=$1 m; shift
  for m in "$@"; do
    [ -f "$root/processed/hrt/$m/budget.json" ] || { echo "[preflight] $root/processed/hrt/$m/budget.json missing: sbatch hpc/build_data.slurm $root"; return 1; }
  done
  echo "[preflight] packed data present in $root for: $*"
}

# Optional A100 fallback (off by default). bf16 needs the 80 GB card; on a 40 GB A100 use the *_q4 arms.
gpu_args() {
  case "${MEDERA_GPU:-h200}" in
    h200) ;;
    a100) echo "--gres=gpu:a100:1 --constraint=a100@80g" ;;
    *) echo "[preflight] MEDERA_GPU must be h200 or a100" >&2; return 1 ;;
  esac
}
