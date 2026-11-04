# shellcheck shell=bash
# Shared checks for the submit helpers. Usage: preflight <data-root> <model-key>...
preflight() {
  local root=$1; shift
  [ -x "$MEDERA_ENV/bin/python" ] || { echo "[preflight] env missing: run bash hpc/setup_env.sh"; return 1; }
  for m in "$@"; do
    HF_HUB_OFFLINE=1 python -m medera.train.download --check --model "$m" >/dev/null \
      || { echo "[preflight] $m not in the local HF cache: run bash hpc/download_models.sh $m"; return 1; }
    [ -f "$root/processed/hrt/$m/budget.json" ] \
      || { echo "[preflight] $root/processed/hrt/$m/budget.json missing: build the data first"; return 1; }
  done
  echo "[preflight] OK: env, model snapshots and packed data for: $*"
}

# Optional A100 fallback (off by default). bf16 needs the 80 GB card; on a 40 GB A100 use the *_q4 arms.
gpu_args() {
  case "${MEDERA_GPU:-h200}" in
    h200) ;;
    a100) echo "--gres=gpu:a100:1 --constraint=a100@80g" ;;
    *) echo "[preflight] MEDERA_GPU must be h200 or a100" >&2; return 1 ;;
  esac
}

# Model key for an arm, from configs/train.yaml.
arm_model() { python -c "import sys; from medera.config import load_yaml; print(load_yaml('configs/train.yaml')['arms'][sys.argv[1]]['model'])" "$1"; }
