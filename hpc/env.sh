# shellcheck shell=bash
# MedEra cluster environment. Source this from every job and script:
#   source /scratch/$USER/med-era/hpc/env.sh
# Every path lives inside the repo clone on /scratch; nothing is written to $HOME.
export MEDERA_ROOT="${MEDERA_ROOT:-/scratch/$USER/med-era}"
export MEDERA_ENV="$MEDERA_ROOT/.env-medera"
_medera_cache="$MEDERA_ROOT/.cache"

export HF_HOME="$_medera_cache/hf"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"   # owner's existing HF login (read only)
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR="$_medera_cache/pip"
export UV_CACHE_DIR="$_medera_cache/uv"
export UV_PYTHON_INSTALL_DIR="$_medera_cache/uv-python"
export UV_PYTHON_BIN_DIR="$_medera_cache/uv-python-bin"
export TRITON_CACHE_DIR="$_medera_cache/triton"
export TRITON_HOME="$_medera_cache/triton-home"   # triton's get_home_dir() defaults to $HOME (~/.triton)
export TORCH_EXTENSIONS_DIR="$_medera_cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$_medera_cache/inductor"
export TORCH_HOME="$_medera_cache/torch"
export CUDA_CACHE_PATH="$_medera_cache/nv"
export MPLCONFIGDIR="$_medera_cache/matplotlib"
export XDG_CACHE_HOME="$_medera_cache"
export TMPDIR="$MEDERA_ROOT/tmp"
export PYTHONNOUSERSITE=1
export WANDB_DIR="$_medera_cache/wandb"
export WANDB_CACHE_DIR="$_medera_cache/wandb/cache"
export WANDB_CONFIG_DIR="$_medera_cache/wandb/config"
export WANDB_DATA_DIR="$_medera_cache/wandb/data"

mkdir -p "$_medera_cache" "$TMPDIR" "$MEDERA_ROOT/runs/logs" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

# W&B credentials (gitignored .env, chmod 600): loaded without ever echoing values.
case $- in *x*) _medera_xtrace=1; set +x ;; esac
# shellcheck source=/dev/null  # the secret file: never followed or linted
if [ -f "$MEDERA_ROOT/.env" ]; then set -a; . "$MEDERA_ROOT/.env"; set +a; fi
if [ -n "${WANDB_API_KEY:-}" ]; then echo "W&B: key loaded" >&2; else echo "W&B: no key, logging disabled" >&2; fi
if [ -n "${_medera_xtrace:-}" ]; then unset _medera_xtrace; set -x; fi
# Optional user-specific overrides (gitignored).
# shellcheck source=/dev/null
if [ -f "$MEDERA_ROOT/hpc/env.local.sh" ]; then source "$MEDERA_ROOT/hpc/env.local.sh"; fi
# shellcheck source=/dev/null
if [ -x "$MEDERA_ENV/bin/python" ]; then source "$MEDERA_ENV/bin/activate"; fi
unset _medera_cache
