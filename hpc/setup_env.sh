#!/usr/bin/env bash
# Build the MedEra env at $MEDERA_ROOT/.env-medera with the pinned stack. Safe to rerun.
#
# 1. Install on the transfer node: it has internet, no per-user CPU cap and fast /scratch I/O. (The
#    login node caps a user at 0.4 CPU, which makes unpacking ~20k files take hours.) discovery-xfer
#    runs CentOS 7, so wheels are selected for the Explorer compute nodes (RHEL 9, glibc 2.34):
#      ssh discovery-xfer 'bash /scratch/$USER/med-era/hpc/setup_env.sh'
# 2. The pin check below runs here (package metadata only). Importing torch needs the compute-node OS
#    (RHEL 9; login nodes kill CPU-busy Python), so the import check is the pytest step of the first
#    CPU job (hpc/build_data.slurm). Re-run only the pin check with: bash hpc/setup_env.sh --verify
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh
PY="$MEDERA_ENV/bin/python"

if [ "${1:-}" != "--verify" ]; then
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  [ -x "$UV" ] || { echo "[setup] uv not found; install uv and rerun"; exit 1; }
  TARGET=(--python-platform x86_64-manylinux_2_34)
  "$UV" python install --no-bin 3.11
  [ -x "$PY" ] || "$UV" venv --python 3.11 "$MEDERA_ENV"
  "$UV" pip install --python "$PY" "${TARGET[@]}" "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
  "$UV" pip install --python "$PY" "${TARGET[@]}" -r requirements.txt
  "$UV" pip install --python "$PY" "${TARGET[@]}" --no-deps -e .
fi

"$PY" - <<'PYEOF'
import importlib.metadata as md, re, sys
pins = {}
for line in open("requirements.txt"):
    m = re.match(r"\s*([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
    if m:
        pins[m.group(1)] = m.group(2)
bad = []
for name, want in pins.items():
    have = md.version(name)
    ok = have.split("+")[0] == want
    print(f"[setup] {name:24s} {have:16s} {'ok' if ok else 'MISMATCH (want ' + want + ')'}")
    bad += [] if ok else [name]
print(f"[setup] python {sys.version.split()[0]}; torch build {md.version('torch')}")
sys.exit(f"[setup] version mismatch: {bad}" if bad else 0)
PYEOF
echo "[setup] DONE env=$MEDERA_ENV (imports are checked by pytest in hpc/build_data.slurm)"
