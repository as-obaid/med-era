#!/usr/bin/env bash
# Build the MedEra env at $MEDERA_ROOT/.env-medera with the pinned stack. Needs internet, so run it
# on the login node (it only downloads and installs wheels). Safe to rerun.
#   cd /scratch/$USER/med-era && bash hpc/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$UV" ] || { echo "[setup] uv not found; install uv and rerun"; exit 1; }

"$UV" python install --no-bin 3.11
[ -x "$MEDERA_ENV/bin/python" ] || "$UV" venv --python 3.11 "$MEDERA_ENV"
PY="$MEDERA_ENV/bin/python"

"$UV" pip install --python "$PY" "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
"$UV" pip install --python "$PY" -r requirements.txt
"$UV" pip install --python "$PY" --no-deps -e .

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
import torch, transformers, peft, fla  # noqa: F401  import check
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401  kernel module importable
print(f"[setup] torch cuda build {torch.version.cuda}")
sys.exit(f"[setup] version mismatch: {bad}" if bad else 0)
PYEOF
echo "[setup] DONE env=$MEDERA_ENV"
