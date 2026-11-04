#!/usr/bin/env bash
# Publish-safety scan of every tracked and staged file. Run before each push; exits 1 on any finding.
#   bash scripts/check_publish.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
fail=0
files() { git ls-files -z --cached -- . ':(exclude)scripts/check_publish.sh'; }

# Secrets, personal paths, other projects, AI attribution. Patterns are split so this file never matches itself.
patterns=(
  'wandb''_v1_'                          # W&B API key prefix
  'WANDB_API_KEY''=[^[:space:]"$}]'       # a key assigned a literal value
  'hf''_[A-Za-z0-9]{30,}'                # Hugging Face token
  'gh[pousr]''_[A-Za-z0-9]{30,}'         # GitHub token
  'github''_pat_'
  'BEGIN [A-Z ]*PRIVATE'' KEY'
  '/Users''/'
  'medical''-llm'
  'Internal''-Docs'
  'Co-Authored''-By'
  'Generated with'' Claude'
  'noreply@''anthropic'
)
for p in "${patterns[@]}"; do
  hits=$(files | xargs -0 grep -n -I -E -e "$p" -- 2>/dev/null)
  if [ -n "$hits" ]; then echo "[check] FOUND /$p/:"; echo "$hits" | head -5; fail=1; fi
done

# Paths that must never be tracked.
tracked=$(git ls-files --cached -- docs/dev claude.md CLAUDE.md .env '.env.*' .netrc wandb runs .cache .env-medera \
          data/smoke 'data/raw/**' 'data/processed/**' '*.safetensors' '*.pt' '*.bin' | grep -v '/\.gitkeep$')
if [ -n "$tracked" ]; then echo "[check] tracked internal/private files:"; echo "$tracked"; fail=1; fi

if [ "$fail" -eq 0 ]; then echo "[check] PASS: $(files | tr -cd '\0' | wc -c | tr -d ' ') files scanned, nothing to leak"; fi
exit "$fail"
