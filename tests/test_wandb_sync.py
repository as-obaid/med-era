"""hpc/wandb_sync.sh against a fake `wandb` CLI: only the adapters' offline runs (ids from runs/checkpoints/*/
wandb_run_id) that are not synced yet, in start order; smoke runs untouched; --dry-run and explicit dirs."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HPC = Path(__file__).resolve().parents[1] / "hpc"
FAKE_WANDB = r'''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_LOG"
[ "$1" = sync ] || exit 1
shift
for d in "$@"; do b=$(basename "$d"); id=${b#offline-run-*-}; touch "$d/run-$id.wandb.synced"; done
'''
SMOKE = "offline-run-20261104_223835-2zay3wnl"
QW1_A = "offline-run-20261105_002040-qwen-w1-bf16-s42-e8f8eb19"
QW1_B = "offline-run-20261105_013549-qwen-w1-bf16-s42-e8f8eb19"
GA1 = "offline-run-20261105_003659-gemma_al-w1-bf16-s42-d725dfb1"


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "hpc").mkdir(parents=True)
    shutil.copy(HPC / "wandb_sync.sh", r / "hpc" / "wandb_sync.sh")
    # stand-in for hpc/env.sh: only WANDB_DIR (the real one also loads .env and the venv)
    (r / "hpc" / "env.sh").write_text('export WANDB_DIR="$MEDERA_ROOT/.cache/wandb"\n')
    off = r / ".cache" / "wandb" / "wandb"
    for d in (SMOKE, QW1_A, QW1_B, GA1):
        (off / d).mkdir(parents=True)
        (off / d / f"run-{d.split('-', 3)[3]}.wandb").write_text("x")
    for name, rid in (("qwen_w1", "qwen-w1-bf16-s42-e8f8eb19"), ("gemma_al_w1", "gemma_al-w1-bf16-s42-d725dfb1")):
        (r / "runs" / "checkpoints" / name).mkdir(parents=True)
        (r / "runs" / "checkpoints" / name / "wandb_run_id").write_text(rid + "\n")
    (off / GA1 / "run-gemma_al-w1-bf16-s42-d725dfb1.wandb.synced").write_text("")   # already uploaded
    b = tmp_path / "bin"
    b.mkdir()
    (b / "wandb").write_text(FAKE_WANDB)
    (b / "wandb").chmod(0o755)
    return r


def run(repo, *args, key=True, rc=0):
    env = {**os.environ, "PATH": f"{repo.parent / 'bin'}:{os.environ['PATH']}", "MEDERA_ROOT": str(repo),
           "FAKE_LOG": str(repo.parent / "wandb_calls")}
    env.pop("WANDB_API_KEY", None)
    if key:
        env["WANDB_API_KEY"] = "not-a-real-key"
    out = subprocess.run(["bash", str(repo / "hpc" / "wandb_sync.sh"), *args], env=env, capture_output=True,
                         text=True, timeout=30)
    assert out.returncode == rc, out.stdout + out.stderr
    calls = repo.parent / "wandb_calls"
    return out.stdout, calls.read_text().splitlines() if calls.exists() else []


def test_syncs_only_unsynced_adapter_runs_in_start_order(repo):
    out, calls = run(repo)
    off = repo / ".cache" / "wandb" / "wandb"
    assert calls == [f"sync {off / QW1_A} {off / QW1_B}"]
    assert "already synced: " + GA1 in out and SMOKE not in calls[0]
    assert not list((off / SMOKE).glob("*.synced"))                      # the smoke run stays offline
    out, calls = run(repo)                                               # a second run has nothing left
    assert "nothing to sync" in out and len(calls) == 1


def test_dry_run_lists_and_uploads_nothing_without_a_key(repo):
    out, calls = run(repo, "--dry-run", key=False)
    assert calls == [] and QW1_A in out and QW1_B in out and SMOKE not in out and "dry run" in out


def test_explicit_dirs_and_missing_key(repo):
    off = repo / ".cache" / "wandb" / "wandb"
    out, calls = run(repo, str(off / SMOKE))
    assert calls == [f"sync {off / SMOKE}"]
    _, calls = run(repo, key=False, rc=1)                                # no key: refuses before any upload
    assert len(calls) == 1
    run(repo, str(repo / "nope"), rc=2)
