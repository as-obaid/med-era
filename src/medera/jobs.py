"""Shared plumbing for the GPU evaluation jobs (perplexity, probing): exit codes, the stop deadline, SIGTERM,
provenance and atomic writes. The deadline and margin come from configs/train.yaml so every GPU job stops the same way.

Exit codes match training, so hpc/dispatch.sh treats them alike: 0 done (DONE marker written), 1 error,
2 refused (pre-registration gate or a mismatch that must not be overwritten; never resubmitted), 3 stopped before the
deadline or on SIGTERM (resubmit to resume), 4 the GPU has under 79 GB.
"""
import json
import os
import signal
import socket
import time
from pathlib import Path

EXIT_OK, EXIT_FAIL, EXIT_REFUSED, EXIT_WALL, EXIT_SMALL_GPU = 0, 1, 2, 3, 4

__all__ = ["EXIT_OK", "EXIT_FAIL", "EXIT_REFUSED", "EXIT_WALL", "EXIT_SMALL_GPU", "Stopper", "provenance",
           "write_json_atomic", "write_text_atomic", "write_progress", "env_versions", "git_sha"]


# lora_cpt imports torch (via train.targets); import it lazily so the CPU table/report paths do not need torch.
def env_versions() -> dict:
    from medera.train.lora_cpt import _env_versions
    return _env_versions()


def git_sha() -> str:
    from medera.train.lora_cpt import _git_sha
    return _git_sha()


def write_progress(path: Path | None, state: dict, total: int, attempt: dict | None) -> None:
    from medera.train.lora_cpt import _write_progress
    _write_progress(path, state, total, attempt)


class Stopper:
    """Decides before each unit of work whether it still fits before the job's deadline (Slurm end time minus
    wall_margin_s, or the wall-time guard), and records SIGTERM so the current unit can finish and the job exit 3."""

    def __init__(self, t_start: float, cfg: dict, env=os.environ, clock=time.time):
        from medera.train.lora_cpt import job_deadline

        self.deadline = job_deadline(t_start, cfg, env)
        self.clock = clock
        self.signal = None

    def install(self) -> "Stopper":
        def _on_term(signum, frame):
            self.signal = signum
        signal.signal(signal.SIGTERM, _on_term)
        return self

    def fits(self, est_s: float) -> bool:
        return self.signal is None and self.clock() + est_s <= self.deadline

    def minutes_left(self) -> float:
        return (self.deadline - self.clock()) / 60


def provenance(gpu: str | None = None) -> dict:
    """Where and from what this attempt ran: captured once at job start (a `git pull` mid-run must not change it)."""
    return {"attempt": int(os.environ.get("MEDERA_ATTEMPT") or 0) or None, "git_sha": git_sha(), "gpu": gpu,
            "job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname()}


def write_json_atomic(path: Path, obj, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent) + "\n")
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
