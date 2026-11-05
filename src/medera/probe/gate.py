"""Pre-registration gate (SCOPE §9): no trained adapter is probed before the pre-registration is approved, committed
and git-tagged, with the probe battery frozen in the same tag.

The gate passes only if all of these hold in the repo the job runs from:
  1. configs/probes.yaml has `status: frozen`;
  2. the tag named by its `prereg_tag` exists and HEAD contains it (git merge-base --is-ancestor);
  3. the tagged commit contains configs/probes.yaml and docs/preregistration.md, and neither differs from the tag in
     the working tree (committed or not), so what is probed is exactly what was registered;
  4. docs/preregistration.md has a line "Status: approved".
Perplexity does not use this gate (it may run before the tag).
"""
import re
import subprocess
from pathlib import Path

PROBES_REL = "configs/probes.yaml"
PREREG_REL = "docs/preregistration.md"
_APPROVED = re.compile(r"^\W*status\W*:?\W*approved\b", re.IGNORECASE | re.MULTILINE)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def check(repo: Path, cfg: dict) -> tuple[bool, list[str], dict]:
    """(ok, reasons it fails, info: tag and its commit). Never raises for a missing tag or repo."""
    reasons, info = [], {"tag": cfg.get("prereg_tag"), "tag_commit": None}
    if cfg.get("status") != "frozen":
        reasons.append(f"{PROBES_REL} has status {cfg.get('status')!r}, not 'frozen'")
    tag = cfg.get("prereg_tag")
    if not tag:
        return False, reasons + ["no prereg_tag in the battery"], info
    r = _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
    if r.returncode != 0:
        reasons.append(f"git tag {tag!r} does not exist here (create and push it, then git fetch --tags)")
        return False, reasons, info
    info["tag_commit"] = r.stdout.strip()
    if _git(repo, "merge-base", "--is-ancestor", info["tag_commit"], "HEAD").returncode != 0:
        reasons.append(f"HEAD does not contain tag {tag!r}")
    for rel in (PROBES_REL, PREREG_REL):
        if _git(repo, "cat-file", "-e", f"{info['tag_commit']}:{rel}").returncode != 0:
            reasons.append(f"tag {tag!r} does not contain {rel}")
        elif _git(repo, "diff", "--quiet", info["tag_commit"], "--", rel).returncode != 0:
            reasons.append(f"{rel} differs from tag {tag!r} (the frozen version must be probed)")
    prereg = repo / PREREG_REL
    if not prereg.exists() or not _APPROVED.search(prereg.read_text(encoding="utf-8")):
        reasons.append(f"{PREREG_REL} has no 'Status: approved' line")
    return not reasons, reasons, info
