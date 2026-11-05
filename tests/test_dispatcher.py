"""hpc/dispatch.sh end to end against fake Slurm tools (sbatch, squeue, sacct, sinfo, scancel shims over a job table).

Each test copies the dispatcher scripts into a scratch repo, runs real bash on them and drives the fake cluster:
submission order and slot limits, resubmission and attempt counting, failures that are never resubmitted,
adoption after a restart, the dry run, the status table and the stop script."""
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

HPC = Path(__file__).resolve().parents[1] / "hpc"
SCRIPTS = ["dispatch.sh", "_dispatch_lib.sh", "dispatch_status.sh", "dispatch_stop.sh"]
ORDER = ["qwen:w1", "gemma_al:w1", "gemma_ex:w1", "qwen:w2", "gemma_al:w2", "gemma_ex:w2",
         "qwen:w3", "gemma_al:w3", "gemma_ex:w3"]
PARTS = "sharing,gpu-short,gpu-interactive"

SINFO = """c0160|sharing|(null)|(null)
c2204|gpu-short|gpu:v100-pcie:2(S:0-7)|zen,prod
d1026|gpu-short|gpu:a100:3|ib,zen2,prod
d1029|gpu-short|gpu:a100:4|zen2,prod,ib,a100@80g
d4041|sharing|gpu:h100:4|ib,cascadelake,prod
d4042|sharing|gpu:l40s:8|ib,zen2,prod
d4052|gpu-short|gpu:h200:8|cascadelake,prod
d4052|gpu-interactive|gpu:h200:8|cascadelake,prod
d4077|sharing|gpu:h200:8|(null)
"""
EXCLUDE = "c2204,d1026,d1029,d4042"          # every A100 is excluded, a100@80g too

# Job table: id|name|partition|state|exitcode|nodelist|comment. Only PENDING/RUNNING rows are "in the queue".
SHIMS = {
    "sbatch": r'''#!/usr/bin/env bash
db=$FAKE_SLURM
printf '%s\n' "$*" >> "$db/sbatch_calls"
if [ -f "$db/refuse" ]; then echo "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2; exit 1; fi
name='' part='' comment='' test=0
for a in "$@"; do
  case $a in
    --job-name=*) name=${a#*=} ;; --partition=*) part=${a#*=} ;; --comment=*) comment=${a#*=} ;; --test-only) test=1 ;;
  esac
done
if [ "$test" = 1 ]; then echo "sbatch: Job 4242 to start at 2026-11-05T00:00:00 using 8 processors on nodes d4041 in partition sharing" >&2; exit 0; fi
n=$(cat "$db/next_id" 2>/dev/null || echo 1000)
echo $((n + 1)) > "$db/next_id"
echo "$n|$name|$part|PENDING|0:0|None assigned|$comment" >> "$db/jobs"
echo "$n"
''',
    "squeue": r'''#!/usr/bin/env bash
db=$FAKE_SLURM name='' fmt='%i'
while [ $# -gt 0 ]; do case $1 in -n) name=$2; shift ;; -o) fmt=$2; shift ;; -u) shift ;; esac; shift; done
[ -f "$db/jobs" ] || exit 0
awk -F'|' -v name="$name" -v fmt="$fmt" '($4 == "PENDING" || $4 == "RUNNING") && (name == "" || $2 == name) {
  out = fmt; gsub(/%i/, $1, out); gsub(/%j/, $2, out); gsub(/%P/, $3, out); gsub(/%k/, $7, out); print out }' "$db/jobs"
''',
    "sacct": r'''#!/usr/bin/env bash
db=$FAKE_SLURM jid='' fields=State
while [ $# -gt 0 ]; do case $1 in -j) jid=$2; shift ;; -o) fields=$2; shift ;; esac; shift; done
[ -f "$db/jobs" ] || exit 0
awk -F'|' -v jid="$jid" -v fields="$fields" '$1 == jid { n = split(fields, f, ","); line = ""
  for (i = 1; i <= n; i++) {
    v = f[i] == "State" ? $4 : f[i] == "ExitCode" ? $5 : f[i] == "NodeList" ? $6 : f[i] == "Partition" ? $3 : "?"
    line = line (i > 1 ? "|" : "") v }
  print line }' "$db/jobs"
''',
    "sinfo": '#!/usr/bin/env bash\ncat "$FAKE_SLURM/sinfo"\n',
    "scancel": '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$FAKE_SLURM/scancel_calls"\n',
}


class Cluster:
    def __init__(self, root: Path):
        self.repo, self.db, self.bin = root / "repo", root / "slurm", root / "bin"
        (self.repo / "hpc").mkdir(parents=True)
        for s in SCRIPTS:
            shutil.copy(HPC / s, self.repo / "hpc" / s)
        (self.repo / "hpc" / "train.slurm").write_text("#!/bin/bash\n")
        (self.repo / "hpc" / "eval.slurm").write_text("#!/bin/bash\n")
        self.db.mkdir()
        (self.db / "sinfo").write_text(SINFO)
        self.bin.mkdir()
        for name, body in SHIMS.items():
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        # SLURM_CONF points nowhere, so even a real Slurm tool reached by mistake (e.g. on a cluster node) fails
        # instead of submitting or cancelling anything.
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "USER": "tester",
                    "FAKE_SLURM": str(self.db), "MEDERA_POLL_S": "0", "SLURM_CONF": str(root / "no-slurm.conf")}
        for tool in SHIMS:
            assert shutil.which(tool, path=self.env["PATH"]) == str(self.bin / tool), tool
        self.logs = self.repo / "runs" / "logs"

    def run(self, script="dispatch.sh", *args, rc=0):
        out = subprocess.run(["bash", str(self.repo / "hpc" / script), *args], env=self.env, capture_output=True,
                             text=True, timeout=60)
        assert out.returncode == rc, out.stdout + out.stderr
        return out.stdout

    def poll(self):
        return self.run("dispatch.sh", "--once")

    # fake cluster state
    def jobs(self) -> dict:
        f = self.db / "jobs"
        rows = [line.split("|") for line in f.read_text().splitlines()] if f.exists() else []
        return {r[0]: dict(zip(["id", "name", "part", "state", "exit", "node", "comment"], r)) for r in rows}

    def set_job(self, jid, state, exit="0:0", node="d4041", part="sharing"):
        rows = self.jobs()
        rows[jid].update(state=state, exit=exit, node=node, part=part)
        (self.db / "jobs").write_text("".join("|".join(r[k] for k in ["id", "name", "part", "state", "exit", "node",
                                                                          "comment"]) + "\n" for r in rows.values()))

    def add_foreign_job(self, jid, part, state="PENDING"):
        with open(self.db / "jobs", "a") as f:
            f.write(f"{jid}|other_job|{part}|{state}|0:0|None assigned|\n")

    def submits(self) -> list[str]:
        f = self.db / "sbatch_calls"
        return f.read_text().splitlines() if f.exists() else []

    def job_of(self, adapter) -> str:
        return self.state()[adapter][1]

    def state(self) -> dict:
        f = self.logs / "dispatch_state.tsv"
        rows = [line.split() for line in f.read_text().splitlines()] if f.exists() else []
        return {r[0]: (int(r[1]), r[2], r[3]) for r in rows}

    def finish(self, adapter, jid):
        arm, w = adapter.split(":")
        d = self.repo / "runs" / "adapters" / f"{arm}_{w}"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}")
        (d / "DONE").write_text("{}\n")
        self.set_job(jid, "COMPLETED")

    def log(self) -> str:
        return (self.logs / "dispatch.log").read_text()


def _name(call: str) -> str:
    return next(a.split("=", 1)[1] for a in call.split() if a.startswith("--job-name="))


@pytest.fixture
def cluster(tmp_path):
    return Cluster(tmp_path)


def test_first_poll_submits_four_in_interleaved_order_with_the_right_flags(cluster):
    cluster.poll()
    calls = cluster.submits()
    assert [_name(c) for c in calls] == ["medera_qwen_w1", "medera_gemma_al_w1", "medera_gemma_ex_w1", "medera_qwen_w2"]
    for c in calls:
        args = c.split()
        assert {"--parsable", f"--partition={PARTS}", "--gres=gpu:1", "--time=01:00:00", f"--exclude={EXCLUDE}",
                "--comment=medera_attempt=1", "--export=ALL,MEDERA_ATTEMPT=1", "hpc/train.slurm"} <= set(args)
        assert not any(a.startswith("--array") or a.startswith("--gres=gpu:h") for a in args)
    assert cluster.state() == {ad: (1, str(1000 + i), "submitted") for i, ad in enumerate(ORDER[:4])}
    cluster.poll()
    assert len(cluster.submits()) == 4                                       # full: nothing more


def test_a_freed_slot_takes_the_next_adapter_and_done_is_skipped(cluster):
    cluster.poll()
    cluster.finish("qwen:w1", "1000")
    cluster.poll()
    assert cluster.state()["qwen:w1"][2] == "done"
    assert [_name(c) for c in cluster.submits()][4:] == ["medera_gemma_al_w2"]
    assert "qwen:w1 DONE" in cluster.log()


@pytest.mark.parametrize("state,code", [("TIMEOUT", "0:0"), ("NODE_FAIL", "0:0"), ("PREEMPTED", "0:0"),
                                        ("FAILED", "3:0")])
def test_resumable_ends_are_resubmitted_as_the_next_attempt(cluster, state, code):
    cluster.poll()
    cluster.set_job("1002", state, code)
    cluster.poll()
    again = cluster.submits()[4]
    assert _name(again) == "medera_gemma_ex_w1"
    assert "--export=ALL,MEDERA_ATTEMPT=2" in again.split() and "--comment=medera_attempt=2" in again.split()
    assert cluster.state()["gemma_ex:w1"] == (2, "1004", "submitted")
    assert "resubmitting to resume" in cluster.log()


def test_gives_up_after_three_attempts(cluster):
    cluster.poll()
    for _ in range(3):
        cluster.set_job(cluster.job_of("gemma_ex:w1"), "TIMEOUT")
        cluster.poll()
    attempts, _, status = cluster.state()["gemma_ex:w1"]
    assert (attempts, status) == (3, "gave_up")
    assert [_name(c) for c in cluster.submits()].count("medera_gemma_ex_w1") == 3
    assert "after 3 attempts: giving up" in cluster.log()


@pytest.mark.parametrize("state,code,final", [("FAILED", "1:0", "failed"), ("OUT_OF_MEMORY", "0:125", "failed"),
                                              ("CANCELLED by 4711", "0:15", "cancelled")])
def test_python_errors_and_cancels_are_never_resubmitted(cluster, state, code, final):
    cluster.poll()
    (cluster.logs / "medera_qwen_w1_1000.out").write_text("Traceback (most recent call last):\nValueError: boom\n")
    cluster.set_job("1000", state, code)
    cluster.poll()
    assert cluster.state()["qwen:w1"] == (1, "1000", final)
    assert "medera_qwen_w1" not in [_name(c) for c in cluster.submits()][4:]
    assert _name(cluster.submits()[4]) == "medera_gemma_al_w2"               # the slot goes to the next adapter
    if final == "failed":
        assert "NOT resubmitted" in cluster.log() and "ValueError: boom" in cluster.log()


def test_completed_without_done_marker_waits_one_poll_then_fails(cluster):
    cluster.poll()
    cluster.set_job("1000", "COMPLETED")
    cluster.poll()
    assert cluster.state()["qwen:w1"][2] == "verify"
    cluster.poll()
    assert cluster.state()["qwen:w1"][2] == "failed"
    assert "medera_qwen_w1" not in [_name(c) for c in cluster.submits()][4:]


def test_small_gpu_exit_excludes_that_node_on_the_retry(cluster):
    cluster.poll()
    cluster.set_job("1001", "FAILED", "4:0", node="d4077")
    cluster.poll()
    retry = cluster.submits()[4]
    assert _name(retry) == "medera_gemma_al_w1" and f"--exclude={EXCLUDE},d4077" in retry.split()


def test_refused_submit_is_retried_without_using_an_attempt(cluster):
    (cluster.db / "refuse").write_text("")
    cluster.poll()
    assert cluster.state() == {} and "submit refused" in cluster.log()
    (cluster.db / "refuse").unlink()
    cluster.poll()
    assert cluster.state()["qwen:w1"] == (1, "1000", "submitted")


def test_slots_respect_every_partitions_qos_including_other_jobs(cluster):
    cluster.add_foreign_job("900", "sharing")                # one of the user's own non-MedEra jobs
    cluster.add_foreign_job("901", "gpu-interactive", state="RUNNING")
    cluster.poll()
    assert len(cluster.submits()) == 3                       # sharing and gpu-interactive: MaxSubmit 4 - 1


def test_restart_adopts_queued_jobs_instead_of_duplicating(cluster):
    cluster.poll()
    (cluster.logs / "dispatch_state.tsv").unlink()           # state lost (or never written)
    cluster.poll()
    assert len(cluster.submits()) == 4
    assert cluster.state() == {ad: (1, str(1000 + i), "submitted") for i, ad in enumerate(ORDER[:4])}
    assert cluster.log().count("adopted queued job") == 4


def test_crash_between_sbatch_and_state_write_is_adopted_with_its_attempt(cluster):
    cluster.poll()
    cluster.set_job("1000", "TIMEOUT")
    cluster.poll()                                            # qwen:w1 attempt 2 = job 1004
    lines = (cluster.logs / "dispatch_state.tsv").read_text().splitlines()
    (cluster.logs / "dispatch_state.tsv").write_text(
        "\n".join("qwen:w1 1 1000 retry" if line.startswith("qwen:w1 ") else line for line in lines) + "\n")
    cluster.poll()
    assert len(cluster.submits()) == 5 and cluster.state()["qwen:w1"] == (2, "1004", "submitted")


def test_loop_exits_when_every_adapter_is_final(cluster):
    for _ in range(3):                                        # 4 + 4 + 1 submissions, every job finishing
        cluster.poll()
        for ad, (_, job, status) in cluster.state().items():
            if status == "submitted":
                cluster.finish(ad, job)
    assert len(cluster.submits()) == 9
    cluster.run("dispatch.sh")                                # the real loop, MEDERA_POLL_S=0
    assert "all adapters final" in cluster.log() and not (cluster.logs / "dispatch.pid").exists()
    assert set(s for _, _, s in cluster.state().values()) == {"done"} and len(cluster.state()) == 9


def test_dry_run_prints_the_plan_and_changes_nothing(cluster):
    out = cluster.run("dispatch.sh", "--dry-run")
    assert cluster.submits() == [] and not (cluster.logs / "dispatch_state.tsv").exists()
    assert not (cluster.logs / "dispatch.pid").exists() and not list(cluster.logs.glob("*dry-run*"))
    plan = [line.split()[2] for line in out.splitlines() if line.strip().startswith("WOULD SUBMIT")]
    assert plan == ORDER[:4]
    nodes = out.split("allowed GPU nodes", 1)[1].split("plan for this poll:")[0].splitlines()
    allowed = nodes[1:nodes.index(next(n for n in nodes if n.startswith("excluded")))]
    assert [n.strip() for n in allowed] == ["d4041 h100", "d4052 h200", "d4077 h200"]
    assert "d1029 a100" in out and "d1026 a100" in out and "d4042 l40s" in out and "a100@80g" not in out
    assert "MEDERA_ATTEMPT=1" in out and "free submit slots: 4" in out


def test_test_only_runs_sbatch_test_only_with_the_dispatch_flags(cluster):
    out = cluster.run("dispatch.sh", "--test-only")
    (call,) = cluster.submits()
    assert call.split()[0] == "--test-only" and f"--exclude={EXCLUDE}" in call.split()
    assert _name(call) == "medera_qwen_w1" and "next adapter: qwen:w1" in out


def test_status_table(cluster):
    cluster.poll()
    cluster.set_job("1000", "RUNNING", node="d4052", part="gpu-short")
    ck = cluster.repo / "runs" / "checkpoints" / "qwen_w1"
    ck.mkdir(parents=True)
    (ck / "progress.json").write_text('{"step": 50, "total": 196, "attempt": 1, "job_id": "1000", "partition": '
                                      '"gpu-short", "node": "d4052", "gpu": "NVIDIA H200", "updated": "x"}')
    rows = {line.split()[0]: line.split() for line in cluster.run("dispatch_status.sh").splitlines()[1:]}
    assert rows["adapter"] == ["adapter", "attempts", "state", "job", "partition/node/gpu", "steps"]
    assert rows["qwen:w1"] == ["qwen:w1", "1", "RUNNING", "1000", "gpu-short/d4052/H200", "50/196"]
    assert rows["gemma_al:w1"] == ["gemma_al:w1", "1", "PENDING", "1001", "pending", "-"]
    assert rows["gemma_ex:w3"] == ["gemma_ex:w3", "0", "new", "-", "-", "-"]


def test_second_dispatcher_is_refused_and_stop_kills_only_the_loop(cluster):
    env = {**cluster.env, "MEDERA_POLL_S": "30"}
    loop = subprocess.Popen(["bash", str(cluster.repo / "hpc" / "dispatch.sh")], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pidfile = cluster.logs / "dispatch.pid"
    try:
        for _ in range(100):
            if pidfile.exists() and len(cluster.submits()) == 4:
                break
            time.sleep(0.1)
        assert pidfile.read_text().split()[1] == str(loop.pid)
        assert "already running" in cluster.run("dispatch.sh", rc=1)
        assert "already running" in cluster.run("dispatch.sh", "--once", rc=1)
        assert "dispatcher: running" in cluster.run("dispatch_status.sh")
        out = cluster.run("dispatch_stop.sh")
        assert loop.wait(timeout=10) == 0
    finally:
        if loop.poll() is None:
            loop.kill()
    assert f"stopped dispatcher pid {loop.pid}" in out and "scancel 1000 1001 1002 1003" in out
    assert not (cluster.db / "scancel_calls").exists()        # never cancels jobs itself
    assert not pidfile.exists() and "dispatcher stopped" in cluster.log()
    assert "dispatcher: not running" in cluster.run("dispatch_status.sh")
    assert "not running" in cluster.run("dispatch_stop.sh")


def test_stop_refuses_a_stale_pid_file(cluster):
    cluster.logs.mkdir(parents=True)
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
        (cluster.logs / "dispatch.pid").write_text(f"{host} {sleeper.pid}\n")
        assert "not the dispatcher" in cluster.run("dispatch_stop.sh", rc=1)
        assert sleeper.poll() is None                         # an unrelated process is never killed
    finally:
        sleeper.kill()


# ------------------------------------------------------------------------------------- eval phases (ppl, probe)
VARIANTS = ["qwen_base", "gemma_base", "qwen_w1", "gemma_al_w1", "gemma_ex_w1", "qwen_w2", "gemma_al_w2",
            "gemma_ex_w2", "qwen_w3", "gemma_al_w3", "gemma_ex_w3"]


def _train_all(cluster, adapters=ORDER):
    for ad in adapters:
        d = cluster.repo / "runs" / "adapters" / ad.replace(":", "_")
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}")
        (d / "DONE").write_text("{}\n")


def _eval_state(cluster, phase):
    f = cluster.logs / f"dispatch_{phase}_state.tsv"
    return {r[0]: (int(r[1]), r[2], r[3]) for r in (line.split() for line in f.read_text().splitlines())}


def _battery(cluster, status="draft", tag=None):
    (cluster.repo / "configs").mkdir(exist_ok=True)
    (cluster.repo / "configs" / "probes.yaml").write_text(f"status: {status}   # draft | frozen\n"
                                                        "prereg_tag: prereg-v1\n")
    if tag:
        git = ["git", "-C", str(cluster.repo), "-c", "user.name=t", "-c", "user.email=t@example.org",
               "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]
        for args in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "freeze"], ["tag", tag]):
            subprocess.run(git + args, check=True, capture_output=True)


def test_ppl_phase_submits_bases_first_through_eval_slurm(cluster):
    _train_all(cluster)
    cluster.run("dispatch.sh", "--phase", "ppl", "--once")
    calls = cluster.submits()
    assert [_name(c) for c in calls] == ["medera_ppl_" + v for v in VARIANTS[:4]]
    assert calls[0].split()[-3:] == ["hpc/eval.slurm", "ppl", "qwen_base"]
    for c in calls:
        assert {f"--partition={PARTS}", "--gres=gpu:1", "--time=01:00:00", f"--exclude={EXCLUDE}",
                "--export=ALL,MEDERA_ATTEMPT=1"} <= set(c.split())
    assert _eval_state(cluster, "ppl")["qwen_base"] == (1, "1000", "submitted")
    assert not (cluster.logs / "dispatch_state.tsv").exists()                 # the training state is untouched
    assert "qwen_base attempt 1 submitted" in (cluster.logs / "dispatch_ppl.log").read_text()


def test_ppl_of_an_untrained_adapter_is_blocked_and_done_comes_from_the_results_marker(cluster):
    _train_all(cluster, ORDER[:1])                                           # only qwen_w1 trained
    cluster.run("dispatch.sh", "--phase", "ppl", "--once")
    assert [_name(c) for c in cluster.submits()] == ["medera_ppl_qwen_base", "medera_ppl_gemma_base",
                                                      "medera_ppl_qwen_w1"]
    st = _eval_state(cluster, "ppl")
    assert st["gemma_ex_w3"] == (0, "-", "blocked")
    assert "gemma_al_w1 BLOCKED, not submitted: adapter runs/adapters/gemma_al_w1 is not trained" in \
        (cluster.logs / "dispatch_ppl.log").read_text()
    d = cluster.repo / "results" / "scores" / "perplexity" / "qwen_base"
    d.mkdir(parents=True)
    (d / "DONE").write_text("{}")
    cluster.set_job("1000", "COMPLETED")
    cluster.run("dispatch.sh", "--phase", "ppl", "--once")
    assert _eval_state(cluster, "ppl")["qwen_base"][2] == "done"
    _train_all(cluster, ORDER[1:2])                                          # gemma_al_w1 finishes training later
    cluster.run("dispatch.sh", "--phase", "ppl", "--once")
    assert _name(cluster.submits()[-1]) == "medera_ppl_gemma_al_w1"


def test_probe_phase_is_blocked_until_frozen_and_tagged_and_the_loop_exits(cluster):
    _train_all(cluster)
    _battery(cluster, status="draft")
    cluster.run("dispatch.sh", "--phase", "probe")                            # the loop, not --once: it must exit
    log = (cluster.logs / "dispatch_probe.log").read_text()
    assert cluster.submits() == [] and "stopping: 11 probe units are blocked" in log and "not frozen" in log
    _battery(cluster, status="frozen")
    cluster.run("dispatch.sh", "--phase", "probe", "--once")
    assert cluster.submits() == [] and "prereg-v1 does not exist" in (cluster.logs / "dispatch_probe.log").read_text()
    _battery(cluster, status="frozen", tag="prereg-v1")
    cluster.run("dispatch.sh", "--phase", "probe", "--once")
    calls = cluster.submits()
    assert [_name(c) for c in calls] == ["medera_probe_" + v for v in VARIANTS[:4]]
    assert calls[2].split()[-3:] == ["hpc/eval.slurm", "probe", "qwen_w1"]
    assert _eval_state(cluster, "probe")["qwen_w1"] == (1, "1002", "submitted")


def test_cap_counts_the_jobs_of_every_phase(cluster):
    _train_all(cluster)
    with open(cluster.db / "jobs", "a") as f:
        f.write("900|medera_qwen_w1|sharing|RUNNING|0:0|d4041|\n901|medera_probe_qwen_base|sharing|PENDING|0:0|x|\n")
    cluster.run("dispatch.sh", "--phase", "ppl", "--once")
    assert len(cluster.submits()) == 2


def test_eval_dry_run_test_only_and_status(cluster):
    _train_all(cluster)
    out = cluster.run("dispatch.sh", "--phase", "ppl", "--dry-run")
    plan = [line.split()[2] for line in out.splitlines() if line.strip().startswith("WOULD SUBMIT")]
    assert plan == VARIANTS[:4] and "hpc/eval.slurm ppl qwen_base" in out and cluster.submits() == []
    assert not (cluster.logs / "dispatch_ppl_state.tsv").exists() and not list(cluster.logs.glob("*dry-run*"))
    _battery(cluster, status="draft")
    out = cluster.run("dispatch.sh", "--phase", "probe", "--test-only")
    assert "phase probe, next unit: qwen_base" in out and "would NOT submit it yet" in out
    assert cluster.submits()[0].split()[0] == "--test-only" and _name(cluster.submits()[0]) == "medera_probe_qwen_base"
    cluster.run("dispatch.sh", "--phase", "ppl", "--once")
    cluster.set_job("1000", "RUNNING", node="d4052", part="gpu-short")
    prog = cluster.repo / "runs" / "eval" / "ppl" / "qwen_base"
    prog.mkdir(parents=True)
    (prog / "progress.json").write_text('{"step": 2, "total": 3, "attempt": 1, "job_id": "1000", "partition": '
                                        '"gpu-short", "node": "d4052", "gpu": "NVIDIA H200", "updated": "x"}')
    table = cluster.run("dispatch_status.sh", "--phase", "ppl").splitlines()[1:]
    rows = {line.split()[0]: line.split() for line in table}
    assert rows["ppl"] == ["ppl", "attempts", "state", "job", "partition/node/gpu", "progress"]
    assert rows["qwen_base"] == ["qwen_base", "1", "RUNNING", "1000", "gpu-short/d4052/H200", "2/3"]
    assert rows["gemma_ex_w3"] == ["gemma_ex_w3", "0", "new", "-", "-", "-"]
    cluster.run("dispatch.sh", "--phase", "nope", rc=2)
