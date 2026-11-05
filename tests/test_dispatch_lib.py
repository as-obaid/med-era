"""hpc/_dispatch_lib.sh: the dispatcher's pure-bash decisions (the login node can't run Python)."""
import subprocess
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "hpc" / "_dispatch_lib.sh"

# Real `sinfo -h -N -p sharing,gpu-short,gpu-interactive -o "%N|%P|%G|%f"` lines (2026-11-04), trimmed.
SINFO = """c0160|sharing|(null)|(null)
c2204|gpu-short|gpu:v100-pcie:2(S:0-7)|zen,prod
d1026|gpu-short|gpu:a100:3|ib,zen2,prod
d1028|gpu-interactive|gpu:a100:4|zen2,prod,ib,a100@80g
d1028|gpu-short|gpu:a100:4|zen2,prod,ib,a100@80g
d3146|sharing|gpu:a100:8|zen2,prod,dgx
d4041|sharing|gpu:h100:4|ib,cascadelake,prod
d4042|sharing|gpu:l40s:8|ib,zen2,prod
d4052|gpu-short|gpu:h200:8|cascadelake,prod
d4052|gpu-interactive|gpu:h200:8|cascadelake,prod
d4077|sharing|gpu:h200:8|(null)
"""


def bash(script: str, stdin: str = "") -> str:
    out = subprocess.run(["bash", "-c", f'source "{LIB}"; {script}'], input=stdin, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_gpu_allow_and_exclude_lists():
    assert bash("gpu_nodes allow", SINFO) == "d1028 a100@80g\nd4041 h100\nd4052 h200\nd4077 h200"
    assert bash("gpu_nodes exclude", SINFO) == "c2204 v100-pcie\nd1026 a100\nd3146 a100\nd4042 l40s"
    assert bash("exclude_csv", SINFO) == "c2204,d1026,d3146,d4042"          # CPU-only nodes never get gpu:1


@pytest.mark.parametrize("state,code,decision", [
    ("PENDING", "0:0", "active"), ("RUNNING", "0:0", "active"), ("REQUEUED", "0:0", "active"),
    ("COMPLETED", "0:0", "completed"),
    ("TIMEOUT", "0:0", "resume"), ("NODE_FAIL", "0:0", "resume"), ("PREEMPTED", "0:0", "resume"),
    ("FAILED", "3:0", "resume"),            # wall-time guard: checkpoint saved, resume
    ("FAILED", "4:0", "resume"),            # landed on a GPU under 79 GB
    ("FAILED", "1:0", "failed"),            # Python error or audit failure: never resubmitted
    ("OUT_OF_MEMORY", "0:125", "failed"),
    ("CANCELLED", "0:0", "cancelled"), ("CANCELLED by 123", "0:15", "cancelled"),
])
def test_classify(state, code, decision):
    assert bash(f'classify "{state}" "{code}"') == decision


def test_adapter_order_interleaves_arms_per_window():
    assert bash("adapters").split() == ["qwen:w1", "gemma_al:w1", "gemma_ex:w1", "qwen:w2", "gemma_al:w2",
                                        "gemma_ex:w2", "qwen:w3", "gemma_al:w3", "gemma_ex:w3"]


def test_state_file_roundtrip(tmp_path):
    f = tmp_path / "state.tsv"
    out = bash(f'STATE="{f}"; state_set qwen:w1 1 111 submitted; state_set gemma_ex:w1 2 222 submitted; '
               f'state_set qwen:w1 2 333 submitted; state_get qwen:w1; state_get gemma_ex:w1; state_get gemma_al:w1')
    assert out.splitlines() == ["2 333 submitted", "2 222 submitted", "0 - new"]
    assert len(f.read_text().splitlines()) == 2


@pytest.mark.parametrize("queue,ours,room", [
    ("", 0, 4),                                                   # empty queue: our cap
    ("sharing,gpu-short,gpu-interactive\n" * 4, 4, 0),             # 4 pending adapter jobs: full
    ("gpu-short\ngpu-short\n", 2, 2),                              # 2 running in gpu-short: cap 4 - 2
    ("sharing\nsharing\nsharing\n", 0, 1),                         # other jobs in sharing: its MaxSubmit 4 - 3
    ("gpu-interactive\n" * 4, 0, 0),                               # gpu-interactive full blocks every new job
    ("gpu-short\n" * 8, 0, 0),
])
def test_submit_room_takes_the_tightest_limit(queue, ours, room):
    assert bash(f'submit_room 4 {ours} sharing,gpu-short,gpu-interactive', queue) == str(room)


def test_attempt_of_comment_and_json_field(tmp_path):
    assert bash('attempt_of_comment "medera_attempt=2"') == "2"
    assert bash('attempt_of_comment "(null)"') == ""
    f = tmp_path / "progress.json"
    f.write_text('{"step": 25, "total": 189, "attempt": 2, "job_id": "11700001", "partition": "sharing", '
                 '"node": "d4077", "gpu": "NVIDIA H200", "updated": "2026-11-05 01:02:03"}')
    got = [bash(f'json_field "{f}" {k}') for k in ("step", "total", "job_id", "gpu", "node", "partition", "updated")]
    assert got == ["25", "189", "11700001", "NVIDIA H200", "d4077", "sharing", "2026-11-05 01:02:03"]
    assert bash(f'json_field "{tmp_path / "missing.json"}" step') == ""
