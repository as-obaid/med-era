"""Training reports on CPU from synthetic run outputs: T2 audit table and copies, training summary and GPU-hours,
the loss figure, and the publish sanitizer they rely on."""
import csv
import json
import math

import pytest

from medera import publish
from medera.config import load_yaml
from medera.train import report

GROUPS = ("attention", "dense_mlp", "router", "experts", "vision", "other")
PRIVATE = ("job_id", "node", "partition", "n0001", "jobid-")


@pytest.fixture
def train_cfg(repo_root):
    return load_yaml(repo_root / "configs" / "train.yaml")


def _audit(arm, window, experts=0, vision=0):
    g = {k: {"lora_tensors": 0, "lora_params": 0, "lora_nonzero_grad": 0, "base_params": 100, "base_dtypes": {}}
         for k in GROUPS}
    g["attention"].update(lora_tensors=4, lora_params=4000, lora_nonzero_grad=4)
    g["dense_mlp"].update(lora_tensors=2, lora_params=2000, lora_nonzero_grad=2)
    g["experts"].update(lora_tensors=experts, lora_params=experts * 1000, lora_nonzero_grad=experts)
    g["vision"].update(lora_tensors=vision, lora_params=vision * 10, lora_nonzero_grad=vision)
    total = {k: sum(x[k] for x in g.values()) for k in ("lora_tensors", "lora_params", "lora_nonzero_grad",
                                                         "base_params")}
    return {"arm": arm, "expect_experts": arm == "gemma_ex", "groups": g, **total, "window": window,
            "precision": "bf16", "model": "m"}


def _train(arm, window, attempts, n_blocks=40, steps=10, step_s=36.0, load_s=120.0):
    losses = [2.0 - 0.05 * i for i in range(steps)]
    return {"arm": arm, "window": window, "config": {"epochs": 2, "n_blocks": n_blocks, "micro_batch": 1,
                                                      "grad_accum": 8, "total_steps": steps},
            "gpu": attempts[-1]["gpu"], "git_sha": "abc1234def", "attempts": attempts, "load_s": load_s,
            "steps": steps, "total_steps": steps, "losses": losses, "step_times": [step_s] * steps,
            "peak_mem_gb": 60.5, "stopped": "done", "audit_failures": []}


def _attempt(i, start, end, gpu="NVIDIA H100 80GB HBM3"):
    return {"attempt": i, "git_sha": "abc1234def", "gpu": gpu, "job_id": f"jobid-{i}", "partition": "sharing",
            "node": "n0001", "start_step": start, "end_step": end, "stopped": "done"}


def _runs(tmp_path, train_cfg):
    runs, results = tmp_path / "runs", tmp_path / "results"
    (runs / "results" / "adapter_audit").mkdir(parents=True)
    (runs / "results" / "train").mkdir(parents=True)
    (runs / "logs").mkdir(parents=True)
    for name in report.adapter_names(train_cfg):
        arm, window = name.rsplit("_", 1)
        audit = _audit(arm, window, experts=3 if arm == "gemma_ex" else 0)
        audit["job_id"] = "jobid-x"                       # a private key in an audit must not reach results/
        (runs / "results" / "adapter_audit" / f"{name}.json").write_text(json.dumps(audit, indent=2))
        if name == "qwen_w1":                             # two attempts; attempt 1's load time is in its log
            att = [_attempt(1, 0, 4), _attempt(2, 4, 10, gpu="NVIDIA H200")]
            (runs / "logs" / f"medera_{name}_jobid-1.out").write_text("[train] loaded in 600s; fla fast path\n")
        elif name == "gemma_al_w2":                       # two attempts, attempt 1's log missing
            att = [_attempt(1, 0, 6), _attempt(2, 6, 10)]
        else:
            att = [_attempt(1, 0, 10)]
        (runs / "results" / "train" / f"{name}.json").write_text(json.dumps(_train(arm, window, att)))
        cell = results / "scores" / "perplexity" / name / f"{window}.json"
        cell.parent.mkdir(parents=True, exist_ok=True)
        cell.write_text(json.dumps({"mean_nll": 1.5}))
    return runs, results


def _csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_audit_table_and_sanitized_copies(tmp_path, train_cfg):
    pytest.importorskip("torch")                          # audit_adapter.check lives next to the torch audit code
    runs, results = _runs(tmp_path, train_cfg)
    bad = _audit("gemma_al", "w3", vision=2)              # a vision tensor adapted: the gate must fail
    (runs / "results" / "adapter_audit" / "gemma_al_w3.json").write_text(json.dumps(bad))
    rows = {r["adapter"]: r for r in report.write_audit(runs, results, report.adapter_names(train_cfg))}
    assert len(rows) == 9 and rows["qwen_w1"]["audit"] == "PASS" and rows["gemma_ex_w1"]["experts_tensors"] == 3
    assert rows["gemma_al_w3"]["audit"] == "FAIL" and "vision" in rows["gemma_al_w3"]["failures"]
    csv_rows = _csv(results / "tables" / "T2_adapter_audit.csv")
    assert {f"{g}_tensors" for g in GROUPS} <= set(csv_rows[0]) and len(csv_rows) == 9
    md = (results / "tables" / "T2_adapter_audit.md").read_text()
    assert "| qwen_w1 | 4 / 4K | 2 / 2K | 0 | 0 | 0 | 0 | 6 | 6,000 |" in md and "FAIL: 2 vision" in md
    copies = sorted(p.name for p in (results / "adapter_audit").iterdir())
    assert len(copies) == 9
    for p in (results / "adapter_audit").iterdir():
        assert "job_id" not in json.loads(p.read_text())


def test_training_summary_gpu_hours_and_privacy(tmp_path, train_cfg):
    runs, results = _runs(tmp_path, train_cfg)
    rows = {r["adapter"]: r for r in report.write_summary(runs, results, report.adapter_names(train_cfg))}
    q = rows["qwen_w1"]
    assert q["attempts"] == 2 and q["step_spans"] == "0-4 4-10" and q["spans_contiguous"] is True
    assert q["gpu_types"] == "H100/H200" and q["load_estimated"] is False
    assert q["gpu_hours"] == pytest.approx((10 * 36 + 600 + 120) / 3600)      # steps + log load + final load
    g = rows["gemma_al_w2"]
    assert g["load_estimated"] is True and g["gpu_hours"] == pytest.approx((10 * 36 + 2 * 120) / 3600)
    single = rows["gemma_ex_w3"]
    assert single["gpu_hours"] == pytest.approx((10 * 36 + 120) / 3600) and single["gpu_types"] == "H100"
    assert single["loss_first10"] == pytest.approx(sum(2.0 - 0.05 * i for i in range(10)) / 10)
    assert single["epoch1_steps"] == 5 and single["loss_epoch1_last10"] == pytest.approx(
        sum(2.0 - 0.05 * i for i in range(5)) / 5)
    assert single["heldout_nll_own_era"] == 1.5 and single["git_sha"] == "abc1234def"
    total = rows["total"]
    assert total["gpu_hours"] == pytest.approx(math.fsum(r["gpu_hours"] for k, r in rows.items() if k != "total"))
    assert total["attempts"] == 11 and total["load_estimated"] is True
    for name in ("training_summary.csv", "training_summary.md"):
        text = (results / "tables" / name).read_text()
        assert not any(s in text for s in PRIVATE), name
    md = (results / "tables" / "training_summary.md").read_text()
    assert "| **total** |" in md and "estimated from the last attempt" in md and "contiguous" in md


def test_epoch_boundary_in_steps():
    cfg = {"n_blocks": 787, "micro_batch": 1, "grad_accum": 8}
    assert report.epoch1_steps(cfg) == 98 and report.epoch2_start(cfg) == pytest.approx(98.375)
    assert report.epoch1_steps({**cfg, "n_blocks": 756}) == 94


def test_gpu_type_and_smoothing():
    assert report.gpu_type("NVIDIA H100 80GB HBM3") == "H100" and report.gpu_type("NVIDIA H200") == "H200"
    assert report.gpu_type(None) == "unknown"
    s = report.smooth([0, 0, 9, 0, 0], k=3)
    assert list(s) == pytest.approx([0, 3, 3, 3, 0])


def test_loss_figure_and_cli(tmp_path, train_cfg, monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("torch")
    runs, results = _runs(tmp_path, train_cfg)
    paths = report.write_loss_figure(runs, results, report.adapter_names(train_cfg))
    assert [p.name for p in paths] == ["training_loss.png", "training_loss.svg"]
    assert all(p.stat().st_size > 1000 for p in paths) and "Muhammad Obaidullah" in paths[1].read_text()
    assert report.main(["--runs-dir", str(runs), "--results-dir", str(results)]) == 0
    for name in ("T2_adapter_audit.md", "training_summary.md"):
        assert (results / "tables" / name).exists()


# ------------------------------------------------------------------------------------------------ publish

def test_scrub_and_leaks():
    raw = {"a": 1, "provenance": {"job_id": "1", "node": "n0001", "partition": "gpu", "gpu": "H100"},
           "list": [{"hostname": "x", "keep": 2}]}
    clean = publish.scrub(raw)
    assert clean == {"a": 1, "provenance": {"gpu": "H100"}, "list": [{"keep": 2}]}
    assert set(publish.leaks(raw)) >= {"provenance.job_id", "provenance.node", "list[0].hostname"}
    assert publish.leaks(clean) == []
    assert publish.leaks({"path": "/scratch/someone/med-era/runs"}) and publish.leaks({"n": "n0002"})
    assert publish.leaks({"sha": "n0001aa", "model": "Qwen/Qwen3.5-27B", "rel": "runs/adapters/x"}) == []


def test_sanitize_in_place_backs_up_and_is_idempotent(tmp_path):
    root, backup = tmp_path / "results" / "scores", tmp_path / "runs" / "scores_raw"
    cell = root / "perplexity" / "qwen_base" / "w1.json"
    cell.parent.mkdir(parents=True)
    raw = json.dumps({"ppl": 4.5, "provenance": {"job_id": "123", "node": "n0001", "gpu": "H100"}}, indent=2) + "\n"
    cell.write_text(raw)
    done = cell.parent / "DONE"
    done.write_text(json.dumps({"variant": "qwen_base", "git_sha": "abc"}) + "\n")
    out = publish.sanitize_in_place(root, backup)
    assert out["checked"] == 2 and out["sanitized"] == 1
    assert json.loads(cell.read_text()) == {"ppl": 4.5, "provenance": {"gpu": "H100"}}
    assert cell.read_text().startswith("{\n  ")                       # layout kept
    assert (backup / "perplexity" / "qwen_base" / "w1.json").read_text() == raw
    assert publish.sanitize_in_place(root, backup)["sanitized"] == 0  # second run: nothing to do, backup intact
    assert (backup / "perplexity" / "qwen_base" / "w1.json").read_text() == raw


def test_sanitize_refuses_paths_it_cannot_remove(tmp_path):
    root = tmp_path / "scores"
    (root / "x").mkdir(parents=True)
    (root / "x" / "w1.json").write_text(json.dumps({"adapter_dir": "/scratch/someone/runs/adapters/a"}))
    with pytest.raises(ValueError, match="private values"):
        publish.sanitize_in_place(root, tmp_path / "backup")
    assert publish.main(["--results-dir", str(tmp_path / "nothing"), "--runs-dir", str(tmp_path)]) == 0
