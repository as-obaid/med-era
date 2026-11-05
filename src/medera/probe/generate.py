"""Probe one variant with the frozen battery (SCOPE §7): completion-style prompts, no chat template.

    python -m medera.probe.generate --plan                        # CPU: battery check and chunk plan
    python -m medera.probe.generate --check-gate --variant qwen_w1   # CPU: is this variant allowed to be probed?
    python -m medera.probe.generate --variant qwen_w1             # GPU: every missing chunk (resumable)
    python -m medera.probe.generate --variant qwen_base --preview # GPU: a base model on a draft battery -> runs/

Every wording is wrapped in the neutral primer and template from configs/probes.yaml and completed by the variant:
the sampled pass once per seed (primary) and one greedy pass (descriptive), with exactly the decoding settings of the
battery (the model's own generation defaults are replaced, not merged). An answer ends at the first stop string
("\\nQuestion:"), the document separator or the tokenizer's end of sequence, or after max_new_tokens.

Output: results/answers/<variant>/<chunk>.jsonl, one record per answer with the variant, probe, paraphrase, seed,
decoding settings, battery sha256, git sha, model revision, adapter sha256, prereg tag and the job's GPU/node.
A chunk is written atomically; a rerun skips chunks already written with the same battery and adapter, and refuses
(exit 2) when it finds answers from a different battery. DONE is written when every chunk exists.

Pre-registration gate (medera.probe.gate): trained adapters are refused (exit 2) unless the battery is frozen and
the pre-registration tag exists and matches. Base models without the gate only run with --preview, which writes to
runs/probe_preview/ (never results/), for reviewing the draft battery.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from medera.config import load_yaml
from medera.jobs import (EXIT_FAIL, EXIT_OK, EXIT_REFUSED, EXIT_SMALL_GPU, EXIT_WALL, Stopper, env_versions,
                         provenance, write_json_atomic, write_progress, write_text_atomic)
from medera.probe import gate
from medera.probe.battery import Chunk, battery_sha256, build_prompt, gen_settings, plan_chunks, validate, wordings
from medera.variants import Variant, adapter_info, get_variant, load_variant

DEFAULT_CHUNK_S = 300   # first-chunk estimate: 400 new tokens for 24 wordings, before any chunk is timed
LOAD_EST_S = 600        # loading a 52-54 GB bf16 base took 2-6 min on the H100s; 10 min with margin
STOP_WINDOW = 16        # last tokens decoded each step to look for a stop string


class Refused(RuntimeError):
    """A condition that must not be retried or overwritten (exit 2)."""


def stop_criteria(tok, stops: list[str], prompt_len: int):
    """Per-row stopping on any stop string in the last STOP_WINDOW generated tokens (transformers marks each row done
    separately; finished rows are padded until the whole batch ends)."""
    import torch
    from transformers import StoppingCriteria

    class StopOnStrings(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            tail = input_ids[:, max(prompt_len, input_ids.shape[1] - STOP_WINDOW):]
            done = [any(s in tok.decode(row, skip_special_tokens=True) for s in stops) for row in tail.tolist()]
            return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

    return StopOnStrings()


def eos_ids(tok, model, model_cfg: dict) -> list[int]:
    """End of an answer: the document separator the adapters were trained with, the tokenizer's eos and the model's
    own generation eos ids (collected before the generation config is replaced)."""
    ids = []
    sep = model_cfg.get("doc_separator")
    if sep:
        ids.append(tok.convert_tokens_to_ids(sep))
    ids.append(tok.eos_token_id)
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids += gen_eos if isinstance(gen_eos, list) else [gen_eos]
    return sorted({int(i) for i in ids if i is not None})


def split_answer(ids: list[int], tok, stops: list[str], eos: set[int], pad_id: int, max_new: int) -> dict:
    end = next((j for j, t in enumerate(ids) if t in eos), None)
    body = ids[:end] if end is not None else list(ids)
    while body and body[-1] == pad_id:
        body.pop()
    raw = tok.decode(body, skip_special_tokens=True)
    cut = min((raw.find(s) for s in stops if s in raw), default=-1)
    text = (raw[:cut] if cut >= 0 else raw).strip()
    finish = "stop" if cut >= 0 else "eos" if end is not None else "length" if len(body) >= max_new else "other"
    return {"text": text, "text_raw": raw, "finish": finish, "n_new_tokens": len(body)}


def generate_chunk(model, tok, cfg: dict, chunk: Chunk, eos: list[int]) -> list[dict]:
    """One generate() call for the chunk's wordings, seeded with chunk.torch_seed, left padding."""
    import torch
    from transformers import GenerationConfig, StoppingCriteriaList

    settings = gen_settings(cfg, chunk.mode)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    gc = GenerationConfig(**settings, num_beams=1, use_cache=True, eos_token_id=eos, pad_token_id=pad_id)
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner.generation_config = gc        # no model-specific defaults (top_k 20 / 64) can leak into the settings
    prompts = [build_prompt(cfg, w.question) for w in chunk.wordings]
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    n_prompt = enc["input_ids"].shape[1]
    torch.manual_seed(chunk.torch_seed)
    with torch.inference_mode():
        out = model.generate(**enc, generation_config=gc,
                             stopping_criteria=StoppingCriteriaList([stop_criteria(tok, cfg["stop"], n_prompt)]))
    new = out[:, n_prompt:].tolist()
    return [split_answer(new[i], tok, cfg["stop"], set(eos), pad_id, settings["max_new_tokens"])
            for i in range(len(chunk.wordings))]


def chunk_status(path: Path, chunk: Chunk, v: Variant, bsha: str, adapter_sha: str | None) -> str:
    """'missing', 'current' (skip), or raises Refused when a different battery, adapter or variant wrote it."""
    if not path.exists():
        return "missing"
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [r for r in recs if (r.get("battery_sha256"), r.get("adapter_sha256"), r.get("variant"), r.get("chunk"))
           != (bsha, adapter_sha, v.name, chunk.id)]
    if bad or len(recs) != len(chunk.wordings):
        raise Refused(f"{path} was written by another battery, adapter or plan ({len(recs)} records, {len(bad)} "
                      "mismatched); move that variant's answers aside before probing again")
    return "current"


def run_variant(v: Variant, cfg: dict, model_and_tok, out_dir: Path, stopper: Stopper, prov: dict, meta: dict,
                model_cfg: dict, progress: Path | None = None, log=print) -> str:
    """Generate every missing chunk of one variant. model_and_tok() loads (model, tokenizer) only when needed.
    Returns "done" or "wall"; raises Refused on a battery/adapter mismatch."""
    bsha, adapter_sha = battery_sha256(cfg), meta.get("adapter_sha256")
    done_file = out_dir / "DONE"
    if done_file.exists() and json.loads(done_file.read_text()).get("battery_sha256") != bsha:
        raise Refused(f"{done_file} belongs to another battery; move {out_dir} aside before probing again")
    plan = plan_chunks(cfg)
    todo = [c for c in plan if chunk_status(out_dir / f"{c.id}.jsonl", c, v, bsha, adapter_sha) == "missing"]
    log(f"[probe] {v.name}: {len(plan) - len(todo)}/{len(plan)} chunks already done; {len(todo)} to do")
    model = tok = eos = None
    longest, done_n = 0.0, len(plan) - len(todo)
    for c in todo:
        est = (1.2 * longest if longest else DEFAULT_CHUNK_S) + (LOAD_EST_S if model is None else 0)
        if not stopper.fits(est):
            log(f"[probe] stopping before {c.id}: ~{est / 60:.1f} min needed, {stopper.minutes_left():.1f} min left "
                "(or SIGTERM)")
            return "wall"
        if model is None:
            model, tok = model_and_tok()
            eos = eos_ids(tok, model, model_cfg)
            log(f"[probe] eos ids {eos}, pad id {tok.pad_token_id}")
        t0 = time.time()
        answers = generate_chunk(model, tok, cfg, c, eos)
        secs = time.time() - t0
        longest = max(longest, secs)
        settings = gen_settings(cfg, c.mode)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        for k, (w, a) in enumerate(zip(c.wordings, answers)):
            lines.append(json.dumps({
                "variant": v.name, "model": v.model, "arm": v.arm, "window": v.window,
                "probe_id": w.probe_id, "dimension": w.dimension, "paraphrase_id": w.paraphrase_id,
                "question": w.question, "mode": c.mode, "seed": c.seed, "chunk": c.id, "chunk_seed": c.torch_seed,
                "index_in_chunk": k, **a, "decoding": settings, "batch_size": cfg["decoding"]["batch_size"],
                "stop": cfg["stop"], "eos_ids": eos, "battery_sha256": bsha, **meta, "provenance": prov,
                "chunk_seconds": round(secs, 2), "created": now}, ensure_ascii=False))
        write_text_atomic(out_dir / f"{c.id}.jsonl", "\n".join(lines) + "\n")
        done_n += 1
        write_progress(progress, {"step": done_n}, len(plan), prov)
        fin = {f: sum(a["finish"] == f for a in answers) for f in ("stop", "eos", "length", "other")}
        log(f"[probe] {v.name} {c.id}: {len(answers)} answers in {secs:.0f}s, finish {fin}")
    write_json_atomic(done_file, {"variant": v.name, "chunks": len(plan), "answers": sum(len(c.wordings) for c in plan),
                                  "battery_sha256": bsha, "prereg": meta.get("prereg"), "git_sha": prov.get("git_sha"),
                                  "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=None)
    return "done"


def load_tokenizer(model_cfg: dict):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_cfg["hf_id"], revision=model_cfg["revision"])


def main(argv=None) -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant")
    ap.add_argument("--plan", action="store_true", help="CPU: validate the battery and print the chunk plan")
    ap.add_argument("--check-gate", action="store_true", help="CPU: report whether --variant may be probed")
    ap.add_argument("--preview", action="store_true",
                    help="base models only: allow a draft battery, write to runs/probe_preview/ (never results/)")
    ap.add_argument("--probes-config", default="configs/probes.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--repo", default=".", help="git checkout the gate inspects")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.probes_config)
    problems = validate(cfg)
    if problems:
        print("[probe] battery is not usable:\n  - " + "\n  - ".join(problems), flush=True)
        return EXIT_REFUSED
    plan = plan_chunks(cfg)
    if args.plan:
        ws = wordings(cfg)
        n_s = sum(len(c.wordings) for c in plan if c.mode == "sampled")
        print(f"[probe] battery {battery_sha256(cfg)[:12]} status={cfg['status']} tag={cfg['prereg_tag']}: "
              f"{len(cfg['probes'])} probes, {len(ws)} wordings; {len(plan)} chunks per variant "
              f"(batch {cfg['decoding']['batch_size']}); {n_s} sampled + {len(ws)} greedy answers per variant")
        for c in plan:
            print(f"  {c.id:18s} {c.mode:7s} seed={c.seed} torch_seed={c.torch_seed} wordings={len(c.wordings)}")
        return EXIT_OK
    if not args.variant:
        ap.error("--variant is required (or use --plan)")
    train_cfg = load_yaml(args.train_config)
    v = get_variant(train_cfg, args.variant)
    ok, reasons, info = gate.check(Path(args.repo), cfg)
    out_dir = Path(args.results_dir) / "answers" / v.name
    if args.preview and not v.is_base:
        print(f"[probe] REFUSED: --preview is for base models only; {v.name} is a trained adapter", flush=True)
        return EXIT_REFUSED
    if args.preview:
        out_dir = Path(args.runs_dir) / "probe_preview" / v.name
    elif not ok:
        kind = "trained adapter" if not v.is_base else "base model (use --preview to try a draft battery)"
        print(f"[probe] REFUSED {v.name} ({kind}): the pre-registration gate is closed:\n  - " + "\n  - ".join(reasons),
              flush=True)
        return EXIT_REFUSED
    print(f"[probe] gate {'open' if ok else 'closed (preview)'} for {v.name}: tag {info['tag']} "
          f"{(info['tag_commit'] or '-')[:12]}; output {out_dir}", flush=True)
    if args.check_gate:
        return EXIT_OK

    import torch

    from medera.train.lora_cpt import MIN_GPU_BYTES, gpu_big_enough

    model_cfg = load_yaml(args.models_config)[v.model]
    runs = Path(args.runs_dir)
    meta = {"model_hf_id": model_cfg["hf_id"], "model_revision": model_cfg["revision"], "precision": "bf16",
            "adapter": None, "adapter_sha256": None, **adapter_info(runs, v),
            "prereg": {"tag": info["tag"], "commit": info["tag_commit"]} if ok else None, "preview": bool(args.preview)}
    meta.pop("adapter_done", None)
    prov = provenance(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    prov["versions"] = env_versions()
    print(f"[probe] variant={v.name} model={model_cfg['hf_id']}@{model_cfg['revision'][:8]} "
          f"adapter={meta['adapter'] or '-'} battery={battery_sha256(cfg)[:12]} attempt={prov['attempt']} "
          f"gpu={prov['gpu']} node={prov['node']} commit={prov['git_sha'][:7]}", flush=True)
    if torch.cuda.is_available() and not gpu_big_enough(torch.cuda.get_device_properties(0).total_memory):
        print(f"[probe] ERROR: {prov['gpu']} is under {MIN_GPU_BYTES / 1e9:.0f} GB. Exit {EXIT_SMALL_GPU}.", flush=True)
        return EXIT_SMALL_GPU
    torch.use_deterministic_algorithms(True, warn_only=True)
    stopper = Stopper(t_start, train_cfg["defaults"]).install()
    print(f"[probe] stop deadline in {stopper.minutes_left():.0f} min", flush=True)

    def loader():
        t0 = time.time()
        tok = load_tokenizer(model_cfg)
        model = load_variant(v, model_cfg, runs)
        print(f"[probe] loaded {v.name} in {time.time() - t0:.0f}s", flush=True)
        return model, tok

    try:
        status = run_variant(v, cfg, loader, out_dir, stopper, prov, meta, model_cfg,
                             progress=runs / "eval" / "probe" / v.name / "progress.json",
                             log=lambda m: print(m, flush=True))
    except Refused as e:
        print(f"[probe] REFUSED: {e}", flush=True)
        return EXIT_REFUSED
    return EXIT_OK if status == "done" else EXIT_WALL if status == "wall" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
