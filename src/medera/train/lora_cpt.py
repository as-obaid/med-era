"""LoRA continued pretraining of one arm on one window, and the --smoke first-job check.

    python -m medera.train.lora_cpt --arm gemma_ex --window w1                 # full run (resumable)
    python -m medera.train.lora_cpt --arm gemma_ex --data-root data/smoke --smoke

Training reads packed 2048-token blocks (`preprocess pack`), trains from the pinned base checkpoint
for one pass, then runs the adapter audit. Exit codes: 0 ok, 1 audit or smoke failure, 3 stopped by
the wall-time guard (resubmit the same arm/window to resume from the last checkpoint).
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from medera.config import load_yaml
from medera.train.audit_adapter import audit, check
from medera.train.targets import expert_targets, linear_targets

PRECISIONS = ("bf16", "qlora4")
EXIT_OK, EXIT_FAIL, EXIT_WALL = 0, 1, 3
MAX_SMOKE_LOSS = 8.0  # a pretrained LM on English text starts far below this; random weights give ~ln(vocab) > 12


def arm_config(train_cfg: dict, arm: str, overrides: dict | None = None) -> dict:
    if arm not in train_cfg["arms"]:
        raise KeyError(f"unknown arm {arm!r}; choose from {sorted(train_cfg['arms'])}")
    cfg = {**train_cfg["defaults"], **train_cfg["arms"][arm], "arm": arm}
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if cfg["precision"] not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {cfg['precision']!r}")
    return cfg


def lr_at(step: int, total: int, warmup: int) -> float:
    """LR multiplier for optimizer step `step` (0-based): linear warmup, then cosine to 0."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def load_model(model_cfg: dict, precision: str):
    """Pinned base checkpoint on GPU 0, bf16 or NF4. Raises if any weight failed to load."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kw = dict(revision=model_cfg["revision"], dtype=torch.bfloat16, attn_implementation="sdpa",
              device_map={"": 0}, output_loading_info=True)
    if precision == "qlora4":
        # Deliberately no prepare_model_for_kbit_training: it upcasts every non-quantized tensor to
        # fp32, and Gemma-4's fused experts are not quantized by bitsandbytes (they stay bf16).
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=model_cfg["skip_quant_modules"])
    model, info = AutoModelForCausalLM.from_pretrained(model_cfg["hf_id"], **kw)
    if info["missing_keys"]:
        raise RuntimeError(f"{len(info['missing_keys'])} weights missing, e.g. {sorted(info['missing_keys'])[:5]}")
    return model


def apply_lora(model, cfg: dict, model_cfg: dict):
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = expert_targets(model, model_cfg["expert_parameters"]) if cfg["experts"] else []
    if cfg["experts"] and not params:
        raise RuntimeError(f"expert arm but no parameters match {model_cfg['expert_parameters']}")
    lora = cfg["lora"]
    config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
                        target_modules=linear_targets(model), target_parameters=params or None,
                        task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def fast_path_status(model) -> bool | None:
    """True if every linear-attention layer uses the fla kernel, False if any uses the torch
    fallback (about 40 s/step for Qwen3.5), None if the model has no linear attention."""
    fns = [m.chunk_gated_delta_rule for m in model.modules() if hasattr(m, "chunk_gated_delta_rule")]
    if not fns:
        return None
    return all(getattr(fn, "__module__", "").startswith("fla") for fn in fns)


def _save_ckpt(path: Path, model, opt, sched, state: dict) -> None:
    import torch
    from peft import get_peft_model_state_dict

    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "state.pt.tmp"
    torch.save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "state": state}, tmp)
    os.replace(tmp, path / "state.pt")


def train(model, blocks: np.ndarray, cfg: dict, max_steps: int | None = None, ckpt_dir: Path | None = None,
          deadline: float | None = None, log=print, tracker=None) -> dict:
    """One pass over `blocks` [n, block_len]. Records the max grad norm per trainable tensor."""
    import torch
    from peft import set_peft_model_state_dict

    device = next(model.parameters()).device
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable = [p for _, p in named]
    opt = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    mb, accum = cfg["micro_batch"], cfg["grad_accum"]
    total = (len(blocks) // mb) // accum
    warmup = max(1, round(cfg["warmup_ratio"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, total, warmup))
    order = np.random.default_rng(cfg["seed"]).permutation(len(blocks))
    state = {"step": 0, "losses": [], "step_times": [], "grad_norms": {}}
    if ckpt_dir is not None and (ckpt_dir / "state.pt").exists():
        ck = torch.load(ckpt_dir / "state.pt", map_location=device, weights_only=False)
        set_peft_model_state_dict(model, ck["adapter"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        state = ck["state"]
        log(f"[train] resumed at step {state['step']}/{total}")
    limit = total if max_steps is None else min(total, max_steps)
    tokens_per_step = mb * accum * blocks.shape[1]
    log(f"[train] blocks={len(blocks)} steps={total} (running to {limit}) warmup={warmup} tokens/step={tokens_per_step:,}")

    model.train()
    stopped = "done" if limit == total else "max_steps"
    while state["step"] < limit:
        t0, loss_sum = time.time(), 0.0
        for k in range(accum):
            start = (state["step"] * accum + k) * mb
            x = torch.from_numpy(blocks[np.sort(order[start:start + mb])].astype(np.int64)).to(device)
            loss = model(input_ids=x, labels=x).loss
            (loss / accum).backward()
            loss_sum += loss.item()
        zero = torch.zeros((), device=device)
        norms = torch.stack([p.grad.detach().float().norm() if p.grad is not None else zero for p in trainable]).tolist()
        for (name, _), g in zip(named, norms):
            state["grad_norms"][name] = max(state["grad_norms"].get(name, 0.0), g)
        torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        state["losses"].append(loss_sum / accum)
        state["step_times"].append(time.time() - t0)

        step = state["step"]
        if step % cfg["log_every"] == 0 or step == limit or step == 1:
            s_per_step = statistics.median(state["step_times"][-20:])
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            eta_h = (limit - step) * s_per_step / 3600
            log(f"[train] step {step}/{limit} loss={state['losses'][-1]:.4f} lr={sched.get_last_lr()[0]:.2e} "
                f"s/step={s_per_step:.2f} tok/s={tokens_per_step / s_per_step:,.0f} peak={peak:.1f}GB eta={eta_h:.2f}h")
            if tracker is not None:
                tracker.log({"loss": state["losses"][-1], "lr": sched.get_last_lr()[0], "s_per_step": s_per_step,
                             "tokens_per_s": tokens_per_step / s_per_step, "peak_mem_gb": peak}, step=step)
        if ckpt_dir is not None and step < limit and step % cfg["save_every_steps"] == 0:
            _save_ckpt(ckpt_dir, model, opt, sched, state)
        if deadline is not None and step < limit and time.time() > deadline:
            if ckpt_dir is not None:
                _save_ckpt(ckpt_dir, model, opt, sched, state)
            stopped = "wall"
            log(f"[train] wall-time guard hit at step {step}/{limit}; checkpoint saved, resubmit to resume")
            break

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return {**state, "total_steps": total, "tokens_per_step": tokens_per_step, "peak_mem_gb": peak, "stopped": stopped}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def start_tracker(cfg: dict, model_cfg: dict, window: str, smoke: bool):
    """W&B run when WANDB_API_KEY is set (jobs run with WANDB_MODE=offline), else None. Logs the
    resolved config and git SHA only: never article text or the environment."""
    if not os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb

        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", "MedEra"), entity=os.environ.get("WANDB_ENTITY"),
            name=f"{cfg['arm']}-{window}-{cfg['precision']}-s{cfg['seed']}" + ("-smoke" if smoke else ""),
            group=cfg["arm"], job_type="smoke" if smoke else "train",
            tags=[cfg["arm"], window, cfg["model"], cfg["precision"]] + (["smoke"] if smoke else []),
            config={**cfg, "window": window, "model_cfg": model_cfg, "git_sha": _git_sha(), "smoke": smoke})
    except Exception as e:  # tracking must never take a training run down
        print(f"[train] W&B disabled: {type(e).__name__}: {e}", flush=True)
        return None


def _env_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "flash-linear-attention", "causal-conv1d"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def main(argv=None) -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--window", default="w1")
    ap.add_argument("--smoke", action="store_true", help="a few real steps + audit; no checkpoints")
    ap.add_argument("--precision", choices=PRECISIONS, help="override the arm's precision")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--models-config", default="configs/models.yaml")
    args = ap.parse_args(argv)

    import torch

    cfg = arm_config(load_yaml(args.train_config), args.arm, {"precision": args.precision})
    model_cfg = load_yaml(args.models_config)[cfg["model"]]
    runs, name = Path(args.runs_dir), f"{args.arm}_{args.window}"
    blocks = np.load(Path(args.data_root) / "processed" / "hrt" / cfg["model"] / f"{args.window}_train.npy", mmap_mode="r")
    if blocks.shape[1] != cfg["block_len"]:
        raise ValueError(f"blocks are {blocks.shape[1]} tokens, config says {cfg['block_len']}")
    torch.manual_seed(cfg["seed"])
    print(f"[train] arm={args.arm} window={args.window} model={model_cfg['hf_id']}@{model_cfg['revision'][:8]} "
          f"precision={cfg['precision']} experts={cfg['experts']} gpu={torch.cuda.get_device_name(0)}", flush=True)

    tracker = start_tracker(cfg, model_cfg, args.window, args.smoke)
    model = load_model(model_cfg, cfg["precision"])
    load_s = time.time() - t_start
    model = apply_lora(model, cfg, model_cfg)
    fast = fast_path_status(model)
    print(f"[train] loaded in {load_s:.0f}s; fla fast path: {fast}", flush=True)

    log = lambda msg: print(msg, flush=True)  # noqa: E731
    if args.smoke:
        res = train(model, blocks, cfg, max_steps=args.max_steps or cfg["smoke_steps"], log=log, tracker=tracker)
    else:
        deadline = t_start + cfg["wall_guard_hours"] * 3600
        res = train(model, blocks, cfg, max_steps=args.max_steps, ckpt_dir=runs / "checkpoints" / name,
                    deadline=deadline, log=log, tracker=tracker)

    report = audit(model, res["grad_norms"], args.arm, cfg["experts"])
    report.update(window=args.window, precision=cfg["precision"], model=cfg["model"])
    failures = check(report)
    summary = {"arm": args.arm, "window": args.window, "precision": cfg["precision"], "model": model_cfg,
               "gpu": torch.cuda.get_device_name(0), "versions": _env_versions(), "git_sha": _git_sha(),
               "load_s": load_s, "steps": res["step"], "total_steps": res["total_steps"], "losses": res["losses"],
               "step_times": res["step_times"], "peak_mem_gb": res["peak_mem_gb"], "fla_fast_path": fast,
               "stopped": res["stopped"]}

    if args.smoke:
        timed = res["step_times"][1:] or res["step_times"]
        s_per_step = statistics.median(timed)
        tok_s = res["tokens_per_step"] / s_per_step
        if res["step"] < 2:
            failures.append(f"only {res['step']} optimizer steps ran (need >= 2 for lora_A grads)")
        if res["losses"] and res["losses"][0] > MAX_SMOKE_LOSS:
            failures.append(f"step-1 loss {res['losses'][0]:.2f} > {MAX_SMOKE_LOSS}: weights look uninitialized")
        if fast is False:
            failures.append("linear attention is on the torch fallback (fla kernels not active)")
        summary.update(s_per_optimizer_step=s_per_step, s_per_micro_step=s_per_step / cfg["grad_accum"],
                       tokens_per_s=tok_s, max_tokens_within_wall_guard=tok_s * cfg["wall_guard_hours"] * 3600,
                       audit=report, failures=failures)
        _write_json(runs / "results" / "smoke" / f"{args.arm}.json", summary)
        loss1 = res["losses"][0] if res["losses"] else float("nan")
        print(f"[smoke] {'PASS' if not failures else 'FAIL'} arm={args.arm} loss1={loss1:.3f} "
              f"s/step={s_per_step:.2f} (accum {cfg['grad_accum']}) tok/s={tok_s:,.0f} peak={res['peak_mem_gb']:.1f}GB "
              f"fits~{tok_s * cfg['wall_guard_hours'] * 3600 / 1e6:.1f}M tokens/{cfg['wall_guard_hours']}h "
              f"expert_tensors={report['groups']['experts']['lora_tensors']} "
              f"expert_nonzero_grad={report['groups']['experts']['lora_nonzero_grad']} "
              f"vision_tensors={report['groups']['vision']['lora_tensors']}", flush=True)
        for f in failures:
            print(f"[smoke]   - {f}", flush=True)
        code = EXIT_FAIL if failures else EXIT_OK
    elif res["stopped"] == "wall":
        code = EXIT_WALL
    else:
        model.save_pretrained(runs / "adapters" / name)
        _write_json(runs / "results" / "adapter_audit" / f"{name}.json", report)
        _write_json(runs / "results" / "train" / f"{name}.json", {**summary, "audit_failures": failures})
        print(f"[train] {'PASS' if not failures else 'AUDIT FAIL'} {name}: adapter -> {runs / 'adapters' / name}", flush=True)
        for f in failures:
            print(f"[train]   - {f}", flush=True)
        code = EXIT_FAIL if failures else EXIT_OK

    if tracker is not None:
        tracker.summary.update({"exit_code": code, "audit_failures": len(failures), "steps": res["step"]})
        tracker.finish(exit_code=code)
    return code


if __name__ == "__main__":
    sys.exit(main())
