"""CLI: python -m medera.preprocess text | pack"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from medera.collect.manifest import read_manifest
from medera.config import load_yaml
from medera.preprocess.budget import equal_budget, pack_blocks, take_budget
from medera.preprocess.filters import exclusion_reason
from medera.preprocess.jats import article_meta, jats_to_text
from medera.preprocess.near_dup import near_duplicates
from medera.preprocess.split import heldout_split


def load_relevance_labels(cfg: dict, root: Path, rows: list[dict], opt_out: bool) -> dict | None:
    """pmcid -> screen label for every harvested article. A missing file or any unlabelled article is a hard
    error, never a silent keep; --no-relevance-screen (smoke data only) skips the screen loudly."""
    if opt_out:
        print("[text] WARNING: --no-relevance-screen: relevance screen NOT applied (smoke data only)", flush=True)
        return None
    path = root / cfg["filters"]["relevance_labels"]
    if not path.exists():
        raise SystemExit(f"[text] {path} is missing: run the relevance screen (collect screen-export / "
                         f"screen-import) first; every harvested article needs a label")
    with open(path, newline="", encoding="utf-8") as f:
        labels = {r["pmcid"]: r["label"] for r in csv.DictReader(f)}
    unlabelled = [r["pmcid"] for r in rows if r["pmcid"] not in labels]
    if unlabelled:
        raise SystemExit(f"[text] {len(unlabelled)} harvested articles have no relevance label in {path} "
                         f"(first: {unlabelled[:10]}); screen them before building")
    return labels


def cmd_text(cfg: dict, root: Path, args) -> None:
    corpus = cfg["corpus"]
    rows = read_manifest(root / "manifests" / f"{corpus}_all.csv")
    raw, out = root / "raw" / corpus, root / "processed" / corpus
    missing = [r["pmcid"] for r in rows if not (raw / r["window"] / f"{r['pmcid']}.xml").exists()]
    if missing:
        raise SystemExit(f"[text] {len(missing)} XML files missing under {raw} (first: {missing[:5]})")
    labels = load_relevance_labels(cfg, root, rows, args.no_relevance_screen)
    keep_labels = set(cfg["filters"]["keep_labels"])
    out.mkdir(parents=True, exist_ok=True)
    texts = {w: {} for w in cfg["windows"]}
    reasons = {w: {} for w in cfg["windows"]}
    dates = {}
    for r in rows:
        w, pmcid = r["window"], r["pmcid"]
        dates[pmcid] = r["first_pub_date"]
        xml = (raw / w / f"{pmcid}.xml").read_bytes()
        try:
            text, meta = jats_to_text(xml), article_meta(xml)
        except ValueError:
            reasons[w][pmcid] = "parse_error"
            continue
        reason = exclusion_reason(text, meta, cfg["filters"])
        if not reason and labels is not None and labels[pmcid] not in keep_labels:
            reason = "off_topic_screen"
        if reason:
            reasons[w][pmcid] = reason
        else:
            texts[w][pmcid] = text

    # Near-duplicates across ALL windows, before the held-out split; the later doc of a pair is dropped.
    window_of = {p: w for w, t in texts.items() for p in t}
    pairs = near_duplicates({p: (dates[p], t) for w in texts for p, t in texts[w].items()})
    with open(out / "near_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, ["kept_pmcid", "kept_window", "dropped_pmcid", "dropped_window", "jaccard"])
        wr.writeheader()
        for kept, dropped, jac in pairs:
            wr.writerow({"kept_pmcid": kept, "kept_window": window_of[kept], "dropped_pmcid": dropped,
                         "dropped_window": window_of[dropped], "jaccard": jac})
            del texts[window_of[dropped]][dropped]
            reasons[window_of[dropped]][dropped] = "near_duplicate"
    print(f"[text] near-duplicates (5-gram Jaccard >= 0.8): {len(pairs)} dropped", flush=True)

    excluded = []
    for w in cfg["windows"]:
        texts_w, reasons_w = texts[w], reasons[w]
        train, held = heldout_split(list(texts_w), cfg["split"]["heldout_frac"], cfg["split"]["seed"])
        held = set(held)
        with open(out / f"{w}.jsonl", "w", encoding="utf-8") as f:
            for pmcid in sorted(texts_w):
                split = "heldout" if pmcid in held else "train"
                f.write(json.dumps({"pmcid": pmcid, "split": split, "text": texts_w[pmcid]}, ensure_ascii=False) + "\n")
        excluded += [{"pmcid": p, "window": w, "reason": reasons_w[p]} for p in sorted(reasons_w)]
        chars = sum(len(t) for t in texts_w.values())
        screened = Counter(labels[r["pmcid"]] for r in rows if r["window"] == w) if labels is not None else {}
        print(f"[text] {w}: kept={len(texts_w)} (train={len(train)} heldout={len(held)}) chars={chars:,} "
              f"excluded={dict(Counter(reasons_w.values()))} screen={dict(screened)}", flush=True)
        if "parse_error" in reasons_w.values():
            print(f"[text] WARNING: {w} has XML that does not parse", flush=True)
    with open(out / "excluded.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, ["pmcid", "window", "reason"])
        wr.writeheader()
        wr.writerows(excluded)
    print(f"[text] wrote {out}/<window>.jsonl and excluded.csv ({len(excluded)} excluded)")


def load_tokenizer(hf_id: str, revision: str):
    from transformers import AutoTokenizer  # lazy: the text step must not need transformers

    return AutoTokenizer.from_pretrained(hf_id, revision=revision)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_pack(cfg: dict, root: Path, args) -> None:
    m = load_yaml(args.models)[args.model]
    tok = load_tokenizer(m["hf_id"], m["revision"])
    separator = m.get("doc_separator") or tok.eos_token  # no key: fall back to the tokenizer's eos
    sep_id = tok.convert_tokens_to_ids(separator) if "doc_separator" in m else tok.eos_token_id
    if sep_id is None or (tok.unk_token_id is not None and sep_id == tok.unk_token_id):
        raise SystemExit(f"[pack] doc separator {separator!r} is not a single known token for {m['hf_id']}")
    bos_id = tok.bos_token_id if tok.bos_token_id is not None and tok("a").input_ids[0] == tok.bos_token_id else None
    corpus = cfg["corpus"]
    src = root / "processed" / corpus
    out = src / args.model
    out.mkdir(parents=True, exist_ok=True)

    def encode(text):
        return tok(text, add_special_tokens=False).input_ids + [sep_id]

    train, held = {}, {}
    for w in cfg["windows"]:
        docs = sorted(_read_jsonl(src / f"{w}.jsonl"), key=lambda d: d["pmcid"])
        train[w] = [encode(d["text"]) for d in docs if d["split"] == "train"]
        held[w] = [encode(d["text"]) for d in docs if d["split"] == "heldout"]
        print(f"[pack] {w}: tokenized {len(train[w])} train + {len(held[w])} heldout docs", flush=True)

    lengths = {w: sum(map(len, d)) for w, d in train.items()}
    budget = equal_budget(lengths)
    min_tokens = cfg["budget"]["min_train_tokens"]
    if budget < min_tokens:
        print(f"[pack] WARNING: budget {budget:,} tokens is below budget.min_train_tokens {min_tokens:,}", flush=True)
    meta = {"model": args.model, "hf_id": m["hf_id"], "revision": m["revision"], "block_len": args.block_len,
            "bos_id": bos_id, "doc_separator": separator, "separator_id": sep_id, "budget_tokens": budget, "min_train_tokens": min_tokens,
            "below_min": budget < min_tokens, "windows": {}}
    for w in cfg["windows"]:
        blocks = pack_blocks(take_budget(train[w], budget, cfg["budget"]["seed"]), args.block_len, bos_id)
        hblocks = pack_blocks([t for d in held[w] for t in d], args.block_len, bos_id)
        np.save(out / f"{w}_train.npy", blocks)
        np.save(out / f"{w}_heldout.npy", hblocks)
        meta["windows"][w] = {"train_docs": len(train[w]), "train_tokens_available": lengths[w],
                              "train_blocks": int(blocks.shape[0]), "heldout_docs": len(held[w]),
                              "heldout_blocks": int(hblocks.shape[0])}
        print(f"[pack] {w}: {lengths[w]:,} train tokens available -> {blocks.shape[0]} blocks; "
              f"heldout {hblocks.shape[0]} blocks", flush=True)
    (out / "budget.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[pack] budget {budget:,} tokens per window, bos_id={bos_id}, doc separator {separator!r}={sep_id}; "
          f"wrote {out}")


def main(argv=None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/corpus.yaml")
    common.add_argument("--data-root", default="data")
    ap = argparse.ArgumentParser(prog="python -m medera.preprocess", description="JATS to text, budget, packing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text", parents=[common], help="hrt_all.csv + raw XML -> <window>.jsonl, excluded.csv")
    t.add_argument("--no-relevance-screen", action="store_true",
                   help="skip the relevance-label gate (smoke data only; logged loudly)")
    p = sub.add_parser("pack", parents=[common], help="tokenize, equal budget, pack -> .npy + budget.json")
    p.add_argument("--model", required=True, help="key in models.yaml")
    p.add_argument("--models", default="configs/models.yaml")
    p.add_argument("--block-len", type=int, default=2048)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    {"text": cmd_text, "pack": cmd_pack}[args.cmd](cfg, Path(args.data_root), args)


if __name__ == "__main__":
    main()
