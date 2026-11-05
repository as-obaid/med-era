"""The probe battery (configs/probes.yaml): validation, prompts, its content hash and the chunk plan. CPU only.

A wording is one probe's original (paraphrase_id 0) or one of its paraphrases (1, 2, ...). A chunk is one generate()
call: up to `decoding.batch_size` wordings in battery order, for one sampled seed or the greedy pass. Chunks are the
unit of resumption: each is written once, atomically, to results/answers/<variant>/<chunk id>.jsonl.
"""
import hashlib
import json
from dataclasses import dataclass

# Everything that changes what is generated. Comments, status and the tag name are not part of the hash.
BATTERY_KEYS = ("format_primer", "prompt_template", "stop", "probes", "decoding")
GEN_KEYS = ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")


@dataclass(frozen=True)
class Wording:
    probe_id: str
    dimension: str
    paraphrase_id: int
    question: str


@dataclass(frozen=True)
class Chunk:
    id: str
    mode: str                 # "sampled" | "greedy"
    seed: int | None          # the battery seed (sampled) or None (greedy)
    index: int                # chunk number within this seed or pass
    wordings: tuple

    @property
    def torch_seed(self) -> int:
        """Seed set right before this chunk's generate() call: the same for every variant, so a given probe, seed and
        chunk use the same random stream across variants (paired comparisons, F5)."""
        return (self.seed or 0) * 1000 + self.index


def validate(cfg: dict) -> list[str]:
    """Problems that make the battery unusable (an empty list means it is usable; it may still be a draft)."""
    p = []
    if cfg.get("status") not in ("draft", "frozen"):
        p.append(f"status must be draft or frozen, got {cfg.get('status')!r}")
    if not cfg.get("prereg_tag"):
        p.append("prereg_tag is missing")
    if not (cfg.get("format_primer") or "").strip():
        p.append("format_primer is empty")
    tmpl = cfg.get("prompt_template") or ""
    if "{primer}" not in tmpl or "{question}" not in tmpl:
        p.append("prompt_template needs {primer} and {question}")
    dims = set(cfg.get("dimensions") or [])
    probes = cfg.get("probes") or []
    if not probes:
        p.append("no probes")
    seen = set()
    for pr in probes:
        pid = pr.get("id")
        if pid in seen:
            p.append(f"duplicate probe id {pid}")
        seen.add(pid)
        if pr.get("dimension") not in dims:
            p.append(f"{pid}: dimension {pr.get('dimension')!r} not in {sorted(dims)}")
        texts = [pr.get("original")] + list(pr.get("paraphrases") or [])
        if not pr.get("original") or len(texts) < 2:
            p.append(f"{pid}: needs an original and at least one paraphrase")
        if any(not isinstance(t, str) or not t.strip() for t in texts) or len(set(texts)) != len(texts):
            p.append(f"{pid}: empty or repeated wording")
    dec = cfg.get("decoding") or {}
    s, g = dec.get("sampled") or {}, dec.get("greedy") or {}
    if not s.get("do_sample") or not s.get("seeds") or len(set(s["seeds"])) != len(s["seeds"]):
        p.append("decoding.sampled needs do_sample true and distinct seeds")
    if s.get("temperature") in (None, 0):
        p.append("decoding.sampled.temperature must be > 0 (Qwen3.5 returns empty output at 0)")
    if g.get("do_sample") is not False:
        p.append("decoding.greedy needs do_sample false")
    if not isinstance(dec.get("batch_size"), int) or dec["batch_size"] < 1:
        p.append("decoding.batch_size must be a positive integer")
    return p


def wordings(cfg: dict) -> list[Wording]:
    out = []
    for pr in cfg["probes"]:
        for i, q in enumerate([pr["original"]] + list(pr["paraphrases"])):
            out.append(Wording(pr["id"], pr["dimension"], i, " ".join(q.split())))
    return out


def build_prompt(cfg: dict, question: str) -> str:
    return cfg["prompt_template"].format(primer=cfg["format_primer"].strip(), question=" ".join(question.split()))


def battery_sha256(cfg: dict) -> str:
    blob = json.dumps({k: cfg.get(k) for k in BATTERY_KEYS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def gen_settings(cfg: dict, mode: str) -> dict:
    """The explicit generation settings of a pass (no seeds). Greedy fixes the sampling fields to neutral values."""
    d = cfg["decoding"][mode]
    out = {k: d[k] for k in GEN_KEYS if k in d}
    out.setdefault("repetition_penalty", 1.0)
    if not out["do_sample"]:
        out.update(temperature=1.0, top_p=1.0, top_k=0)
    return out


def plan_chunks(cfg: dict) -> list[Chunk]:
    ws, bs = wordings(cfg), cfg["decoding"]["batch_size"]
    groups = [tuple(ws[i : i + bs]) for i in range(0, len(ws), bs)]
    out = [Chunk(f"sampled-s{seed}-c{i:02d}", "sampled", seed, i, g)
           for seed in cfg["decoding"]["sampled"]["seeds"] for i, g in enumerate(groups)]
    out += [Chunk(f"greedy-c{i:02d}", "greedy", None, i, g) for i, g in enumerate(groups)]
    return out
