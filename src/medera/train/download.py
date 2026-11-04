"""Pre-download the pinned base checkpoints into HF_HOME. Run where there is internet."""
import argparse

from huggingface_hub import snapshot_download

from medera.config import load_yaml

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--model", default="all", help="model key from models.yaml, or 'all'")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.models)
    keys = list(cfg) if args.model == "all" else [args.model]
    failed = []
    for key in keys:
        m = cfg[key]
        try:
            path = snapshot_download(m["hf_id"], revision=m["revision"], allow_patterns=PATTERNS)
            print(f"[download] {key}: OK {path}", flush=True)
        except Exception as e:  # keep going so one gated model does not block the other
            hint = " (gated: accept the license on the model page and log in with `hf auth login`)" if m.get("gated") else ""
            print(f"[download] {key}: FAILED {type(e).__name__}: {e}{hint}", flush=True)
            failed.append(key)
    if failed:
        raise SystemExit(f"[download] failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
