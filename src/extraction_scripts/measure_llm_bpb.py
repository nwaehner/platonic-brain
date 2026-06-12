"""
measure_llm_bpb.py — language-model performance for the alignment x-axis.

Computes each LLM's performance as **1 − bits-per-byte** over 4M tokens of OpenWebText
(Gokaslan & Cohen 2019) — the exact protocol the Platonic Representation Hypothesis
(Huh et al. 2024) uses for its language x-axis. Those per-model numbers are not published
in a table nor shipped by the platonic-rep repo, so we compute them here.

bits-per-byte is tokenizer-independent: bpb = (Σ token NLL in nats / ln2) / (UTF-8 bytes
of the same text span). All models score over the *same* OpenWebText text buffer.

Output: tokens/llm_bpb.json  →  { "<stem>": {"perf": 1-bpb, "bpb": ..., "n_tokens": ...,
"n_bytes": ...} }.  `src/alignment_plots/_common.py` loads this for LLM_PERF.

Usage:
  python src/extraction_scripts/measure_llm_bpb.py --all                 # bloom+openllama+llama
  python src/extraction_scripts/measure_llm_bpb.py --model bigscience/bloomz-560m
  python src/extraction_scripts/measure_llm_bpb.py --all --n-tokens 4000000 --ctx 1024
  python src/extraction_scripts/measure_llm_bpb.py --model bigscience/bloomz-560m --test
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import extract_llm_captions_neurolm as neuro       # load_text_model, _model_stem, model lists
import extract_llm_captions as gen                  # LLAMA_MODELS

OWT_DATASET = "Skylion007/openwebtext"
LN2 = math.log(2.0)


def gather_owt_text(target_chars: int) -> str:
    """Stream OpenWebText documents until we have at least `target_chars` characters."""
    from datasets import load_dataset
    ds = load_dataset(OWT_DATASET, split="train", streaming=True, trust_remote_code=True)
    parts, total = [], 0
    for doc in ds:
        t = doc.get("text", "")
        if not t:
            continue
        parts.append(t)
        total += len(t) + 1
        if total >= target_chars:
            break
    return "\n".join(parts)


@torch.no_grad()
def model_bpb(model_name, text, n_tokens, ctx, device):
    """Return (bpb, n_pred_tokens, n_bytes) for one model over the first n_tokens."""
    tokenizer, model = neuro.load_text_model(model_name, device)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) < n_tokens:
        print(f"  WARN: only {len(ids)} tokens available (<{n_tokens}); using all.")
    ids = ids[:n_tokens]
    # byte span of exactly these tokens (tokenizer-agnostic via decode)
    n_bytes = len(tokenizer.decode(ids).encode("utf-8"))

    ids_t = torch.tensor(ids, dtype=torch.long)
    total_nats, n_pred = 0.0, 0
    for i in range(0, len(ids_t) - 1, ctx):
        win = ids_t[i:i + ctx + 1].to(device)        # +1 so every label has a context
        if win.numel() < 2:
            break
        out = model(input_ids=win.unsqueeze(0), labels=win.unsqueeze(0))
        n = win.numel() - 1
        total_nats += float(out.loss) * n
        n_pred += n
    bpb = (total_nats / LN2) / max(n_bytes, 1)

    del model, tokenizer
    gc.collect()
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bpb, n_pred, n_bytes


def parse_args():
    ap = argparse.ArgumentParser(description="1-BPB OpenWebText scores for LLMs.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--all-bloom", action="store_true")
    g.add_argument("--all-openllama", action="store_true")
    g.add_argument("--all-llama", action="store_true")
    g.add_argument("--all", action="store_true", help="bloom + openllama + llama")
    ap.add_argument("--n-tokens", type=int, default=4_000_000)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "tokens" / "llm_bpb.json"))
    ap.add_argument("--test", action="store_true", help="50k tokens, quick sanity run.")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_tokens = 50_000 if args.test else args.n_tokens

    if args.all:
        models = neuro.BLOOM_MODELS + neuro.OPENLLAMA_MODELS + gen.LLAMA_MODELS
    elif args.all_bloom:
        models = neuro.BLOOM_MODELS
    elif args.all_openllama:
        models = neuro.OPENLLAMA_MODELS
    elif args.all_llama:
        models = gen.LLAMA_MODELS
    else:
        models = [args.model]

    print(f"Device={device}  n_tokens={n_tokens}  ctx={args.ctx}")
    print("Gathering OpenWebText buffer...")
    text = gather_owt_text(target_chars=n_tokens * 6)   # ~6 chars/token headroom
    print(f"  buffer chars={len(text):,}")

    out_path = Path(args.out)
    scores = {}
    if out_path.exists():
        scores = json.loads(out_path.read_text())

    for m in models:
        stem = neuro._model_stem(m)
        print(f"\n=== {m} ({stem}) ===")
        t0 = time.time()
        bpb, n_pred, n_bytes = model_bpb(m, text, n_tokens, args.ctx, device)
        perf = 1.0 - bpb
        scores[stem] = {"perf": perf, "bpb": bpb, "n_tokens": n_pred, "n_bytes": n_bytes}
        print(f"  bpb={bpb:.4f}  perf(1-bpb)={perf:.4f}  "
              f"(pred_tokens={n_pred:,}, bytes={n_bytes:,}, {time.time()-t0:.0f}s)")
        if not args.test:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(scores, indent=2))   # checkpoint after each
            print(f"  wrote {out_path}")

    print("\nFinal scores:")
    for stem, d in scores.items():
        print(f"  {stem:16s} perf={d['perf']:.4f}  bpb={d['bpb']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
