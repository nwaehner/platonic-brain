"""
extract_llm_captions.py — generic LLM caption embeddings for ANY EEG window scheme.

The original NeuroLM-only extractor is frozen as `extract_llm_captions_neurolm.py`
(8 s windows = exactly two 4 s captions). This generic version aligns the 4 s Qwen
captions to whichever EEG model's window scheme you pass via `--eeg-model`, then
extracts the same layerwise masked-mean-pooled hidden states for BLOOM / OpenLLaMA /
**LLaMA** models.

Window→caption alignment (the "include every overlapping caption" rule):
  Season 7 = 2700 clips × 4 s = 10,800 s. EEG window w of length L spans [wL, (w+1)L).
  Caption c spans [4c, 4(c+1)). A window includes EVERY caption that overlaps it
  (overlap length > 0), concatenated with a space — exactly as Aristotelian
  concatenates multi-caption inputs. This reproduces the NeuroLM 2-caption case and
  generalises to 1.5→2 (STEEGFormer) and 2.5→3 (REVE):

    EEG model     window  #windows  captions/window
    neurolm       8 s     1350      2
    femba         5 s     2160      2
    luna          5 s     2160      2
    steegformer   6 s     1800      2
    reve          10 s    1080      3

Adjacent non-overlapping windows share boundary captions (unavoidable when L is not a
multiple of 4 s). This inflates temporal autocorrelation in alignment — handled
downstream by the shift-null / block-permutation tests in src/alignment_plots/.

Output: <out-dir>/<eeg_model>/<stem>_layerwise.npz   (mirrors HF llms/<eeg_model>/)
  embeddings        (n_layers, W, D)  float32
  window_start_sec  (W,)  float64   start time (s) of each window
  n_caps_per_window (W,)  int64
  win_sec, eeg_model, model_name, n_layers

Usage:
  python extract_llm_captions.py --eeg-model femba --model bigscience/bloomz-560m
  python extract_llm_captions.py --eeg-model neurolm --all-llama
  python extract_llm_captions.py --eeg-model steegformer --all-bloom --batch-size 4
  python extract_llm_captions.py --eeg-model reve --model bigscience/bloomz-560m --test
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Reuse the helpers from the frozen NeuroLM extractor (same directory).
import extract_llm_captions_neurolm as neuro


# ── Window schemes (mirror src/extraction_scripts/extract_<model>.py) ──────────
CLIP_SEC = 4
SEASON7_TOTAL_SEC = neuro.SEASON7_N_CLIPS * CLIP_SEC   # 2700 × 4 = 10,800 s

WINDOW_SCHEMES = {
    "neurolm":     {"win_sec": 8,  "n_windows": 1350},
    "femba":       {"win_sec": 5,  "n_windows": 2160},
    "luna":        {"win_sec": 5,  "n_windows": 2160},
    "steegformer": {"win_sec": 6,  "n_windows": 1800},
    "reve":        {"win_sec": 10, "n_windows": 1080},
    # Native 4 s clip granularity (1 caption per window, 2700 windows). Used by
    # video_vs_language so video and language are matched at the raw clip level
    # with no aggregation on either side. Pair with vision extracted at
    # `--eeg-family clip4s --window-seconds 4`.
    "clip4s":      {"win_sec": 4,  "n_windows": 2700},
}

# Original LLaMA weights (the "33b" checkpoint ships as llama-30b).
LLAMA_MODELS = [
    "huggyllama/llama-13b",
    "huggyllama/llama-30b",
    "huggyllama/llama-65b",
]


def overlapping_captions(w, win_sec):
    """Caption indices whose [4c, 4c+4) interval overlaps window [wL, (w+1)L)."""
    w0, w1 = w * win_sec, (w + 1) * win_sec
    c_lo = max(0, w0 // CLIP_SEC - 1)
    c_hi = min(neuro.SEASON7_N_CLIPS, w1 // CLIP_SEC + 2)
    return [c for c in range(c_lo, c_hi)
            if min((c + 1) * CLIP_SEC, w1) - max(c * CLIP_SEC, w0) > 0]


def build_window_texts(captions: List[str], win_sec: int, n_windows: int):
    """Return (texts[W], n_caps[W]) for the given window scheme."""
    if len(captions) != neuro.SEASON7_N_CLIPS:
        raise ValueError(f"Expected {neuro.SEASON7_N_CLIPS} captions, got {len(captions)}")
    if win_sec * n_windows != SEASON7_TOTAL_SEC:
        raise ValueError(f"win_sec {win_sec} × n_windows {n_windows} "
                         f"≠ {SEASON7_TOTAL_SEC} s")
    texts, n_caps = [], []
    for w in range(n_windows):
        sel = overlapping_captions(w, win_sec)
        texts.append(" ".join(captions[c] for c in sel))
        n_caps.append(len(sel))
    return texts, n_caps


def run(model_name, texts, n_caps, win_sec, eeg_model, out_dir, device,
        batch_size, max_length, test):
    stem = neuro._model_stem(model_name)
    suffix = "_test" if test else "_layerwise"
    model_dir = out_dir / eeg_model
    out_path = model_dir / f"{stem}{suffix}.npz"
    if out_path.exists():
        print(f"\n  {out_path} already exists — skipping.")
        return

    n_windows = len(texts)
    if test:
        rng = random.Random(0)
        idx = sorted(rng.sample(range(n_windows), min(50, n_windows)))
        sub_texts = [texts[i] for i in idx]
        win_start = np.array([i * win_sec for i in idx], dtype=np.float64)
        caps = np.array([n_caps[i] for i in idx], dtype=np.int64)
        print(f"\n  TEST MODE: {len(sub_texts)} random windows "
              f"(caps/window: min={caps.min()} max={caps.max()})")
    else:
        sub_texts = texts
        win_start = np.arange(n_windows, dtype=np.float64) * win_sec
        caps = np.array(n_caps, dtype=np.int64)

    print(f"\n{'='*64}\n  {model_name}  [{eeg_model}: {win_sec}s × {n_windows} win]")
    print(f"  windows={len(sub_texts)}  caps/window mean={caps.mean():.2f}  "
          f"batch={batch_size}  device={device}\n{'='*64}")

    t0 = time.time()
    tokenizer, model = neuro.load_text_model(model_name, device)
    print(f"  Model ready in {time.time()-t0:.1f}s; extracting...")
    embeddings = neuro.collect_text_activations(
        sub_texts, tokenizer, model, device, batch_size, max_length)
    print(f"  shape {embeddings.shape}")

    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=embeddings, window_start_sec=win_start,
             n_caps_per_window=caps, win_sec=win_sec, eeg_model=eeg_model,
             model_name=model_name, n_layers=embeddings.shape[0])
    print(f"  Saved → {out_path}")

    del model, tokenizer
    gc.collect()
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generic LLM caption embeddings aligned to any EEG window scheme.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eeg-model", required=True, choices=list(WINDOW_SCHEMES))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", metavar="HF_MODEL_ID")
    g.add_argument("--all-bloom", action="store_true")
    g.add_argument("--all-openllama", action="store_true")
    g.add_argument("--all-llama", action="store_true")
    g.add_argument("--stats", action="store_true",
                   help="Print caption/window statistics only (no model loaded).")
    ap.add_argument("--out-dir", default="embeddings/llms", metavar="PATH")
    ap.add_argument("--caption-cache", default="data/captions-qwen-2.5-vl-7b.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--test", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scheme = WINDOW_SCHEMES[args.eeg_model]
    win_sec, n_windows = scheme["win_sec"], scheme["n_windows"]

    captions = neuro.load_captions(Path(args.caption_cache).expanduser().resolve())
    texts, n_caps = build_window_texts(captions, win_sec, n_windows)

    if args.stats:
        caps = np.array(n_caps)
        clens = np.array([len(t) for t in texts])
        print(f"\n[{args.eeg_model}] {win_sec}s × {n_windows} windows")
        print(f"  captions/window: min={caps.min()} max={caps.max()} mean={caps.mean():.2f}")
        print(f"  text chars/window: mean={clens.mean():.0f} p95={np.percentile(clens,95):.0f} "
              f"max={clens.max()}  (~{clens.mean()/4:.0f} tokens mean)")
        return

    if args.all_bloom:
        models = neuro.BLOOM_MODELS
    elif args.all_openllama:
        models = neuro.OPENLLAMA_MODELS
    elif args.all_llama:
        models = LLAMA_MODELS
    else:
        models = [args.model]

    out_dir = Path(args.out_dir).expanduser().resolve()
    print(f"Device : {device}\nOutput : {out_dir}/{args.eeg_model}/\nModels : {models}")
    for m in models:
        run(m, texts, n_caps, win_sec, args.eeg_model, out_dir, device,
            args.batch_size, args.max_length, args.test)
    print("\nDone.")


if __name__ == "__main__":
    main()
