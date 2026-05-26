"""
extract_llm_captions.py

Extract layerwise LLM embeddings from CineBrain video captions,
aligned to NeuroLM 8-second windows (two consecutive 4-second captions
concatenated with a space, exactly as Aristotelian concatenates multi-caption
inputs in v2t_experiment.py).

Methodology matches Aristotelian (mlbio-epfl/Aristotelian) exactly:
  - AutoModelForCausalLM with output_hidden_states=True
  - Left-padding, eos_token used as pad_token
  - Masked mean-pooling over valid (non-padding) token positions
  - All hidden layers extracted (layer 0 = embedding, layers 1..N = transformer blocks)
  - No token truncation (Aristotelian passes max_length=None)

Caption alignment to NeuroLM 8-second windows:
  - 8100 CineBrain clips (4 s each) are sorted by numeric clip ID.
  - First 2700 sorted clips correspond to Season 7 (first 10 episodes).
  - NeuroLM window w (0-indexed) spans clips 2w and 2w+1.
    Text input = caption[2w] + " " + caption[2w+1]
  - 1350 windows total, matching N_WINDOWS in extract_neurolm.py.

Models supported (Aristotelian "videoprh" modelset):
  BLOOM (instruction-tuned bloomz variants):
    bigscience/bloomz-560m  (~1.1 GB bf16)
    bigscience/bloomz-1b1   (~2.1 GB bf16)
    bigscience/bloomz-1b7   (~3.4 GB bf16)
    bigscience/bloomz-3b    (~5.5 GB bf16)
    bigscience/bloomz-7b1   (~13 GB bf16)
  OpenLLaMA:
    openlm-research/open_llama_3b   (~5.5 GB bf16)
    openlm-research/open_llama_7b   (~13 GB bf16)
    openlm-research/open_llama_13b  (~24 GB bf16)

Output: embeddings/<stem>_layerwise.npz
  embeddings    (n_layers, W=1350, D)  float32
  window_starts (W,)   int64   EEG segment index for each window (0, 10, …, 13490)
  model_name    str
  n_layers      int

Usage:
  # Single model:
  python extract_llm_captions.py --model bigscience/bloomz-560m

  # All bloomz variants sequentially:
  python extract_llm_captions.py --all-bloom

  # All OpenLLaMA variants sequentially:
  python extract_llm_captions.py --all-openllama

  # Quick test (50 random windows, no file saved):
  python extract_llm_captions.py --model bigscience/bloomz-560m --test

  # Show caption length statistics only (no model loading):
  python extract_llm_captions.py --stats

  # Custom batch size and output dir:
  python extract_llm_captions.py --model bigscience/bloomz-3b \\
      --batch-size 4 --out-dir /data/embeddings

Requirements:
  pip install torch transformers huggingface_hub requests
  (no GPU required but strongly recommended for 3b+ models)

HuggingFace token:
  If the dataset is gated, set HF_TOKEN env variable before running.
  The captions JSON is fetched from HF and cached in --caption-cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Dataset / window constants (match extract_neurolm.py) ─────────────────────
HF_DATASET_REPO  = "Fudan-fMRI/CineBrain"
CAPTIONS_FILE    = "captions-qwen-2.5-vl-7b.json"
SEASON7_N_CLIPS  = 2700   # first 2700 clips (sorted by ID) = first 10 episodes
N_WINDOWS        = 1350   # = SEASON7_N_CLIPS // 2
SEGS_PER_WINDOW  = 10     # EEG segments per 8-second window

# ── Model registry (Aristotelian "videoprh" modelset) ─────────────────────────
BLOOM_MODELS = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b",
    "bigscience/bloomz-7b1",
]

OPENLLAMA_MODELS = [
    "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b",
    "openlm-research/open_llama_13b",
]

# ── Output stem mapping ────────────────────────────────────────────────────────
def _model_stem(model_name: str) -> str:
    """Convert HuggingFace model ID to a short filename stem."""
    stem = model_name.split("/")[-1]        # e.g. "bloomz-560m"
    stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)
    return stem


# ─────────────────────────────────────────────────────────────────────────────
# Caption loading and pairing
# ─────────────────────────────────────────────────────────────────────────────

def _clip_id(video_path: str) -> int:
    """Extract numeric clip ID from e.g. './videos/001450.mp4' → 1450."""
    m = re.search(r"(\d+)", Path(video_path).stem)
    if m is None:
        raise ValueError(f"Cannot parse clip ID from: {video_path}")
    return int(m.group(1))


def load_captions(cache_path: Path) -> List[str]:
    """
    Load captions from cache or download from HuggingFace.

    Returns a list of 2700 caption strings sorted by clip ID (Season 7 only),
    so index i corresponds to clip i and NeuroLM window w uses captions[2w] and
    captions[2w+1].
    """
    if not cache_path.exists():
        print(f"Downloading {CAPTIONS_FILE} from HuggingFace to {cache_path} ...")
        import requests
        import warnings
        url = (
            f"https://huggingface.co/datasets/{HF_DATASET_REPO}"
            f"/resolve/main/{CAPTIONS_FILE}"
        )
        token = os.environ.get("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = requests.get(url, headers=headers, verify=False, timeout=300)
        r.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(r.content)
        print("  Done.")

    print(f"Loading captions from {cache_path} ...")
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)

    # Sort by numeric clip ID so temporal order is correct
    data.sort(key=lambda e: _clip_id(e["video"]))

    total = len(data)
    print(f"  {total} total clips; taking first {SEASON7_N_CLIPS} (Season 7).")

    if total < SEASON7_N_CLIPS:
        raise RuntimeError(
            f"Expected at least {SEASON7_N_CLIPS} clips, got {total}."
        )

    season7 = data[:SEASON7_N_CLIPS]
    texts = [e["text"] for e in season7]

    # Sanity check: all texts non-empty
    empty = sum(1 for t in texts if not t or not t.strip())
    if empty:
        print(f"  WARNING: {empty} empty captions in Season 7 slice.")

    return texts


def build_paired_texts(captions: List[str]) -> List[str]:
    """
    Concatenate consecutive caption pairs to form 8-second window inputs.

    Matches Aristotelian's multi-caption aggregation:
        concatenated = " ".join(sample_caps)   # v2t_experiment.py

    Returns 1350 strings: captions[0]+" "+captions[1], captions[2]+" "+captions[3], ...
    """
    assert len(captions) == SEASON7_N_CLIPS, (
        f"Expected {SEASON7_N_CLIPS} captions, got {len(captions)}"
    )
    paired = [
        captions[2 * w] + " " + captions[2 * w + 1]
        for w in range(N_WINDOWS)
    ]
    return paired


# ─────────────────────────────────────────────────────────────────────────────
# Caption statistics (called by --stats)
# ─────────────────────────────────────────────────────────────────────────────

def print_caption_stats(captions: List[str], paired: List[str]) -> None:
    """Print character and estimated token length statistics."""

    def _stats(lengths: List[int], label: str) -> None:
        s = sorted(lengths)
        print(f"\n{label}:")
        print(f"  n       = {len(s)}")
        print(f"  min     = {min(s)}")
        print(f"  max     = {max(s)}")
        print(f"  mean    = {statistics.mean(s):.0f}")
        print(f"  median  = {statistics.median(s):.0f}")
        print(f"  p95     = {s[int(0.95 * len(s))]}")
        print(f"  p99     = {s[int(0.99 * len(s))]}")

    char_single = [len(t) for t in captions]
    char_paired = [len(t) for t in paired]
    tok_single  = [c // 4 for c in char_single]   # rough: ~4 chars/token
    tok_paired  = [c // 4 for c in char_paired]

    _stats(char_single, "Single caption character lengths (4s clips)")
    _stats(char_paired, "Paired caption character lengths (8s windows)")
    _stats(tok_single,  "Estimated tokens — single (chars/4)")
    _stats(tok_paired,  "Estimated tokens — paired  (chars/4)")

    print("\nContext window limits:")
    print("  BLOOM / OpenLLaMA: 2048 tokens")
    p99_tok = sorted(tok_paired)[int(0.99 * len(tok_paired))]
    if p99_tok < 2048:
        print(f"  P99 paired tokens = {p99_tok} → fits without truncation.")
    else:
        print(f"  WARNING: P99 paired tokens = {p99_tok} exceeds 2048 — "
              "consider --max-length.")

    print("\nAristotelian comparison:")
    print("  PE-Video human captions: ~50-100 tokens (short 1-2 sentence descriptions)")
    print(f"  CineBrain Qwen captions: ~{statistics.mean(tok_single):.0f} tokens mean "
          f"(detailed scene descriptions)")
    print("  Both fit within context. Aristotelian uses no truncation (max_length=None).")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (Aristotelian prh_models.py: load_text_model)
# ─────────────────────────────────────────────────────────────────────────────

def load_text_model(model_name: str, device: str):
    """
    Load tokenizer and causal LM exactly as Aristotelian does in prh_models.py.

    Key settings:
      - output_hidden_states=True  (needed to get per-layer representations)
      - padding_side = "left"      (standard for autoregressive LMs at inference)
      - pad_token = eos_token      (BLOOM and OpenLLaMA have no dedicated pad)
      - bfloat16 on GPU when supported, float32 otherwise
    """
    print(f"  Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if "huggyllama" in model_name:
        tokenizer.pad_token = "[PAD]"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = None
    if device != "cpu" and torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

    print(f"  Loading model (dtype={torch_dtype}) ...")
    if torch_dtype is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True, torch_dtype=torch_dtype
        )

    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e9:.2f}B")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# Token mean-pooling (Aristotelian prh_pipeline.py: _pool_tokens_masked)
# ─────────────────────────────────────────────────────────────────────────────

def _pool_tokens_masked(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked mean-pool over the sequence dimension.

    x    : (B, T, D)
    mask : (B, T)   attention_mask from tokenizer (1=real, 0=padding)
    Returns (B, D).
    """
    m = mask.unsqueeze(-1).float()           # (B, T, 1)
    return (x * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction (Aristotelian prh_pipeline.py: collect_text_activations)
# ─────────────────────────────────────────────────────────────────────────────

def collect_text_activations(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Extract mean-pooled hidden states for every layer.

    Matches Aristotelian's collect_text_activations() exactly:
      - Tokenize with left-padding, optional truncation
      - Forward pass with output_hidden_states=True
      - Masked mean-pool each layer's output over sequence tokens
      - Return stacked array (n_layers, n_texts, D)

    n_layers = len(model outputs hidden_states) = 1 (embedding) + n_transformer_blocks
    """
    batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
    acts: Optional[List[List[torch.Tensor]]] = None
    n_layers_seen = 0

    for batch_idx, batch in enumerate(batches):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        hidden = out.hidden_states   # tuple of n_layers tensors, each (B, T, D)
        mask   = enc["attention_mask"]

        if acts is None:
            n_layers_seen = len(hidden)
            acts = [[] for _ in range(n_layers_seen)]

        for layer_idx in range(n_layers_seen):
            pooled = _pool_tokens_masked(hidden[layer_idx], mask)  # (B, D)
            acts[layer_idx].append(pooled.detach().cpu())

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx+1}/{len(batches)}")

    if acts is None:
        raise RuntimeError("No texts processed.")

    # Stack: each element is list of (B, D) tensors → cat → (n_texts, D)
    # Then stack layers → (n_layers, n_texts, D)
    layer_arrays = [
        torch.cat(layer_chunks, dim=0).float().numpy()   # (n_texts, D)
        for layer_chunks in acts
    ]
    return np.stack(layer_arrays, axis=0)   # (n_layers, n_texts, D)


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def run(
    model_name: str,
    paired_texts: List[str],
    out_dir: Path,
    device: str,
    batch_size: int,
    max_length: Optional[int],
    test: bool,
) -> None:
    stem     = _model_stem(model_name)
    suffix   = "_test" if test else "_layerwise"
    out_path = out_dir / f"{stem}{suffix}.npz"

    if out_path.exists():
        print(f"\n  {out_path.name} already exists — skipping.")
        return

    # In test mode: 50 random window indices
    if test:
        rng     = random.Random(0)
        indices = sorted(rng.sample(range(N_WINDOWS), 50))
        texts   = [paired_texts[i] for i in indices]
        window_starts = np.array(
            [i * SEGS_PER_WINDOW for i in indices], dtype=np.int64
        )
        print(f"\n  TEST MODE: {len(texts)} random windows")
    else:
        texts   = paired_texts
        indices = list(range(N_WINDOWS))
        window_starts = np.arange(N_WINDOWS, dtype=np.int64) * SEGS_PER_WINDOW

    print(f"\n{'='*64}")
    print(f"  {model_name}")
    print(f"  windows={len(texts)}  batch_size={batch_size}  device={device}")
    print(f"  max_length={max_length}  output={out_path.name}")
    print(f"{'='*64}")

    t0 = time.time()
    tokenizer, model = load_text_model(model_name, device)
    print(f"  Model ready in {time.time()-t0:.1f} s")

    print(f"  Extracting activations ...")
    t1 = time.time()
    embeddings = collect_text_activations(
        texts, tokenizer, model, device, batch_size, max_length
    )
    # embeddings: (n_layers, n_windows, D)
    elapsed = time.time() - t1
    n_layers = embeddings.shape[0]
    D        = embeddings.shape[2]
    print(f"  Done in {elapsed:.1f} s — shape {embeddings.shape}")
    print(f"  n_layers={n_layers}  D={D}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=embeddings,
        window_starts=window_starts,
        model_name=model_name,
        n_layers=n_layers,
    )
    print(f"  Saved → {out_path}")

    del model, tokenizer
    gc.collect()
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract layerwise LLM embeddings from CineBrain captions "
                    "(aligned to NeuroLM 8-second windows).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_group = ap.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", metavar="HF_MODEL_ID",
        help="Single HuggingFace model ID to run.",
    )
    model_group.add_argument(
        "--all-bloom", action="store_true",
        help="Run all bloomz variants sequentially (560m → 7b1).",
    )
    model_group.add_argument(
        "--all-openllama", action="store_true",
        help="Run all OpenLLaMA variants sequentially (3b → 13b).",
    )
    model_group.add_argument(
        "--stats", action="store_true",
        help="Print caption length statistics only (no model loaded).",
    )
    ap.add_argument(
        "--out-dir", default="embeddings", metavar="PATH",
        help="Output directory for .npz files.",
    )
    ap.add_argument(
        "--caption-cache", default="data/captions-qwen-2.5-vl-7b.json",
        metavar="PATH",
        help="Local path to cache the captions JSON.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=8, metavar="N",
        help="Texts per forward pass. Reduce for large models (e.g. 1 or 2 for 7b+).",
    )
    ap.add_argument(
        "--max-length", type=int, default=None, metavar="N",
        help="Tokenizer max_length (truncation). None = no truncation, "
             "matching Aristotelian. P99 paired tokens ≈ 989 << 2048 context, "
             "so truncation is not needed.",
    )
    ap.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="Torch device (e.g. 'cuda', 'cuda:1', 'cpu'). "
             "Defaults to CUDA if available.",
    )
    ap.add_argument(
        "--test", action="store_true",
        help="Test mode: 50 random windows, no file saved.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    device = (
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir      = Path(args.out_dir).expanduser().resolve()
    caption_path = Path(args.caption_cache).expanduser().resolve()

    captions    = load_captions(caption_path)
    paired      = build_paired_texts(captions)

    if args.stats:
        print_caption_stats(captions, paired)
        return

    if args.all_bloom:
        models = BLOOM_MODELS
    elif args.all_openllama:
        models = OPENLLAMA_MODELS
    elif args.model:
        models = [args.model]
    else:
        print("Specify --model, --all-bloom, --all-openllama, or --stats.")
        return

    print(f"\nDevice : {device}")
    print(f"Output : {out_dir}")
    print(f"Windows: {N_WINDOWS} × 8 s  (Season 7, first 10 episodes)")
    print(f"Models : {models}")

    for model_name in models:
        run(
            model_name  = model_name,
            paired_texts= paired,
            out_dir     = out_dir,
            device      = device,
            batch_size  = args.batch_size,
            max_length  = args.max_length,
            test        = args.test,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
