"""
extract_vjepa2.py

Extract layerwise V-JEPA 2 embeddings for video windows aligned to EEG windows
of any length. Mirrors extract_videomae.py in structure, output layout, and
window→clip mapping, but uses Meta's V-JEPA 2 video encoder.

Key differences vs extract_videomae.py:
  * Spatial resolution: 256 (V-JEPA 2 native), not 224.
  * Variable frame count per window — NOT uniformly downsampled to 16.
    Source clips are 8 fps and decode to 33 frames each (one terminal frame
    beyond the nominal 4 s); concatenated across windows tiled at CLIP_SECONDS
    = 4.0 s this typically yields:
        5 s →  42 frames   6 s →  50 frames
        8 s →  66 frames  10 s →  82 frames  (trimmed to 82 — see guard below)
    All counts must be even (tubelet_size=2); the guard at the end of
    _frames_for_window drops a final frame when necessary. The range lies
    within or near the
    16–64 frame range V-JEPA 2 was actually pretrained on (Bardes et al. 2025
    §2.4: progressive-resolution training at 16 → 64 frames; explicitly tested
    up to 128 / 256 frames at inference). 3D-RoPE positional encoding handles
    the length variation natively.
  * Model: facebook/vjepa2-vit{l,h,g}-fpc64-256 via AutoModel.  No q_bias/
    v_bias key remapping — V-JEPA 2 ships with modern HF weight names.
  * Forward uses skip_predictor=True (we only need the encoder) and
    output_hidden_states=True; per-layer mean-pool over all spatiotemporal
    patch tokens (V-JEPA 2 has no CLS token).

Time alignment per EEG model (D = window_seconds):
    FEMBA, LUNA       :  5 s →  1.25 clips
    ST-EEGFormer      :  6 s →  1.5  clips
    NeuroLM           :  8 s →  2    clips (exact alignment)
    REVE              : 10 s →  2.5  clips

Source data, window source, output association: identical to extract_videomae.py.

Output: embeddings/vjepa2_{size}__{eeg_family}.npz
    embeddings      (n_layers, W, D_emb)  float32
    window_starts_s (W,)        float64
    window_seconds  scalar      int
    size            scalar      str  ('large' / 'huge' / 'giant')
    n_layers        scalar      int
    eeg_family      scalar      str

Embedding dim D_emb per size:  large=1024  huge=1280  giant=1408
n_layers per size:             large=24    huge=32    giant=40

Usage:
    # 5 s FEMBA + LUNA shared
    python extract_vjepa2.py --eeg-family femba_luna --window-seconds 5
    # → vjepa2_large__femba_luna.npz, vjepa2_huge__femba_luna.npz,
    #   vjepa2_giant__femba_luna.npz

    # Single size:
    python extract_vjepa2.py --size large --eeg-family neurolm --window-seconds 8

    # Half-precision (recommended for huge / giant on tight VRAM):
    python extract_vjepa2.py --eeg-family reve --window-seconds 10 --dtype fp16

VRAM (single window, fp32, 80-frame worst case):
    sequence_length = (80/2) * (256/16)^2 = 40 * 256 = 10 240 tokens
    large  ~  3 GB activations
    huge   ~  5 GB
    giant  ~  8 GB
    Switch to fp16 for headroom; weights are released as fp32 but the encoder
    runs cleanly in fp16 with sdpa attention.

Requirements:
    pip install opencv-python "transformers>=4.53" torch numpy huggingface_hub
"""

from __future__ import annotations

import argparse
import gc
import os
import tarfile
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

# ── HuggingFace dataset configuration ────────────────────────────────────────
HF_DATASET_REPO = "Fudan-fMRI/CineBrain"
HF_VIDEOS_TAR   = "videos.tar"

CB_ROOT      = Path(__file__).parents[1]
EMBED_DIR    = CB_ROOT / "embeddings"
CLIP_CACHE   = CB_ROOT / ".clip_cache"
CLIP_MARKER  = CLIP_CACHE / ".extracted"
CLIP_SECONDS = 4.0
# Pre-extracted clips ONLY consulted in --test mode (so smoke tests on a dev
# machine don't have to re-download 2.6 GB). Production / server runs always
# pull videos.tar from HuggingFace via _ensure_clips_extracted().
CLIP_DIR_LOCAL_TEST = CB_ROOT / "Data" / "clips"

# Season 7 (the season participants watched during EEG recording) spans clips
# 0..2699 inclusive. Clips 2700..8099 must not appear in cross-modal alignment.
MAX_CLIP_IDX   = 2699
SEASON_END_SEC = (MAX_CLIP_IDX + 1) * CLIP_SECONDS    # 10800 s

HF_MODELS = {
    "large": "facebook/vjepa2-vitl-fpc64-256",
    "huge":  "facebook/vjepa2-vith-fpc64-256",
    "giant": "facebook/vjepa2-vitg-fpc64-256",
}

FRAME_SIZE   = 256   # V-JEPA 2 native input resolution
TUBELET_SIZE = 2     # frame count MUST be even

# Source video fps inside videos.tar (CineBrain clips are 8 fps × 4 s = 32
# frames per clip). Used only for documentation / sanity prints.
SOURCE_FPS = 8


# ─────────────────────────────────────────────────────────────────────────────
# Token resolution (env var → file → home directory)
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_SEARCH_PATHS: list[Path] = [
    Path(__file__).parent / "hf_token.txt",
    Path.home() / "hf_token.txt",
]

def _get_hf_token() -> str | None:
    if token := os.environ.get("HF_TOKEN"):
        return token.strip()
    if path_str := os.environ.get("HF_TOKEN_FILE"):
        p = Path(path_str)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        raise FileNotFoundError(f"HF_TOKEN_FILE set but file not found: {p}")
    for p in _TOKEN_SEARCH_PATHS:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# videos.tar — download once, extract once, then index by clip index
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_clips_extracted() -> Path:
    if CLIP_MARKER.exists():
        return CLIP_CACHE

    from huggingface_hub import hf_hub_download
    print(f"  Downloading {HF_VIDEOS_TAR} from HuggingFace (2.59 GB, "
          f"cached after first run)...")
    tar_path = Path(hf_hub_download(
        repo_id   = HF_DATASET_REPO,
        filename  = HF_VIDEOS_TAR,
        repo_type = "dataset",
        token     = _get_hf_token(),
    ))

    print(f"  Extracting clips to {CLIP_CACHE} (one-time, ~2.6 GB)...")
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(CLIP_CACHE)
    CLIP_MARKER.touch()
    print(f"  Extraction complete.")
    return CLIP_CACHE


_CLIP_INDEX: dict[int, Path] = {}
_USE_LOCAL_CLIPS = False  # set to True only by run(..., test=True)

def _resolve_clip_path(clip_idx: int) -> Path:
    if not _CLIP_INDEX:
        if _USE_LOCAL_CLIPS:
            clips_dir = CLIP_DIR_LOCAL_TEST
            if not (clips_dir.exists() and any(clips_dir.glob("*.mp4"))):
                raise RuntimeError(
                    f"--test requested local clips at {clips_dir} but none "
                    f"were found there."
                )
            print(f"  [test mode] using local clips from {clips_dir}")
        else:
            clips_dir = _ensure_clips_extracted()
        for p in clips_dir.rglob("*.mp4"):
            digits = "".join(c for c in p.stem if c.isdigit())
            if digits:
                _CLIP_INDEX[int(digits)] = p
        if not _CLIP_INDEX:
            raise RuntimeError(
                f"No .mp4 files found under {clips_dir}. "
                "Tar may have a different layout than expected."
            )
        print(f"  Indexed {len(_CLIP_INDEX)} clips "
              f"(range {min(_CLIP_INDEX)}..{max(_CLIP_INDEX)})")

    if clip_idx not in _CLIP_INDEX:
        raise KeyError(
            f"Clip index {clip_idx} not in extracted tar "
            f"(have {min(_CLIP_INDEX)}..{max(_CLIP_INDEX)})."
        )
    return _CLIP_INDEX[clip_idx]


@lru_cache(maxsize=8)
def _load_clip_frames(clip_idx: int) -> tuple:
    """Decode all frames of clip clip_idx as (H, W, 3) uint8 RGB at 256×256."""
    path = _resolve_clip_path(clip_idx)
    cap  = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open clip {clip_idx} at {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Clip {clip_idx} decoded to 0 frames")
    return tuple(frames)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Season 7 window starts (in seconds)
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_starts_s(window_seconds: int) -> np.ndarray:
    """Non-overlapping window starts tiling Season 7."""
    return np.arange(0.0, SEASON_END_SEC, float(window_seconds), dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Window → frames  (variable length; tubelet_size=2 → must be even)
# ─────────────────────────────────────────────────────────────────────────────

def _frames_for_window(start_sec: float, window_sec: float) -> list[np.ndarray]:
    """
    Concatenate all native frames covering [start_sec, start_sec + window_sec)
    by pulling whole or partial clips. No uniform downsampling — V-JEPA 2 is
    fed all available frames so the effective sampling rate matches the source
    (8 fps), keeping motion statistics in the regime the encoder was trained on.

    CineBrain clips decode to 33 frames @ 8 fps, so frame counts per window
    (before the even-count guard) are typically:
        5 s window:  33 +  9 =  42 frames
        6 s window:  33 + 17 =  50 frames
        8 s window:  33 + 33 =  66 frames    (close to pretrain length 64)
       10 s window:  33 + 33 + 17 = 83 → 82 frames (final frame trimmed)
    """
    end_sec    = start_sec + window_sec
    first_clip = int(start_sec // CLIP_SECONDS)
    last_clip  = int(np.ceil(end_sec / CLIP_SECONDS)) - 1

    if last_clip > MAX_CLIP_IDX:
        raise ValueError(
            f"Window [{start_sec:.2f}, {end_sec:.2f})s requires clip "
            f"{last_clip}, but Season 7 ends at clip {MAX_CLIP_IDX} "
            f"(t={SEASON_END_SEC:.0f}s)."
        )

    frames: list[np.ndarray] = []
    for c in range(first_clip, last_clip + 1):
        clip_frames = _load_clip_frames(c)
        n_frames    = len(clip_frames)
        clip_t0     = c * CLIP_SECONDS
        ov_t0       = max(start_sec, clip_t0)
        ov_t1       = min(end_sec,   clip_t0 + CLIP_SECONDS)
        f_start     = int(np.floor((ov_t0 - clip_t0) / CLIP_SECONDS * n_frames))
        f_end       = int(np.ceil ((ov_t1 - clip_t0) / CLIP_SECONDS * n_frames))
        f_start     = max(0, f_start)
        f_end       = min(n_frames, f_end)
        frames.extend(clip_frames[f_start:f_end])

    # V-JEPA 2 tubelet_size = 2 → number of frames must be even.  At 8 fps with
    # integer-second windows this is already satisfied for 5/6/8/10 s, but the
    # floor/ceil rounding above can in principle produce an odd count at clip
    # boundaries when the source fps is not exactly 8.  Guard by dropping the
    # final frame if needed.
    if len(frames) % TUBELET_SIZE != 0:
        frames = frames[: len(frames) - (len(frames) % TUBELET_SIZE)]

    if len(frames) < 4:
        raise RuntimeError(
            f"Only {len(frames)} frames for window starting {start_sec:.2f}s "
            f"(need ≥ 4 for V-JEPA 2 with tubelet_size=2)."
        )
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# V-JEPA 2 forward pass
# ─────────────────────────────────────────────────────────────────────────────

def _embed_frames(processor, model, device, dtype, frames: list[np.ndarray]) -> np.ndarray:
    """
    Run a variable-length frame list through V-JEPA 2 and return one mean-pooled
    vector per transformer block: shape (n_layers, D).

    The processor accepts an array of shape (T, H, W, C) uint8 RGB (matches
    the HF docs example `video = np.ones((64, 256, 256, 3))`) and produces
    `pixel_values_videos` of shape (1, T, C, H, W).

    `skip_predictor=True` skips the predictor head (we only need the encoder).
    `output_hidden_states=True` returns a tuple of length (num_hidden_layers + 1):
    the first element is the input embedding, the rest are the post-block
    hidden states.  We drop the first to match the EEG-extractor convention
    (and extract_videomae.py:307).

    V-JEPA 2 has no CLS token — mean-pool over all spatiotemporal patch tokens.
    """
    video = np.stack(frames, axis=0)  # (T, H, W, 3) uint8 RGB
    inputs = processor(video, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
              for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            skip_predictor       = True,
            output_hidden_states = True,
        )

    # hidden_states: tuple of (num_hidden_layers + 1) tensors, each (1, N_tok, D)
    layers = outputs.hidden_states[1:]
    pooled = [h.mean(dim=1).squeeze(0) for h in layers]   # each (D,)
    return torch.stack(pooled, dim=0).float().cpu().numpy()   # (n_layers, D)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

EEG_FAMILIES = ("femba_luna", "steegformer", "neurolm", "reve")

_TORCH_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def run(
    size: str,
    eeg_family: str,
    window_seconds: int,
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    test: bool = False,
) -> None:
    # Always use HF data; _USE_LOCAL_CLIPS stays False
    suffix   = "__TEST" if test else ""
    out_path = out_dir / f"vjepa2_{size}__{eeg_family}{suffix}.npz"

    if out_path.exists():
        print(f"\n  {out_path.name} already exists — skipping.")
        return

    starts_s = _canonical_starts_s(window_seconds)
    if test:
        starts_s = starts_s[:5]   # first 5 windows only
    W        = len(starts_s)
    last_end = float(starts_s[-1]) + window_seconds

    expected_frames = window_seconds * SOURCE_FPS

    print(f"\n{'='*72}")
    print(f"  V-JEPA 2 {size}  |  window={window_seconds} s "
          f"({window_seconds / CLIP_SECONDS:.2f} clips, "
          f"~{expected_frames} frames @ {SOURCE_FPS} fps)  "
          f"|  EEG family='{eeg_family}'")
    print(f"  output    : {out_path.name}")
    print(f"  windows={W}  device={device}  dtype={dtype}")
    print(f"  time range: [{float(starts_s[0]):.2f}, {last_end:.2f}] s "
          f"(within Season 7: clips 0..{MAX_CLIP_IDX})")
    print(f"{'='*72}")

    repo_id = HF_MODELS[size]
    print(f"  Loading {repo_id}...")
    t0        = time.time()
    processor = AutoVideoProcessor.from_pretrained(repo_id, token=_get_hf_token())
    model     = AutoModel.from_pretrained(
        repo_id,
        torch_dtype       = dtype,
        attn_implementation = "sdpa",
        token             = _get_hf_token(),
    )
    model = model.to(device).eval()

    D        = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded in {time.time() - t0:.1f} s  |  n_layers={n_layers}  D={D}  "
          f"tubelet={model.config.tubelet_size}  patch={model.config.patch_size}  "
          f"crop={model.config.crop_size}")

    embeddings = np.zeros((n_layers, W, D), dtype=np.float32)
    t_loop     = time.time()

    for w_idx, start_sec in enumerate(starts_s):
        frames = _frames_for_window(float(start_sec), float(window_seconds))
        embeddings[:, w_idx] = _embed_frames(
            processor, model, device, dtype, frames
        )

        if (w_idx + 1) % 50 == 0 or w_idx + 1 == W:
            elapsed = time.time() - t_loop
            rate    = (w_idx + 1) / elapsed
            eta     = (W - w_idx - 1) / rate
            print(f"    {w_idx+1:>4}/{W}  t={start_sec:>8.2f}s  "
                  f"frames={len(frames):>3}  {rate:.2f} win/s  "
                  f"ETA {eta/60:.1f} min")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings      = embeddings,
        window_starts_s = starts_s,
        window_seconds  = window_seconds,
        size            = size,
        n_layers        = n_layers,
        eeg_family      = eeg_family,
    )
    print(f"\n  Saved {embeddings.shape} (n_layers, W, D) → {out_path.name}")

    del model, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract V-JEPA 2 embeddings aligned to EEG windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--size", choices=list(HF_MODELS), default=None,
        help="Model size. Omit to run all three (large → huge → giant).",
    )
    ap.add_argument(
        "--eeg-family", required=True, choices=EEG_FAMILIES, metavar="NAME",
        help="EEG model family this output is paired with (used for the "
             "filename tag and metadata). Choices: " + ", ".join(EEG_FAMILIES),
    )
    ap.add_argument(
        "--window-seconds", required=True, type=int, metavar="N",
        help="EEG window length in seconds (5 for femba_luna, 6 for "
             "steegformer, 8 for neurolm, 10 for reve).",
    )
    ap.add_argument(
        "--out-dir", default="embeddings", type=Path, metavar="PATH",
        help="Output directory for .npz files.",
    )
    ap.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="Torch device (e.g. 'cuda', 'cuda:1', 'cpu'). "
             "Defaults to CUDA if available.",
    )
    ap.add_argument(
        "--dtype", choices=list(_TORCH_DTYPES), default="fp32", metavar="DTYPE",
        help="Compute dtype for the encoder. Use fp16 / bf16 to fit huge or "
             "giant on tighter VRAM. Output embeddings are always saved as fp32.",
    )
    ap.add_argument(
        "--test", action="store_true",
        help="Smoke-test mode: process only the first 5 windows and save with "
             "a '__TEST' suffix so real outputs are never overwritten.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = _TORCH_DTYPES[args.dtype]

    if device.type == "cpu" and dtype != torch.float32:
        print(f"  Warning: dtype={args.dtype} on CPU is not well supported — "
              f"forcing fp32.")
        dtype = torch.float32

    sizes = [args.size] if args.size else ["large", "huge", "giant"]
    for size in sizes:
        run(size, args.eeg_family, args.window_seconds, out_dir, device, dtype,
            test=args.test)

    print("\nDone.")


if __name__ == "__main__":
    main()
