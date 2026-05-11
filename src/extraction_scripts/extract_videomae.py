"""
extract_videomae.py

Extract layerwise VideoMAE embeddings for video windows aligned to EEG windows
of any length. For each EEG window of D seconds the script:
  1. Loads the corresponding video range from the CineBrain `videos.tar`
     (8100 clips, each clip = 4 s),
  2. Concatenates the relevant frames (whole and/or partial clips),
  3. Uniformly samples 16 frames,
  4. Runs them through VideoMAE with output_hidden_states=True and, for every
     transformer block, mean-pools the spatiotemporal tokens → one (D,) vector
     per layer per window.

So the output is layerwise — shape (n_layers, W, D) — mirroring the EEG
extractors' (n_layers, W, S, D), so cross-modal alignment can be measured
between every pair of layers (VideoMAE layer i ↔ EEG-model layer j), as done
in the Platonic / Aristotelian representation-alignment papers, rather than
only against a single (last) layer.

n_layers = model.config.num_hidden_layers (the encoder-block outputs;
hidden_states[0], the input-embedding output, is dropped — matching the EEG
extractors, which capture post-block hidden states only).

Time alignment per EEG model (D = window_seconds):
    FEMBA, LUNA       :  5 s →  1.25 clips
    ST-EEGFormer      :  6 s →  1.5  clips
    NeuroLM           :  8 s →  2    clips (exact alignment)
    REVE              : 10 s →  2.5  clips

Source data:
    HuggingFace dataset Fudan-fMRI/CineBrain → `videos.tar` (2.59 GB).
    On first run the tar is downloaded (cached by huggingface_hub) and
    extracted once to CB_ROOT/.clip_cache/. Subsequent runs reuse it.

Window source:
    Windows are generated deterministically from `--window-seconds`:
        window_starts_s = np.arange(0, 10800, window_seconds)
    matching every EEG extractor's non-overlapping tiling of Season 7. No EEG
    NPZ is required — you do NOT need to have run any EEG extractor first.

Output association:
    Each output is tagged with `--eeg-family`. FEMBA and LUNA both use the
    same 5 s window tiling, so they share one VideoMAE NPZ tagged 'femba_luna'.
    Choices:
        femba_luna   — 5 s windows  (covers both FEMBA and LUNA)
        steegformer  — 6 s windows
        neurolm      — 8 s windows
        reve         — 10 s windows
    One VideoMAE NPZ per (size, family) pair.

Output: embeddings/videomae_{size}__{eeg_family}.npz
    embeddings      (n_layers, W, D_emb)  float32
    window_starts_s (W,)        float64   canonical start time in seconds
    window_seconds  scalar      int
    size            scalar      str       VideoMAE size ('base' / 'large')
    n_layers        scalar      int       number of transformer blocks
    eeg_family      scalar      str       e.g. 'neurolm', 'reve', 'femba'

Usage:
    # 5 s FEMBA + LUNA shared
    python extract_videomae.py --eeg-family femba_luna --window-seconds 5
    # → videomae_base__femba_luna.npz, videomae_large__femba_luna.npz

    # 6 s ST-EEGFormer
    python extract_videomae.py --eeg-family steegformer --window-seconds 6
    # → videomae_base__steegformer.npz, videomae_large__steegformer.npz

    # 8 s NeuroLM
    python extract_videomae.py --eeg-family neurolm --window-seconds 8

    # 10 s REVE
    python extract_videomae.py --eeg-family reve --window-seconds 10

Disk space:
    ~2.6 GB cached tar + ~2.6 GB extracted clips ≈ 5.2 GB total on first run.

Requirements:
    pip install opencv-python transformers torch numpy huggingface_hub
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
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor

# ── HuggingFace dataset configuration ────────────────────────────────────────
HF_DATASET_REPO = "Fudan-fMRI/CineBrain"
HF_VIDEOS_TAR   = "videos.tar"

CB_ROOT      = Path(__file__).parents[1]
EMBED_DIR    = CB_ROOT / "embeddings"
CLIP_CACHE   = CB_ROOT / ".clip_cache"           # extracted clips live here
CLIP_MARKER  = CLIP_CACHE / ".extracted"         # touched once tar fully unpacks
CLIP_SECONDS = 4.0                               # each clip is 4 s long

# Season 7 (the season the participants were watching during EEG recording)
# spans clips 0..2699 inclusive — i.e. the first 10 episodes × 270 clips/episode.
# Clips 2700..8099 are from other seasons not shown to the participants and
# must not appear in any cross-modal alignment.
MAX_CLIP_IDX = 2699
SEASON_END_SEC = (MAX_CLIP_IDX + 1) * CLIP_SECONDS    # 10800 s

HF_MODELS = {
    "base":  "MCG-NJU/videomae-base",
    "large": "MCG-NJU/videomae-large",
}

N_FRAMES   = 16    # VideoMAE fixed input length
FRAME_SIZE = 224   # VideoMAE expected spatial resolution


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
    """
    Download videos.tar (cached by huggingface_hub) and extract it once into
    CLIP_CACHE. Returns the cache directory.
    """
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

def _resolve_clip_path(clip_idx: int) -> Path:
    """
    Map clip index → on-disk path. The mapping is built once on first call by
    scanning CLIP_CACHE for *.mp4 files and parsing digit runs from filenames,
    so the internal naming convention inside videos.tar does not need to be
    hardcoded.
    """
    if not _CLIP_INDEX:
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
    """
    Decode all frames of clip clip_idx. Returns an immutable tuple of
    (H, W, 3) uint8 RGB arrays. lru_cache reuses results for adjacent
    windows that share clips.
    """
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
    """
    Non-overlapping window starts tiling Season 7 (0..10800 s) — matches every
    EEG extractor's window iteration regardless of underlying unit (segments
    or 1000-Hz samples), since all of them tile contiguously from t=0.
        5 s → 2160 windows
        6 s → 1800 windows
        8 s → 1350 windows
       10 s → 1080 windows
    """
    return np.arange(0.0, SEASON_END_SEC, float(window_seconds), dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Window → frames  (handles partial clips for non-integer clip counts)
# ─────────────────────────────────────────────────────────────────────────────

def _frames_for_window(start_sec: float, window_sec: float) -> list[np.ndarray]:
    """
    Concatenate the frames covering [start_sec, start_sec + window_sec) by
    pulling whole or partial clips. Each clip i covers [i*4, (i+1)*4) seconds.

    For a 10 s window starting at 0 s this yields:
      clip 0 (full, 4 s) + clip 1 (full, 4 s) + clip 2 first 2 s.
    """
    end_sec    = start_sec + window_sec
    first_clip = int(start_sec // CLIP_SECONDS)
    last_clip  = int(np.ceil(end_sec / CLIP_SECONDS)) - 1

    if last_clip > MAX_CLIP_IDX:
        raise ValueError(
            f"Window [{start_sec:.2f}, {end_sec:.2f})s requires clip "
            f"{last_clip}, but Season 7 ends at clip {MAX_CLIP_IDX} "
            f"(t={SEASON_END_SEC:.0f}s). The EEG NPZ should not contain "
            f"windows extending beyond Season 7."
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
    return frames


def _sample_uniform(frames: list[np.ndarray], n: int = N_FRAMES) -> list[np.ndarray]:
    """Uniformly sample n frames across the full available range."""
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# VideoMAE forward pass
# ─────────────────────────────────────────────────────────────────────────────

def _embed_frames(processor, model, device, frames: list[np.ndarray]) -> np.ndarray:
    """Run 16 frames through VideoMAE → (n_layers, D): for each transformer
    block, mean-pool the spatiotemporal tokens. n_layers = num_hidden_layers
    (the input embedding output hidden_states[0] is dropped). The VideoMAE
    encoder has no CLS token, so the mean is over all tokens."""
    inputs = processor(frames, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # out.hidden_states: tuple of (num_hidden_layers + 1) tensors, each (1, T, D)
    layers = out.hidden_states[1:]
    pooled = [h.mean(dim=1).squeeze(0) for h in layers]   # each (D,)
    return torch.stack(pooled, dim=0).cpu().float().numpy()   # (n_layers, D)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

EEG_FAMILIES = ("femba_luna", "steegformer", "neurolm", "reve")

def run(
    size: str,
    eeg_family: str,
    window_seconds: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    out_path = out_dir / f"videomae_{size}__{eeg_family}.npz"

    if out_path.exists():
        print(f"\n  {out_path.name} already exists — skipping.")
        return

    starts_s = _canonical_starts_s(window_seconds)
    W        = len(starts_s)
    last_end = float(starts_s[-1]) + window_seconds

    print(f"\n{'='*72}")
    print(f"  VideoMAE-{size}  |  window={window_seconds} s "
          f"({window_seconds / CLIP_SECONDS:.2f} clips)  "
          f"|  EEG family='{eeg_family}'")
    print(f"  output    : {out_path.name}")
    print(f"  windows={W}  device={device}")
    print(f"  time range: [{float(starts_s[0]):.2f}, {last_end:.2f}] s "
          f"(within Season 7: clips 0..{MAX_CLIP_IDX})")
    print(f"  embedding[i] will correspond 1-to-1 with EEG window i.")
    print(f"{'='*72}")

    print(f"  Loading {HF_MODELS[size]}...")
    t0        = time.time()
    processor = VideoMAEImageProcessor.from_pretrained(HF_MODELS[size])

    # Transformers 5.x renamed q_bias/v_bias → query.bias/value.bias but the
    # MCG-NJU checkpoints still use the old names.  Load the raw checkpoint,
    # remap keys, then initialise the model from config + load_state_dict so
    # from_pretrained never sees the mismatched names.
    from huggingface_hub import hf_hub_download
    from transformers import VideoMAEConfig
    import safetensors.torch as _st

    try:
        _ckpt = _st.load_file(
            hf_hub_download(HF_MODELS[size], "model.safetensors", repo_type="model")
        )
    except Exception:
        _ckpt = torch.load(
            hf_hub_download(HF_MODELS[size], "pytorch_model.bin", repo_type="model"),
            map_location="cpu", weights_only=True,
        )

    _remapped = {}
    for _k, _v in _ckpt.items():
        if _k.endswith(".q_bias"):
            _remapped[_k.replace(".q_bias", ".query.bias")] = _v
            # VideoMAE has no key bias; zero-init so the model loads cleanly
            _remapped[_k.replace(".q_bias", ".key.bias")] = torch.zeros_like(_v)
        elif _k.endswith(".v_bias"):
            _remapped[_k.replace(".v_bias", ".value.bias")] = _v
        else:
            _remapped[_k] = _v
    del _ckpt

    _config = VideoMAEConfig.from_pretrained(HF_MODELS[size])
    _full   = VideoMAEForPreTraining(_config)
    _missing, _unexpected = _full.load_state_dict(_remapped, strict=False)
    del _remapped
    if _missing or _unexpected:
        print(f"  load_state_dict: {len(_missing)} missing, {len(_unexpected)} unexpected keys")

    model    = _full.videomae.to(device).eval()
    D        = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    del _full
    print(f"  Loaded in {time.time() - t0:.1f} s  |  n_layers={n_layers}  D={D}")

    embeddings = np.zeros((n_layers, W, D), dtype=np.float32)
    t_loop     = time.time()

    for w_idx, start_sec in enumerate(starts_s):
        frames = _frames_for_window(float(start_sec), float(window_seconds))
        if len(frames) == 0:
            raise RuntimeError(f"No frames for window {w_idx} at t={start_sec:.2f}s")
        frames_16             = _sample_uniform(frames, N_FRAMES)
        embeddings[:, w_idx]  = _embed_frames(processor, model, device, frames_16)

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

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract VideoMAE embeddings aligned to EEG windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--size", choices=list(HF_MODELS), default=None,
        help="Model size. Omit to run both base and large.",
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
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    sizes = [args.size] if args.size else ["base", "large"]
    for size in sizes:
        run(size, args.eeg_family, args.window_seconds, out_dir, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
