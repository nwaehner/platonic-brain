"""
extract_videomae_ft_kinetics.py

Extract layerwise VideoMAE embeddings using the Kinetics-400 fine-tuned checkpoints
(base, large, huge) for video windows aligned to EEG windows of any length.

Pipeline is identical to extract_videomae.py. The only differences are:

  1. Model checkpoints — Kinetics-400 fine-tuned:
       MCG-NJU/videomae-base-finetuned-kinetics
       MCG-NJU/videomae-large-finetuned-kinetics
       MCG-NJU/videomae-huge-finetuned-kinetics

  2. The classification head is stripped; only the encoder backbone (VideoMAEModel)
     is used for feature extraction, identical to the pretrained-only variant.

  3. Output filenames are prefixed with 'videomae_ft_kinetics_' to avoid
     collisions with the pretrained-only NPZs.

Output: embeddings/videomae_ft_kinetics_{size}__{eeg_family}.npz
    embeddings      (n_layers, W, D_emb)  float32
    window_starts_s (W,)                  float64
    window_seconds  scalar                int
    size            scalar                str
    n_layers        scalar                int
    eeg_family      scalar                str

Usage:
    python extract_videomae_ft_kinetics.py --eeg-family femba_luna --window-seconds 5
    python extract_videomae_ft_kinetics.py --eeg-family steegformer --window-seconds 6
    python extract_videomae_ft_kinetics.py --eeg-family neurolm --window-seconds 8
    python extract_videomae_ft_kinetics.py --eeg-family reve --window-seconds 10

    # Single size
    python extract_videomae_ft_kinetics.py --eeg-family neurolm --window-seconds 8 --size huge
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
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# ── HuggingFace dataset configuration ────────────────────────────────────────
HF_DATASET_REPO = "Fudan-fMRI/CineBrain"
HF_VIDEOS_TAR   = "videos.tar"

CB_ROOT      = Path(__file__).parents[1]
EMBED_DIR    = CB_ROOT / "embeddings"
CLIP_CACHE   = CB_ROOT / ".clip_cache"
CLIP_MARKER  = CLIP_CACHE / ".extracted"
CLIP_SECONDS = 4.0

MAX_CLIP_IDX   = 2699
SEASON_END_SEC = (MAX_CLIP_IDX + 1) * CLIP_SECONDS    # 10 800 s

HF_MODELS = {
    "base":  "MCG-NJU/videomae-base-finetuned-kinetics",
    "large": "MCG-NJU/videomae-large-finetuned-kinetics",
    "huge":  "MCG-NJU/videomae-huge-finetuned-kinetics",
}

N_FRAMES   = 16
FRAME_SIZE = 224


# ── Token resolution ──────────────────────────────────────────────────────────

_TOKEN_SEARCH_PATHS: list[Path] = [
    CB_ROOT / "tokens" / "hf_token.txt",
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


# ── videos.tar — download once, extract once ─────────────────────────────────

def _ensure_clips_extracted() -> Path:
    if CLIP_MARKER.exists():
        return CLIP_CACHE
    from huggingface_hub import hf_hub_download
    print(f"  Downloading {HF_VIDEOS_TAR} from HuggingFace (2.59 GB, cached after first run)...")
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
    print("  Extraction complete.")
    return CLIP_CACHE


_CLIP_INDEX: dict[int, Path] = {}

def _resolve_clip_path(clip_idx: int) -> Path:
    if not _CLIP_INDEX:
        clips_dir = _ensure_clips_extracted()
        for p in clips_dir.rglob("*.mp4"):
            digits = "".join(c for c in p.stem if c.isdigit())
            if digits:
                _CLIP_INDEX[int(digits)] = p
        if not _CLIP_INDEX:
            raise RuntimeError(f"No .mp4 files found under {clips_dir}.")
        print(f"  Indexed {len(_CLIP_INDEX)} clips "
              f"(range {min(_CLIP_INDEX)}..{max(_CLIP_INDEX)})")
    if clip_idx not in _CLIP_INDEX:
        raise KeyError(f"Clip index {clip_idx} not in extracted tar.")
    return _CLIP_INDEX[clip_idx]


@lru_cache(maxsize=8)
def _load_clip_frames(clip_idx: int) -> tuple:
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


# ── Window tiling ─────────────────────────────────────────────────────────────

def _canonical_starts_s(window_seconds: int) -> np.ndarray:
    return np.arange(0.0, SEASON_END_SEC, float(window_seconds), dtype=np.float64)


def _frames_for_window(start_sec: float, window_sec: float) -> list[np.ndarray]:
    end_sec    = start_sec + window_sec
    first_clip = int(start_sec // CLIP_SECONDS)
    last_clip  = int(np.ceil(end_sec / CLIP_SECONDS)) - 1
    if last_clip > MAX_CLIP_IDX:
        raise ValueError(
            f"Window [{start_sec:.2f}, {end_sec:.2f})s requires clip "
            f"{last_clip} beyond Season 7 (clip {MAX_CLIP_IDX})."
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
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


# ── VideoMAE forward pass ─────────────────────────────────────────────────────

def _embed_frames(processor, model, device, frames: list[np.ndarray]) -> np.ndarray:
    """Run 16 frames through the VideoMAE encoder → (n_layers, D).
    Mean-pools spatiotemporal tokens per layer; drops the input-embedding output."""
    inputs = processor(frames, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    layers = out.hidden_states[1:]                              # drop embedding layer
    pooled = [h.mean(dim=1).squeeze(0) for h in layers]        # each (D,)
    return torch.stack(pooled, dim=0).cpu().float().numpy()     # (n_layers, D)


# ── Run ───────────────────────────────────────────────────────────────────────

EEG_FAMILIES = ("femba_luna", "steegformer", "neurolm", "reve")

def run(
    size: str,
    eeg_family: str,
    window_seconds: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    out_path = out_dir / f"videomae_ft_kinetics_{size}__{eeg_family}.npz"

    if out_path.exists():
        print(f"\n  {out_path.name} already exists — skipping.")
        return

    starts_s = _canonical_starts_s(window_seconds)
    W        = len(starts_s)
    last_end = float(starts_s[-1]) + window_seconds

    print(f"\n{'='*72}")
    print(f"  VideoMAE-ft-kinetics-{size}  |  window={window_seconds} s "
          f"({window_seconds / CLIP_SECONDS:.2f} clips)  |  EEG family='{eeg_family}'")
    print(f"  checkpoint : {HF_MODELS[size]}")
    print(f"  output     : {out_path.name}")
    print(f"  windows={W}  device={device}")
    print(f"  time range : [{float(starts_s[0]):.2f}, {last_end:.2f}] s")
    print(f"{'='*72}")

    print(f"  Loading {HF_MODELS[size]}...")
    t0        = time.time()
    processor = VideoMAEImageProcessor.from_pretrained(HF_MODELS[size])

    # Load VideoMAEForVideoClassification, remap legacy key names, then
    # extract only the encoder backbone (.videomae) — discarding the head.
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

    # Remap legacy attention-bias key names produced by the MCG-NJU codebase.
    _remapped = {}
    for _k, _v in _ckpt.items():
        if _k.endswith(".q_bias"):
            _remapped[_k.replace(".q_bias", ".query.bias")] = _v
            _remapped[_k.replace(".q_bias", ".key.bias")]   = torch.zeros_like(_v)
        elif _k.endswith(".v_bias"):
            _remapped[_k.replace(".v_bias", ".value.bias")] = _v
        else:
            _remapped[_k] = _v
    del _ckpt

    _config = VideoMAEConfig.from_pretrained(HF_MODELS[size])
    _full   = VideoMAEForVideoClassification(_config)
    _missing, _unexpected = _full.load_state_dict(_remapped, strict=False)
    del _remapped
    if _missing or _unexpected:
        print(f"  load_state_dict: {len(_missing)} missing, "
              f"{len(_unexpected)} unexpected keys")

    # Strip the classification head — keep only the encoder.
    model    = _full.videomae.to(device).eval()
    D        = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    del _full
    print(f"  Loaded in {time.time() - t0:.1f} s  |  n_layers={n_layers}  D={D}")

    embeddings = np.zeros((n_layers, W, D), dtype=np.float32)
    t_loop     = time.time()

    for w_idx, start_sec in enumerate(starts_s):
        frames = _frames_for_window(float(start_sec), float(window_seconds))
        if not frames:
            raise RuntimeError(f"No frames for window {w_idx} at t={start_sec:.2f}s")
        frames_16            = _sample_uniform(frames, N_FRAMES)
        embeddings[:, w_idx] = _embed_frames(processor, model, device, frames_16)

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


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract Kinetics-finetuned VideoMAE embeddings aligned to EEG windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--size", choices=list(HF_MODELS), default=None,
        help="Model size. Omit to run base, large, and huge.",
    )
    ap.add_argument(
        "--eeg-family", required=True, choices=EEG_FAMILIES, metavar="NAME",
        help="EEG family tag: " + ", ".join(EEG_FAMILIES),
    )
    ap.add_argument(
        "--window-seconds", required=True, type=int, metavar="N",
        help="Window length in seconds (5/6/8/10).",
    )
    ap.add_argument(
        "--out-dir", default="embeddings", type=Path, metavar="PATH",
        help="Output directory for .npz files.",
    )
    ap.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="Torch device (e.g. 'cuda', 'cpu'). Defaults to CUDA if available.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    device  = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    sizes = [args.size] if args.size else list(HF_MODELS)
    for size in sizes:
        run(size, args.eeg_family, args.window_seconds, out_dir, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
