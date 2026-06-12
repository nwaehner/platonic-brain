"""
extract_dinov2.py

Extract layerwise DINOv2 embeddings for video windows aligned to EEG windows
of any length. For each EEG window of D seconds the script:
  1. Loads the corresponding video range from the CineBrain `videos.tar`
     (8100 clips, each clip = 4 s),
  2. Concatenates the relevant frames (whole and/or partial clips),
  3. Uniformly samples **D frames (1 frame per second)**,
  4. Runs each frame through DINOv2 and captures per-block CLS tokens via
     forward hooks,
  5. For every transformer block: takes the CLS token (sequence position 0),
     then averages it across the frames of the window → one (D_emb,) vector
     per layer per window.

The CLS token is used (not mean-pooled patch tokens) to match the protocol of
Huh et al. 2024 (platonic-rep), which pools ViT features as ``v[:, 0, :]`` per
layer; the Aristotelian paper follows that same protocol.

Model and preprocessing match the Aristotelian paper (Attias et al. 2024)
exactly:
  - timm models: vit_{small,base,large,giant}_patch14_dinov2.lvd142m
  - loaded with timm.create_model(name, pretrained=True, num_classes=0)
  - preprocessing: timm.data.resolve_data_config + create_transform (eval mode)
  - native DINOv2 resolution: 518×518 (patch_size=14 → 37×37 patch grid,
    no positional embedding interpolation)
  - layerwise extraction via forward hooks (not get_intermediate_layers, which
    has ordering issues in timm ≤1.0.26)

So the output is layerwise — shape (n_layers, W, D_emb) — mirroring the EEG
extractors' (n_layers, W, S, D), so cross-modal alignment can be measured
between every pair of layers (DINOv2 layer i ↔ EEG-model layer j), as done in
the Platonic / Aristotelian representation-alignment papers, rather than only
against a single (last) layer.

n_layers = len(model.blocks) (all transformer block outputs; no embedding-layer
output is included — matching the EEG extractors which capture post-block
hidden states only).

DINOv2 is a purely spatial image model; no temporal modeling is performed.
Cross-modal alignment with EEG/video models reflects static scene content only.

Time alignment per EEG model (D = window_seconds → frames per window):
    FEMBA, LUNA       :  5 s →  1.25 clips →  5 frames
    ST-EEGFormer      :  6 s →  1.5  clips →  6 frames
    NeuroLM           :  8 s →  2    clips →  8 frames
    REVE              : 10 s →  2.5  clips → 10 frames

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
    The output is tagged with `--eeg-family`. FEMBA and LUNA both use the
    same 5 s window tiling, so they share one DINOv2 NPZ tagged 'femba_luna'.
    Choices:
        femba_luna   — 5 s windows  (covers both FEMBA and LUNA)
        steegformer  — 6 s windows
        neurolm      — 8 s windows
        reve         — 10 s windows
    One DINOv2 NPZ per (DINOv2 size, family) pair.

Output: embeddings/dinov2_{size}__{eeg_family}.npz
    embeddings      (n_layers, W, D_emb)  float32
    window_starts_s (W,)        float64   canonical start time in seconds
    window_seconds  scalar      int
    size            scalar      str       DINOv2 size (small/base/large/giant)
    n_layers        scalar      int       number of transformer blocks
    n_frames        scalar      int       frames per window (= window_seconds)
    eeg_family      scalar      str       e.g. 'neurolm', 'reve', 'femba'

Usage:
    # 5 s FEMBA + LUNA shared (all four sizes)
    python extract_dinov2.py --eeg-family femba_luna --window-seconds 5
    # → dinov2_{small,base,large,giant}__femba_luna.npz

    # 6 s ST-EEGFormer
    python extract_dinov2.py --eeg-family steegformer --window-seconds 6

    # 8 s NeuroLM
    python extract_dinov2.py --eeg-family neurolm --window-seconds 8

    # 10 s REVE, single size
    python extract_dinov2.py --size giant --eeg-family reve --window-seconds 10

Disk space:
    ~2.6 GB cached tar + ~2.6 GB extracted clips ≈ 5.2 GB on first run.

Requirements:
    pip install timm torch numpy opencv-python pillow huggingface_hub
"""

from __future__ import annotations

import argparse
import gc
import os
import ssl
import tarfile
import time
from functools import lru_cache
from pathlib import Path

# Windows SSL fix — must run before any network import
ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests as _requests
_orig_request = _requests.Session.request
def _no_verify(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig_request(self, method, url, **kwargs)
_requests.Session.request = _no_verify

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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

# timm model names matching the Aristotelian paper (Attias et al. 2024)
TIMM_MODELS = {
    "small": "vit_small_patch14_dinov2.lvd142m",
    "base":  "vit_base_patch14_dinov2.lvd142m",
    "large": "vit_large_patch14_dinov2.lvd142m",
    "giant": "vit_giant_patch14_dinov2.lvd142m",
}

# Native DINOv2 training resolution: patch_size=14, 37×37 patch grid → 518 px.
# Using native resolution avoids positional embedding interpolation, matching
# the Aristotelian and Platonic-rep paper protocols.
FRAME_SIZE = 518


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
    """Map clip index → on-disk path (built once on first call)."""
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
    (FRAME_SIZE, FRAME_SIZE, 3) uint8 RGB arrays.
    lru_cache reuses results for adjacent windows that share clips.
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
    Caps at MAX_CLIP_IDX (Season 7 boundary).
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


def _sample_uniform(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Uniformly sample n frames across the full available range (1 fps for DINOv2)."""
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 forward pass  (per-frame CLS via hooks, then mean across frames)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_frames(
    transform,
    model: torch.nn.Module,
    device: torch.device,
    frames: list[np.ndarray],
) -> np.ndarray:
    """
    Batch all frames through DINOv2 and capture per-block CLS tokens via
    forward hooks. For each transformer block, average the CLS token over the
    window's frames. Returns (n_layers, D); n_layers = len(model.blocks).

    Forward hooks are used instead of get_intermediate_layers() because timm
    <=1.0.26's list-argument form has unexpected output ordering.

    Uses the CLS token to match Huh et al. 2024 (platonic-rep) and the
    Aristotelian paper (Attias et al. 2024), which pool ViT features as
    ``v[:, 0, :]`` per layer.
    """
    # frames are (FRAME_SIZE, FRAME_SIZE, 3) uint8 RGB numpy arrays
    imgs = torch.stack([transform(Image.fromarray(f)) for f in frames]).to(device)

    hook_cls: list[torch.Tensor] = []
    hooks = [
        blk.register_forward_hook(
            lambda m, inp, out: hook_cls.append(out[:, 0, :].detach())
        )
        for blk in model.blocks
    ]
    with torch.no_grad():
        model(imgs)
    for h in hooks:
        h.remove()

    # hook_cls: list of (n_frames, D) tensors, one per block in order
    # mean over frames per block → (D,); stack blocks → (n_layers, D)
    pooled = [cls.mean(dim=0) for cls in hook_cls]
    return torch.stack(pooled, dim=0).cpu().float().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

EEG_FAMILIES = ("femba_luna", "steegformer", "neurolm", "reve", "clip4s")

def run(
    size: str,
    eeg_family: str,
    window_seconds: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    out_path = out_dir / f"dinov2_{size}__{eeg_family}.npz"

    if out_path.exists():
        print(f"\n  {out_path.name} already exists — skipping.")
        return

    n_frames = window_seconds   # 1 frame per second
    starts_s = _canonical_starts_s(window_seconds)
    W        = len(starts_s)
    last_end = float(starts_s[-1]) + window_seconds

    model_name = TIMM_MODELS[size]
    print(f"\n{'='*72}")
    print(f"  DINOv2-{size}  |  window={window_seconds} s "
          f"({window_seconds / CLIP_SECONDS:.2f} clips, {n_frames} frames @ 1 fps)")
    print(f"  timm model : {model_name}")
    print(f"  resolution : {FRAME_SIZE}x{FRAME_SIZE} (native, no pos-embed interpolation)")
    print(f"  EEG family : {eeg_family}")
    print(f"  output     : {out_path.name}")
    print(f"  windows={W}  device={device}")
    print(f"  time range: [{float(starts_s[0]):.2f}, {last_end:.2f}] s "
          f"(within Season 7: clips 0..{MAX_CLIP_IDX})")
    print(f"  embedding[i] will correspond 1-to-1 with EEG window i.")
    print(f"{'='*72}")

    print(f"  Loading {model_name} ...")
    t0        = time.time()
    model     = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    data_cfg  = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**data_cfg, is_training=False)
    n_layers  = len(model.blocks)
    D         = model.embed_dim
    print(f"  Loaded in {time.time() - t0:.1f} s  |  n_layers={n_layers}  D={D}")
    print(f"  timm data config: input_size={data_cfg['input_size']}  "
          f"mean={data_cfg['mean']}  std={data_cfg['std']}")

    embeddings = np.zeros((n_layers, W, D), dtype=np.float32)
    t_loop     = time.time()

    for w_idx, start_sec in enumerate(starts_s):
        frames = _frames_for_window(float(start_sec), float(window_seconds))
        if len(frames) == 0:
            raise RuntimeError(f"No frames for window {w_idx} at t={start_sec:.2f}s")
        sampled               = _sample_uniform(frames, n_frames)
        embeddings[:, w_idx]  = _embed_frames(transform, model, device, sampled)

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
        n_frames        = n_frames,
        eeg_family      = eeg_family,
        timm_model      = model_name,
        frame_size      = FRAME_SIZE,
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
        description="Extract DINOv2 embeddings aligned to EEG windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--size", choices=list(TIMM_MODELS), default=None,
        help="Model size. Omit to run all four (small, base, large, giant).",
    )
    ap.add_argument(
        "--eeg-family", required=True, choices=EEG_FAMILIES, metavar="NAME",
        help="EEG model family this output is paired with (used for the "
             "filename tag and metadata). Choices: " + ", ".join(EEG_FAMILIES),
    )
    ap.add_argument(
        "--window-seconds", required=True, type=int, metavar="N",
        help="EEG window length in seconds (5 for femba_luna, 6 for "
             "steegformer, 8 for neurolm, 10 for reve). Frames per window "
             "= N (1 fps).",
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

    sizes = [args.size] if args.size else list(TIMM_MODELS)
    for size in sizes:
        run(size, args.eeg_family, args.window_seconds, out_dir, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
