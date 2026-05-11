"""
extract_steegformer.py

Extract layerwise ST-EEGFormer embeddings from CineBrain Season 7 EEG data.

Data source  : HuggingFace  Fudan-fMRI/CineBrain  (downloaded and cached automatically)
Season 7     : segments 0–13499 per subject (first 10 episodes, shared by all 6 subjects)
Window       : 6 s = 6000 samples @1000 Hz (= 7.5 × 0.8 s segments, non-integer boundary
               handled in sample space with a rolling buffer)
Preprocessing: resample 1000→128 Hz via mne.filter.resample(), then z-score per channel.
               No additional bandpass — CineBrain is already bandpassed at 0.1–30 Hz.
Output       : embeddings/steegformer_{size}_layerwise.npz
               embeddings    (n_layers, W=1800, S=6, D)  float32
               window_starts (W,)     int64  global sample index of each window start
               subjects      (S,)     str
               senloc        (62,)    int64  ST-EEGFormer channel vocab indices
               channel_names (62,)    str

Usage:
    # Test (50 windows, sub-0001, base model):
    python extract_steegformer.py --steegformer-repo /path/to/STEEGFormer --size base --test

    # Full run, all sizes:
    python extract_steegformer.py --steegformer-repo /path/to/STEEGFormer

Model weights are downloaded automatically from huggingface.co/eugenehp/ST-EEGFormer.

Requirements:
    pip install huggingface_hub safetensors mne torch numpy

Disk space:
    ~12 GB per subject tar (6 subjects = ~72 GB cached in HF_HOME).
    Set HF_HOME env variable to redirect the HuggingFace cache.

HuggingFace token:
    If the dataset is gated, set HF_TOKEN env variable before running.
"""

from __future__ import annotations

import argparse
import gc
import io
import os
import pickle
import sys
import tarfile
import time
from pathlib import Path

import mne
import numpy as np
import torch

# ── Dataset constants ─────────────────────────────────────────────────────────
HF_DATASET_REPO  = "Fudan-fMRI/CineBrain"
HF_WEIGHTS_REPO  = "eugenehp/ST-EEGFormer"
SUBJECTS        = [f"sub-{i:04d}" for i in range(1, 7)]

# Season 7 = first 10 episodes, segments 0..13499 (established by cinebrain_loader.py)
MAX_SEGMENT          = 13500   # exclusive; 13500 × 800 = 10 800 000 samples
SEG_SAMPLES          = 800     # samples per .npy file @ 1000 Hz
SOURCE_FS            = 1000    # CineBrain sampling rate (Hz)
TARGET_FS            = 128     # ST-EEGFormer native rate (Hz)
WINDOW_SOURCE_SAMPLES = 6000   # 6 s @ 1000 Hz
WINDOW_TARGET_SAMPLES = 768    # 6 s @ 128 Hz  (6000 × 128 / 1000 = 768 exact)
N_WINDOWS            = (MAX_SEGMENT * SEG_SAMPLES) // WINDOW_SOURCE_SAMPLES  # 1800

# ── CineBrain channel layout ──────────────────────────────────────────────────
CH_NAMES_EEG = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7",
    "C3","Cz","C4","T8","TP9","CP5","CP1","CP2","CP6","TP10","P7","P3",
    "Pz","P4","P8","PO9","O1","Oz","O2","PO10","AF7","AF3","AF4","AF8",
    "F5","F1","F2","F6","FT9","FT7","FC3","FC4","FT8","FT10","C5","C1",
    "C2","C6","TP7","CP3","CPz","CP4","TP8","P5","P1","P2","P6","PO7",
    "PO3","POz","PO4","PO8",
]
# TP9 (index 16) and TP10 (index 21) are absent from ST-EEGFormer vocabulary
_DROP = {"TP9", "TP10"}
CHANNEL_MASK    = [i for i, ch in enumerate(CH_NAMES_EEG) if ch not in _DROP]  # 62 indices
CHANNEL_NAMES_62 = [CH_NAMES_EEG[i] for i in CHANNEL_MASK]

# ── ST-EEGFormer model registry ───────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "small": {
        "factory":        "vit_small_patch16",
        "n_layers":       8,
        "embed_dim":      512,
        "hf_weights_file": "ST-EEGFormer_small_encoder.safetensors",
    },
    "base": {
        "factory":        "vit_base_patch16",
        "n_layers":       12,
        "embed_dim":      768,
        "hf_weights_file": "ST-EEGFormer_base_encoder.safetensors",
    },
    "large": {
        "factory":        "vit_large_patch16",
        "n_layers":       24,
        "embed_dim":      1024,
        "hf_weights_file": "ST-EEGFormer_large_encoder.safetensors",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Channel vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def build_senloc(steegformer_repo: Path) -> np.ndarray:
    """
    Load the ST-EEGFormer channel vocabulary pickle and return an int64 array
    (62,) mapping each of CineBrain's 62 retained channels to its vocabulary
    integer index expected by the model's spatial positional embedding.
    """
    pkl_path = steegformer_repo / "pretrain" / "senloc_file" / "sen_chan_idx.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Channel vocab pickle not found at {pkl_path}.\n"
            "Run pretrain/senloc_file/channel_position_information.ipynb first, "
            "or copy sen_chan_idx.pkl into that directory."
        )
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    vocab: dict[str, int] = data["channels_mapping"]

    missing = [ch for ch in CHANNEL_NAMES_62 if ch not in vocab]
    if missing:
        raise ValueError(f"Channels not in ST-EEGFormer vocabulary: {missing}")

    senloc = np.array([vocab[ch] for ch in CHANNEL_NAMES_62], dtype=np.int64)
    print(f"  senloc built for {len(senloc)} channels, "
          f"index range [{senloc.min()}, {senloc.max()}]")
    return senloc


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace data access
# ─────────────────────────────────────────────────────────────────────────────

def get_tar_path(subject: str) -> Path:
    """
    Download (and cache) the subject's EEG tar from HuggingFace.
    ~12 GB per subject; cached in HF_HOME after the first download.
    """
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    print(f"  [{subject}] Fetching EEG tar from HuggingFace "
          f"(cached after first run)...")
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=f"{subject}/EEG_preprocessed_data.tar",
        repo_type="dataset",
        token=token,
    )
    return Path(path)


# ─────────────────────────────────────────────────────────────────────────────
# Segment loading and window iteration
# ─────────────────────────────────────────────────────────────────────────────

def _load_segment(tar: tarfile.TarFile, seg_id: int) -> np.ndarray:
    """Load one .npy segment from the open tar. Returns (69, 800) float64."""
    name = f"eeg_02/{seg_id}.npy"
    member = tar.extractfile(name)
    if member is None:
        raise FileNotFoundError(f"Segment {name} not found in tar")
    arr = np.load(io.BytesIO(member.read()))
    if arr.shape != (69, SEG_SAMPLES):
        arr = arr[:69, :SEG_SAMPLES]   # guard against rare (70, 800) anomalies
    return arr


def iter_windows(tar_path: Path, window_indices: list[int] | None = None):
    """
    Yield (window_sample_start, eeg_62ch) for non-overlapping 6 s windows
    covering Season 7 (segments 0–13499).

    Uses a sample-level rolling buffer to handle the non-integer
    segment/window boundary: 7.5 segments = 6000 samples per window.

    Parameters
    ----------
    window_indices : sorted list of global window indices to yield, or None
        for all N_WINDOWS = 1800 windows in order.

    Yields
    ------
    sample_start : int
        Global sample index (at SOURCE_FS = 1000 Hz) of the first sample.
    window : np.ndarray  (62, 6000)  float64
        Raw EEG in Volts for the 62 retained channels.
    """
    selected = set(window_indices) if window_indices is not None else None
    last_idx = max(window_indices) if window_indices is not None else N_WINDOWS - 1

    buffer = np.empty((len(CHANNEL_MASK), 0), dtype=np.float64)
    sample_offset = 0  # sample index of buffer[0]
    w_counter = 0      # global window index

    with tarfile.open(tar_path, "r") as tar:
        for seg_id in range(MAX_SEGMENT):
            seg = _load_segment(tar, seg_id)                      # (69, 800)
            eeg = seg[CHANNEL_MASK, :].astype(np.float64)        # (62, 800)
            buffer = np.concatenate([buffer, eeg], axis=1)

            while buffer.shape[1] >= WINDOW_SOURCE_SAMPLES:
                if selected is None or w_counter in selected:
                    window = buffer[:, :WINDOW_SOURCE_SAMPLES].copy()
                    yield sample_offset, window
                buffer = buffer[:, WINDOW_SOURCE_SAMPLES:]
                sample_offset += WINDOW_SOURCE_SAMPLES
                w_counter += 1
                if w_counter > last_idx:
                    return


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_window(raw: np.ndarray) -> np.ndarray:
    """
    Preprocess a 6 s EEG window to match ST-EEGFormer's training distribution.

    Input  : (62, 6000)  float64  Volts  @ 1000 Hz
    Output : (62,  768)  float32  z-scored @ 128 Hz

    Steps (Appendix E.3 of the ST-EEGFormer paper):
      1. Resample 1000 → 128 Hz via mne.filter.resample()
         (No additional bandpass: CineBrain already bandpassed at 0.1–30 Hz.)
      2. Z-score per channel: zero mean, unit variance.
    """
    # 1. Resample — mne.filter.resample handles anti-aliasing internally
    eeg_128 = mne.filter.resample(
        raw.astype(np.float64),
        up=TARGET_FS,
        down=SOURCE_FS,
        axis=1,
        verbose=False,
    )
    eeg_128 = eeg_128[:, :WINDOW_TARGET_SAMPLES].astype(np.float32)  # (62, 768)

    # 2. Z-score per channel
    mean = eeg_128.mean(axis=1, keepdims=True)   # (62, 1)
    std  = eeg_128.std(axis=1,  keepdims=True)   # (62, 1)
    eeg_128 = (eeg_128 - mean) / (std + 1e-8)

    return eeg_128


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    size: str,
    steegformer_repo: Path,
    device: torch.device,
):
    """
    Instantiate ST-EEGFormer and load encoder weights from HuggingFace
    (eugenehp/ST-EEGFormer, cached after first download).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as safetensors_load

    cfg = MODEL_CONFIGS[size]

    # Make the STEEGFormer source importable
    easy_start = steegformer_repo / "easy_start"
    if str(easy_start) not in sys.path:
        sys.path.insert(0, str(easy_start))

    from models_vit_eeg import (  # type: ignore
        vit_small_patch16, vit_base_patch16, vit_large_patch16,
    )
    _factory = {
        "vit_small_patch16": vit_small_patch16,
        "vit_base_patch16":  vit_base_patch16,
        "vit_large_patch16": vit_large_patch16,
    }
    model = _factory[cfg["factory"]]()

    # Download encoder weights from HF (cached after first run)
    print(f"  Fetching {cfg['hf_weights_file']} from HuggingFace ...")
    weights_path = hf_hub_download(
        repo_id=HF_WEIGHTS_REPO,
        filename=cfg["hf_weights_file"],
    )

    state_dict = safetensors_load(weights_path, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Weights loaded  ({len(missing)} missing keys, "
          f"{len(unexpected)} unexpected keys)")

    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Layerwise forward pass
# ─────────────────────────────────────────────────────────────────────────────

def embed_window_layerwise(
    model,
    window: np.ndarray,
    senloc: np.ndarray,
    n_layers: int,
    embed_dim: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run one preprocessed window through ST-EEGFormer and capture the CLS token
    output after each transformer block.

    Input
    -----
    window : (62, 768) float32, z-scored
    senloc : (62,)     int64,   channel vocabulary indices

    Output
    ------
    (n_layers, embed_dim)  float32 — CLS token embedding per block
    """
    captured: list[torch.Tensor] = []
    hooks = []

    def _hook(module, inp, out):
        # out: (B, Seq_total + 1, D)  — CLS token is at position 0
        captured.append(out[:, 0, :].detach().cpu())  # (1, D)

    for blk in model.blocks:
        hooks.append(blk.register_forward_hook(_hook))

    eeg_t    = torch.from_numpy(window).unsqueeze(0).to(device)           # (1, 62, 768)
    senloc_t = torch.from_numpy(senloc).unsqueeze(0).long().to(device)    # (1, 62)

    with torch.no_grad():
        # Call forward_features directly to skip the classification head and
        # avoid the @autocast(device_type='cuda') decorator on forward().
        model.forward_features(eeg_t, senloc_t)

    for h in hooks:
        h.remove()

    result = np.zeros((n_layers, embed_dim), dtype=np.float32)
    for layer_idx, cls_out in enumerate(captured):
        result[layer_idx] = cls_out.squeeze(0).numpy()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    size: str,
    steegformer_repo: Path,
    out_dir: Path,
    device: torch.device,
    test: bool = False,
) -> None:
    cfg       = MODEL_CONFIGS[size]
    n_layers  = cfg["n_layers"]
    embed_dim = cfg["embed_dim"]

    suffix   = "_test" if test else "_layerwise"
    out_path = out_dir / f"steegformer_{size}{suffix}.npz"
    if out_path.exists():
        print(f"\n  {out_path.name} already exists — skipping.")
        return

    # In test mode: 50 random windows, 1 subject (sub-0001)
    subjects_to_run = [SUBJECTS[0]] if test else SUBJECTS
    n_win           = 50           if test else N_WINDOWS
    test_indices    = sorted(
        np.random.default_rng(0).choice(N_WINDOWS, n_win, replace=False).tolist()
    ) if test else None

    print(f"\n{'='*64}")
    print(f"  ST-EEGFormer-{size}  |  layers={n_layers}  D={embed_dim}  "
          f"W={n_win}  S={len(subjects_to_run)}"
          + ("  [TEST MODE]" if test else ""))
    print(f"{'='*64}")

    senloc = build_senloc(steegformer_repo)

    t0    = time.time()
    model = load_model(size, steegformer_repo, device)
    print(f"  Model ready in {time.time()-t0:.1f} s")

    # Pre-allocate (n_layers, W, S, D)
    embeddings    = np.zeros(
        (n_layers, n_win, len(subjects_to_run), embed_dim), dtype=np.float32
    )
    window_starts = np.zeros(n_win, dtype=np.int64)
    starts_saved  = False

    for s_idx, subject in enumerate(subjects_to_run):
        print(f"\n  [{subject}]")
        tar_path = get_tar_path(subject)

        t_sub = time.time()
        for w_idx, (sample_start, raw_window) in enumerate(iter_windows(tar_path, test_indices)):
            window    = preprocess_window(raw_window)               # (62, 768)
            layer_emb = embed_window_layerwise(
                model, window, senloc, n_layers, embed_dim, device
            )                                                        # (n_layers, D)
            embeddings[:, w_idx, s_idx, :] = layer_emb

            if not starts_saved:
                window_starts[w_idx] = sample_start

            if (w_idx + 1) % 200 == 0:
                elapsed = time.time() - t_sub
                rate    = (w_idx + 1) / elapsed
                eta     = (n_win - w_idx - 1) / rate
                print(f"    {w_idx+1}/{n_win} windows  "
                      f"{rate:.1f} win/s  ETA {eta/60:.1f} min")


        starts_saved = True
        print(f"  [{subject}] done in {(time.time()-t_sub):.1f} s")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=embeddings,
        window_starts=window_starts,
        subjects=np.array(subjects_to_run),
        size=size,
        n_layers=n_layers,
        senloc=senloc,
        channel_names=np.array(CHANNEL_NAMES_62),
    )
    print(f"\n  Saved {embeddings.shape} → {out_path}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract ST-EEGFormer layerwise embeddings from CineBrain Season 7.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--steegformer-repo", required=True, metavar="PATH",
        help="Path to local clone of github.com/LiuyinYang1101/STEEGFormer",
    )
    ap.add_argument(
        "--size", choices=list(MODEL_CONFIGS), default=None,
        help="Model size to run. Omit to run all three sequentially.",
    )
    ap.add_argument(
        "--out-dir", default="embeddings", metavar="PATH",
        help="Output directory for .npz files.",
    )
    ap.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="Torch device (e.g. 'cuda', 'cuda:1', 'cpu'). "
             "Defaults to CUDA if available.",
    )
    ap.add_argument(
        "--test", action="store_true",
        help="Test mode: run 50 windows on sub-0001 only. "
             "Saves steegformer_{size}_test.npz with shape (n_layers, 50, 1, D).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    steegformer_repo = Path(args.steegformer_repo).expanduser().resolve()
    out_dir          = Path(args.out_dir).expanduser().resolve()

    if not steegformer_repo.exists():
        raise FileNotFoundError(f"STEEGFormer repo not found: {steegformer_repo}")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")
    print(f"Output : {out_dir}")
    print(f"Windows: {N_WINDOWS} per subject × {len(SUBJECTS)} subjects × 6 s each")

    if args.test:
        print("TEST MODE — 50 windows, sub-0001 only, all requested sizes.")

    sizes = [args.size] if args.size else list(MODEL_CONFIGS)
    for size in sizes:
        run(size, steegformer_repo, out_dir, device, test=args.test)

    print("\nDone.")


if __name__ == "__main__":
    main()
