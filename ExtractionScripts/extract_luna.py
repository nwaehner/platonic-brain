"""
extract_luna.py

Extract layerwise LUNA embeddings from CineBrain Season 7 EEG data.

Data source  : HuggingFace  Fudan-fMRI/CineBrain  (downloaded and cached automatically)
Model weights: HuggingFace  PulpBio/LUNA           (downloaded and cached automatically)
Season 7     : segments 0–13499 per subject (first 10 episodes, shared by all 6 subjects)
Window       : 5 s = 5000 samples @ 1000 Hz → resample to 1280 samples @ 256 Hz
               → S = 1280 / 40 = 32 patches (exact LUNA pre-training length, no truncation)
Channels     : 22 bipolar pairs (full TUEG double-banana montage).
               TP9 ≈ A1 and TP10 ≈ A2 (left/right mastoid-adjacent electrodes on the
               GSN-HydroCel-64 cap); used for the A1-T3 and T4-A2 ear-referenced pairs.
               Window iteration uses a rolling sample buffer because 5 s / 0.8 s = 6.25
               segments per window (non-integer boundary).
Normalisation: z-score per channel per window (matching LUNA pre-training pipeline).
3-D positions: standard_1020 midpoints for each bipolar pair (from MNE); LUNA applies
               per-sample min-max normalisation internally inside prepare_tokens().
Output       : embeddings/luna_{base|large|huge}_layerwise.npz
               embeddings        (n_layers, W=2160, S=6, D)  float32
               window_starts     (W,)  int64  global sample index @ 1000 Hz
               subjects          (S,)  str
               channel_names     (22,) str    bipolar pair labels e.g. "Fp1-F7"
               channel_locations (22, 3) float32  raw 3-D midpoints in metres
               size              str
               n_layers          int

Architecture (config/model/LUNA_*.yaml + models/LUNA.py in pulp-bio/BioFoundation):
    LUNA-Base : depth= 8, embed_dim= 64, num_queries=4, D=Q×E=256,   7M params
    LUNA-Large: depth=10, embed_dim= 96, num_queries=6, D=Q×E=576,  43M params
    LUNA-Huge : depth=24, embed_dim=128, num_queries=8, D=Q×E=1024, 311M params

Layerwise extraction: forward hook on each RotaryTransformerBlock in model.blocks.
    Each block output shape: (B, S=32, D) → mean-pool over S → (D,) per layer.

Usage:
    # Test (50 random windows, sub-0001, base model):
    python extract_luna.py --luna-repo /path/to/BioFoundation --size base --test

    # Full run, all three sizes sequentially:
    python extract_luna.py --luna-repo /path/to/BioFoundation

    # Single size, explicit output dir and device:
    python extract_luna.py --luna-repo /path/to/BioFoundation --size large \\
        --out-dir /data/embeddings --device cuda:0

Requirements:
    pip install huggingface_hub safetensors mne scipy torch numpy

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
import sys
import tarfile
import time
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import resample_poly

# ── Dataset constants ─────────────────────────────────────────────────────────
HF_DATASET_REPO       = "Fudan-fMRI/CineBrain"
HF_WEIGHTS_REPO       = "PulpBio/LUNA"
SUBJECTS              = [f"sub-{i:04d}" for i in range(1, 7)]

MAX_SEGMENT           = 13500   # exclusive; Season 7 = segments 0–13499
SEG_SAMPLES           = 800     # samples per .npy segment @ 1000 Hz
SOURCE_FS             = 1000    # CineBrain sampling rate (Hz)
TARGET_FS             = 256     # LUNA pre-training rate (Hz)
PATCH_SIZE            = 40      # LUNA patch size in timesteps (model config)
WINDOW_SOURCE_SAMPLES = 5000    # 5 s @ 1000 Hz
WINDOW_TARGET_SAMPLES = 1280    # 5 s @ 256 Hz (5000 × 256 / 1000 = 1280 exact)
N_PATCHES             = WINDOW_TARGET_SAMPLES // PATCH_SIZE   # 32 (exact, no truncation)
N_BIPOLAR             = 22      # bipolar channels (full TUEG double-banana)
N_WINDOWS             = (MAX_SEGMENT * SEG_SAMPLES) // WINDOW_SOURCE_SAMPLES  # 2160

# ── CineBrain 64-channel layout (axis-0 indices 0..63 → standard 10-20 names) ─
# Identical ordering to extract_steegformer.py
_CH_NAMES_EEG = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7",
    "C3","Cz","C4","T8","TP9","CP5","CP1","CP2","CP6","TP10","P7","P3",
    "Pz","P4","P8","PO9","O1","Oz","O2","PO10","AF7","AF3","AF4","AF8",
    "F5","F1","F2","F6","FT9","FT7","FC3","FC4","FT8","FT10","C5","C1",
    "C2","C6","TP7","CP3","CPz","CP4","TP8","P5","P1","P2","P6","PO7",
    "PO3","POz","PO4","PO8",
]

# ── Bipolar channel pairs (TUEG double-banana, all 22) ───────────────────────
# Each entry is (anode_name, cathode_name); differential = raw[a] - raw[b].
# TP9 and TP10 are the mastoid-adjacent electrodes on the GSN-HydroCel-64 cap
# and serve as A1 and A2 for the two ear-referenced pairs.
_BIPOLAR_PAIRS: list[tuple[str, str]] = [
    # Left temporal chain
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    # Right temporal chain
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    # Transverse chain
    ("T7", "C3"), ("C3", "Cz"), ("Cz", "C4"), ("C4", "T8"),
    # Left parasagittal chain
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    # Right parasagittal chain
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    # Ear-referenced chains (TP9 ≈ A1, TP10 ≈ A2)
    ("TP9", "T7"), ("T8", "TP10"),
]

BIPOLAR_NAMES: list[str] = [f"{a}-{b}" for a, b in _BIPOLAR_PAIRS]

_NAME_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(_CH_NAMES_EEG)}
BIPOLAR_INDICES: list[tuple[int, int]] = [
    (_NAME_TO_IDX[a], _NAME_TO_IDX[b]) for a, b in _BIPOLAR_PAIRS
]

# ── LUNA model registry ───────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "base": {
        "depth": 8, "embed_dim": 64, "num_queries": 4, "num_heads": 2,
        "hf_file": "LUNA_base.safetensors", "out_stem": "luna_base",
    },
    "large": {
        "depth": 10, "embed_dim": 96, "num_queries": 6, "num_heads": 2,
        "hf_file": "LUNA_large.safetensors", "out_stem": "luna_large",
    },
    "huge": {
        "depth": 24, "embed_dim": 128, "num_queries": 8, "num_heads": 2,
        "hf_file": "LUNA_huge.safetensors", "out_stem": "luna_huge",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 3-D electrode coordinates
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_locations() -> np.ndarray:
    """
    Return (22, 3) float32 — MNE standard_1005 midpoint positions (metres)
    for each bipolar pair.  standard_1005 is used instead of standard_1020
    because it includes TP9 and TP10 (absent from the classic 19-electrode set).
    LUNA normalises channel_locations internally via per-sample min-max inside
    prepare_tokens(), so raw metre values are fine.
    """
    montage  = mne.channels.make_standard_montage("standard_1005")
    pos: dict[str, np.ndarray] = montage.get_positions()["ch_pos"]

    locs = np.zeros((N_BIPOLAR, 3), dtype=np.float32)
    for i, (a, b) in enumerate(_BIPOLAR_PAIRS):
        locs[i] = (pos[a] + pos[b]) / 2.0
    return locs


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace access
# ─────────────────────────────────────────────────────────────────────────────

def get_tar_path(subject: str) -> Path:
    """Download (and cache) the subject's EEG tar from HuggingFace. ~12 GB each."""
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


def _download_weights(filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    print(f"  Fetching {filename} from HuggingFace (cached after first run)...")
    return Path(hf_hub_download(repo_id=HF_WEIGHTS_REPO, filename=filename))


# ─────────────────────────────────────────────────────────────────────────────
# Segment loading and window iteration
# ─────────────────────────────────────────────────────────────────────────────

def _load_segment(tar: tarfile.TarFile, seg_id: int) -> np.ndarray:
    """Load one .npy segment from the open tar. Returns (69, 800) float64."""
    name   = f"eeg_02/{seg_id}.npy"
    member = tar.extractfile(name)
    if member is None:
        raise FileNotFoundError(f"Segment {name} not found in tar")
    arr = np.load(io.BytesIO(member.read()))
    if arr.shape != (69, SEG_SAMPLES):
        arr = arr[:69, :SEG_SAMPLES]
    return arr


def iter_windows(tar_path: Path, window_indices: list[int] | None = None):
    """
    Yield (sample_start, eeg_64ch) for non-overlapping 5 s windows covering
    Season 7 (segments 0–13499).

    Uses a sample-level rolling buffer because 5 s / 0.8 s = 6.25 segments
    per window (non-integer boundary).

    Parameters
    ----------
    window_indices : sorted list of global window indices to yield, or None
        for all N_WINDOWS = 2160 windows in order.

    Yields
    ------
    sample_start : int
        Global sample index at SOURCE_FS = 1000 Hz of the first sample.
    window : np.ndarray  (64, 5000)  float64
        Raw EEG in Volts for the 64 EEG channels.
    """
    selected  = set(window_indices) if window_indices is not None else None
    last_idx  = max(window_indices) if window_indices is not None else N_WINDOWS - 1

    buffer        = np.empty((64, 0), dtype=np.float64)
    sample_offset = 0
    w_counter     = 0

    with tarfile.open(tar_path, "r") as tar:
        for seg_id in range(MAX_SEGMENT):
            seg    = _load_segment(tar, seg_id)
            buffer = np.concatenate(
                [buffer, seg[:64, :].astype(np.float64)], axis=1
            )

            while buffer.shape[1] >= WINDOW_SOURCE_SAMPLES:
                if selected is None or w_counter in selected:
                    yield sample_offset, buffer[:, :WINDOW_SOURCE_SAMPLES].copy()
                buffer        = buffer[:, WINDOW_SOURCE_SAMPLES:]
                sample_offset += WINDOW_SOURCE_SAMPLES
                w_counter     += 1
                if w_counter > last_idx:
                    return


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_window(raw: np.ndarray) -> np.ndarray:
    """
    Preprocess one raw 5 s EEG window to LUNA input format.

    Input  : (64, 5000) float64  Volts  @ 1000 Hz
    Output : (22, 1280) float32  z-scored @ 256 Hz

    Steps (consistent with LUNA/TUEG pre-training pipeline):
      1. Compute 22 bipolar differentials: raw[a] - raw[b]
      2. Resample 1000 → 256 Hz via scipy resample_poly()
      3. Z-score per channel: zero mean, unit variance
    No additional bandpass — CineBrain is already filtered at 0.1–30 Hz + 50 Hz
    notch, which is a subset of LUNA's training range (0.1–75 Hz).
    """
    bipolar = np.stack(
        [raw[a] - raw[b] for a, b in BIPOLAR_INDICES], axis=0,
    ).astype(np.float64)                                           # (22, 5000)

    bipolar = resample_poly(bipolar, up=TARGET_FS, down=SOURCE_FS, axis=1)
    bipolar = bipolar[:, :WINDOW_TARGET_SAMPLES].astype(np.float32)  # (22, 1280)

    mean    = bipolar.mean(axis=1, keepdims=True)
    std     = bipolar.std(axis=1,  keepdims=True)
    bipolar = (bipolar - mean) / (std + 1e-8)

    return bipolar


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(size: str, luna_repo: Path, device: torch.device):
    """
    Instantiate LUNA and load pre-trained weights from HuggingFace.
    Returns (model, n_layers, embed_dim).

    num_classes=2 selects the classification forward path (no reconstruction
    decoder), giving a clean forward pass for embedding extraction.  The
    encoder block weights load correctly from the pre-training checkpoint via
    strict=False; the unused classification head is randomly initialised and
    its output is discarded.
    """
    from safetensors.torch import load_file as safetensors_load

    if str(luna_repo) not in sys.path:
        sys.path.insert(0, str(luna_repo))

    from models.LUNA import LUNA  # type: ignore

    cfg          = MODEL_CONFIGS[size]
    weights_path = _download_weights(cfg["hf_file"])

    model = LUNA(
        patch_size  = PATCH_SIZE,
        embed_dim   = cfg["embed_dim"],
        depth       = cfg["depth"],
        num_heads   = cfg["num_heads"],
        num_queries = cfg["num_queries"],
        num_classes = 2,
        drop_path   = 0.0,
    )

    print("  Loading weights...")
    state_dict = safetensors_load(weights_path, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")

    model.to(device).eval()
    n_layers  = cfg["depth"]
    embed_dim = cfg["embed_dim"] * cfg["num_queries"]   # D = Q × E
    print(f"  Blocks: {n_layers}   D (Q×E): {embed_dim}")
    return model, n_layers, embed_dim


# ─────────────────────────────────────────────────────────────────────────────
# Layerwise forward pass
# ─────────────────────────────────────────────────────────────────────────────

def embed_window_layerwise(
    model,
    window: np.ndarray,
    channel_locs: torch.Tensor,
    n_layers: int,
    embed_dim: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run one preprocessed window through LUNA and capture the mean-pooled
    output after each RotaryTransformerBlock in model.blocks.

    Input
    -----
    window       : (22, 1280) float32, z-scored bipolar EEG
    channel_locs : (1, 22, 3) float32 — raw 3-D midpoints; normalised inside LUNA

    Output
    ------
    (n_layers, embed_dim)  float32 — mean over S=32 patches per block
    """
    captured: list[torch.Tensor] = []
    hooks = []

    def _hook(module, inp, out):
        # out: (B, S, Q*E) = (1, 32, D) — mean-pool over patch dimension
        captured.append(out.mean(dim=1).detach().cpu())   # (1, D)

    for blk in model.blocks:
        hooks.append(blk.register_forward_hook(_hook))

    x   = torch.from_numpy(window).unsqueeze(0).to(device)   # (1, 22, 1280)
    loc = channel_locs.to(device)                             # (1, 22, 3)

    with torch.no_grad():
        model(x, None, loc)

    for h in hooks:
        h.remove()

    result = np.zeros((n_layers, embed_dim), dtype=np.float32)
    for layer_idx, pooled in enumerate(captured):
        result[layer_idx] = pooled.squeeze(0).numpy()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    size: str,
    luna_repo: Path,
    out_dir: Path,
    device: torch.device,
    channel_locs: np.ndarray,
    test: bool = False,
) -> None:
    cfg      = MODEL_CONFIGS[size]
    suffix   = "_test" if test else "_layerwise"
    out_path = out_dir / f"{cfg['out_stem']}{suffix}.npz"

    if out_path.exists():
        print(f"\n  {out_path.name} already exists — skipping.")
        return

    subjects_to_run = [SUBJECTS[0]] if test else SUBJECTS
    n_win           = 50           if test else N_WINDOWS
    test_indices    = sorted(
        np.random.default_rng(0).choice(N_WINDOWS, n_win, replace=False).tolist()
    ) if test else None

    print(f"\n{'='*64}")
    print(f"  LUNA-{size}  |  blocks={cfg['depth']}  D={cfg['embed_dim']*cfg['num_queries']}"
          f"  W={n_win}  S={len(subjects_to_run)}"
          + ("  [TEST MODE]" if test else ""))
    print(f"{'='*64}")

    t0 = time.time()
    model, n_layers, embed_dim = load_model(size, luna_repo, device)
    print(f"  Model ready in {time.time()-t0:.1f} s")

    locs_t = torch.from_numpy(channel_locs).unsqueeze(0)   # (1, 22, 3), reused every call

    embeddings    = np.zeros(
        (n_layers, n_win, len(subjects_to_run), embed_dim), dtype=np.float32
    )
    window_starts = np.zeros(n_win, dtype=np.int64)
    starts_saved  = False

    for s_idx, subject in enumerate(subjects_to_run):
        print(f"\n  [{subject}]")
        tar_path = get_tar_path(subject)

        t_sub = time.time()
        for w_idx, (sample_start, raw_window) in enumerate(
            iter_windows(tar_path, test_indices)
        ):
            window    = preprocess_window(raw_window)
            layer_emb = embed_window_layerwise(
                model, window, locs_t, n_layers, embed_dim, device
            )
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
        print(f"  [{subject}] done in {(time.time()-t_sub)/60:.1f} min")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings        = embeddings,
        window_starts     = window_starts,
        subjects          = np.array(subjects_to_run),
        channel_names     = np.array(BIPOLAR_NAMES),
        channel_locations = channel_locs,
        size              = size,
        n_layers          = n_layers,
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
        description="Extract LUNA layerwise embeddings from CineBrain Season 7.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--luna-repo", required=True, metavar="PATH",
        help="Path to local clone of github.com/pulp-bio/BioFoundation "
             "(needed for model source code; weights are downloaded from HF).",
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
        help="Test mode: 50 random windows on sub-0001 only. "
             "Saves luna_{base|large|huge}_test.npz with shape (n_layers, 50, 1, D).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    luna_repo = Path(args.luna_repo).expanduser().resolve()
    out_dir   = Path(args.out_dir).expanduser().resolve()

    if not luna_repo.exists():
        raise FileNotFoundError(f"BioFoundation repo not found: {luna_repo}")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device   : {device}")
    print(f"Output   : {out_dir}")
    print(f"Windows  : {N_WINDOWS} per subject × {len(SUBJECTS)} subjects × 5 s each")
    print(f"Channels : {N_BIPOLAR} bipolar pairs (full TUEG double-banana, TP9≈A1, TP10≈A2)")
    print(f"Patches  : {N_PATCHES} per window (P={PATCH_SIZE} @ {TARGET_FS} Hz)")

    if args.test:
        print("TEST MODE — 50 random windows, sub-0001 only, all requested sizes.")

    print("\nBuilding bipolar channel 3-D coordinates from MNE standard_1020...")
    channel_locs = build_channel_locations()
    print(f"  channel_locations: {channel_locs.shape}  "
          f"range [{channel_locs.min():.4f}, {channel_locs.max():.4f}] m")

    sizes = [args.size] if args.size else list(MODEL_CONFIGS)
    for size in sizes:
        run(size, luna_repo, out_dir, device, channel_locs, test=args.test)

    print("\nDone.")


if __name__ == "__main__":
    main()
