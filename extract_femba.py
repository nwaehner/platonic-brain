"""
extract_femba.py

Extract layerwise FEMBA embeddings from CineBrain Season 7 EEG data.

Data source  : HuggingFace  Fudan-fMRI/CineBrain        (downloaded and cached automatically)
Model weights: HuggingFace  PulpBio/FEMBA  TUSL folder  (downloaded and cached automatically)
Season 7     : segments 0–13499 per subject (first 10 episodes, shared by all 6 subjects)
Window       : 5 s = 5000 samples @ 1000 Hz → resample to 1280 samples @ 256 Hz
               → 80 time tokens (2D patch (2,16), stride (2,16), grid = 11×80)
Channels     : 22 bipolar pairs (full TUEG double-banana montage).
               TP9 ≈ A1 and TP10 ≈ A2 (mastoid-adjacent electrodes on the GSN-HydroCel-64
               cap); used for the ear-referenced pairs A1-T3 and T4-A2.
               Rolling sample buffer used because 5 s / 0.8 s = 6.25 segments per window.
Normalisation: IQR per channel + clip [−20, 20] (matching FEMBA pre-training pipeline).
Output       : embeddings/femba_{tiny|base|large}_tusl_layerwise.npz
               embeddings    (n_layers, W=2160, S=6, D)  float32
               window_starts (W,)  int64  global sample index @ 1000 Hz
               subjects      (S,)  str
               channel_names (22,) str    bipolar pair labels e.g. "Fp1-F7"
               size          str
               dataset       str   ("TUSL")
               n_layers      int

Architecture (docs/model/FEMBA.md + models/FEMBA.py in pulp-bio/BioFoundation):
    2D patch embedding: patch=(2,16), stride=(2,16) → grid=(11 ch-rows, 80 time-cols)
    Token sequence  : 80 time tokens of dim D = 11 × embed_dim
    FEMBA-Tiny : num_blocks= 2, embed_dim=35, D=385,   7.8M params
    FEMBA-Base : num_blocks=12, embed_dim=35, D=385,  47.7M params
    FEMBA-Large: num_blocks= 4, embed_dim=79, D=869,  77.8M params

Layerwise extraction: forward hook on each LayerNorm in model.norm_layers (post-residual
    normalised output after each BiMamba block).
    Each norm_layer output shape: (B, 80, D) → mean-pool over 80 tokens → (D,) per layer.

Weights: TUSL fine-tuned (slowing event classification) from PulpBio/FEMBA on HuggingFace.
    All three size variants share the same pretrained encoder (single pretraining run on
    TUEG filtered to remove TUAB ∪ TUAR ∪ TUSL subjects). Encoder weights load via
    strict=False; the fine-tuned MambaClassifier head is ignored.

Usage:
    # Test (50 random windows, sub-0001, tiny model):
    python extract_femba.py --femba-repo /path/to/BioFoundation --size tiny --test

    # Full run, all three sizes sequentially:
    python extract_femba.py --femba-repo /path/to/BioFoundation

    # Single size, explicit output dir and device:
    python extract_femba.py --femba-repo /path/to/BioFoundation --size large \\
        --out-dir /data/embeddings --device cuda:0

Requirements:
    pip install huggingface_hub safetensors mamba-ssm scipy torch numpy

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

import numpy as np
import torch
from scipy.signal import resample_poly

# ── Dataset constants ─────────────────────────────────────────────────────────
HF_DATASET_REPO       = "Fudan-fMRI/CineBrain"
HF_WEIGHTS_REPO       = "PulpBio/FEMBA"
HF_WEIGHTS_DATASET    = "TUSL"
SUBJECTS              = [f"sub-{i:04d}" for i in range(1, 7)]

MAX_SEGMENT           = 13500   # exclusive; Season 7 = segments 0–13499
SEG_SAMPLES           = 800     # samples per .npy segment @ 1000 Hz
SOURCE_FS             = 1000    # CineBrain sampling rate (Hz)
TARGET_FS             = 256     # FEMBA pre-training rate (Hz)
PATCH_ROWS            = 2       # channel dimension of each 2D patch
PATCH_COLS            = 16      # time dimension of each 2D patch
GRID_ROWS             = 11      # (22 - 2) // 2 + 1  — channel-axis grid size
GRID_COLS             = 80      # (1280 - 16) // 16 + 1  — time-axis grid size
WINDOW_SOURCE_SAMPLES = 5000    # 5 s @ 1000 Hz
WINDOW_TARGET_SAMPLES = 1280    # 5 s @ 256 Hz  (5000 × 256 / 1000 = 1280 exact)
N_BIPOLAR             = 22      # bipolar channels (full TUEG double-banana)
N_WINDOWS             = (MAX_SEGMENT * SEG_SAMPLES) // WINDOW_SOURCE_SAMPLES  # 2160

# ── CineBrain 64-channel layout (axis-0 indices 0..63 → standard 10-20 names) ─
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
# Ordering matches make_tueg_bipolar.py standard_order exactly.
# TP9 and TP10 are the mastoid-adjacent electrodes on the GSN-HydroCel-64 cap
# and serve as A1 and A2 for the ear-referenced pairs.
_BIPOLAR_PAIRS: list[tuple[str, str]] = [
    # Left temporal chain
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    # Right temporal chain
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    # Transverse chain
    ("T7", "C3"),  ("C3", "Cz"), ("Cz", "C4"), ("C4", "T8"),
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

# ── FEMBA model registry ──────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "tiny": {
        "num_blocks": 2,  "embed_dim": 35,
        "hf_file":   "TUSL/FEMBA_tiny.safetensors",
        "out_stem":  "femba_tiny_tusl",
    },
    "base": {
        "num_blocks": 12, "embed_dim": 35,
        "hf_file":   "TUSL/FEMBA_base.safetensors",
        "out_stem":  "femba_base_tusl",
    },
    "large": {
        "num_blocks": 4,  "embed_dim": 79,
        "hf_file":   "TUSL/FEMBA_large.safetensors",
        "out_stem":  "femba_large_tusl",
    },
}


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
    Preprocess one raw 5 s EEG window to FEMBA input format.

    Input  : (64, 5000) float64  Volts  @ 1000 Hz
    Output : (22, 1280) float32  IQR-normalised @ 256 Hz

    Steps (matching FEMBA/TUEG pre-training pipeline):
      1. Compute 22 bipolar differentials: raw[a] - raw[b]
      2. Resample 1000 → 256 Hz via scipy resample_poly()
      3. IQR normalise per channel: (x − p25) / (p75 − p25 + 1e-8)
      4. Clip to [−20, 20]
    No additional bandpass — CineBrain is already filtered at 0.1–30 Hz + 50 Hz
    notch, which is a subset of FEMBA's training range (0.1–75 Hz).
    """
    bipolar = np.stack(
        [raw[a] - raw[b] for a, b in BIPOLAR_INDICES], axis=0,
    ).astype(np.float64)                                            # (22, 5000)

    bipolar = resample_poly(bipolar, up=TARGET_FS, down=SOURCE_FS, axis=1)
    bipolar = bipolar[:, :WINDOW_TARGET_SAMPLES]                    # (22, 1280)

    p25  = np.percentile(bipolar, 25, axis=1, keepdims=True)
    p75  = np.percentile(bipolar, 75, axis=1, keepdims=True)
    bipolar = (bipolar - p25) / (p75 - p25 + 1e-8)
    bipolar = np.clip(bipolar, -20.0, 20.0).astype(np.float32)

    return bipolar


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(size: str, femba_repo: Path, device: torch.device):
    """
    Instantiate FEMBA and load TUSL fine-tuned weights from HuggingFace.
    Returns (model, n_layers, embed_dim).

    num_classes=0 selects reconstruction mode (no classification head), giving
    a clean forward pass for embedding extraction. The encoder weights
    (patch_embed, mamba_blocks, norm_layers, pos_embed) load correctly via
    strict=False; the fine-tuned MambaClassifier keys are ignored.
    """
    from safetensors.torch import load_file as safetensors_load

    if str(femba_repo) not in sys.path:
        sys.path.insert(0, str(femba_repo))

    from models.FEMBA import FEMBA  # type: ignore

    cfg          = MODEL_CONFIGS[size]
    weights_path = _download_weights(cfg["hf_file"])

    model = FEMBA(
        seq_length   = WINDOW_TARGET_SAMPLES,
        num_channels = N_BIPOLAR,
        embed_dim    = cfg["embed_dim"],
        num_blocks   = cfg["num_blocks"],
        patch_size   = (PATCH_ROWS, PATCH_COLS),
        stride       = (PATCH_ROWS, PATCH_COLS),
        num_classes  = 0,
    )

    print("  Loading weights...")
    state_dict = safetensors_load(weights_path, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")

    model.to(device).eval()
    n_layers  = cfg["num_blocks"]
    embed_dim = GRID_ROWS * cfg["embed_dim"]   # D = 11 × embed_dim
    print(f"  Blocks: {n_layers}   D (11×embed_dim): {embed_dim}")
    return model, n_layers, embed_dim


# ─────────────────────────────────────────────────────────────────────────────
# Layerwise forward pass
# ─────────────────────────────────────────────────────────────────────────────

def embed_window_layerwise(
    model,
    window: np.ndarray,
    n_layers: int,
    embed_dim: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run one preprocessed window through FEMBA and capture the mean-pooled
    output after each LayerNorm in model.norm_layers.

    Input
    -----
    window : (22, 1280) float32, IQR-normalised bipolar EEG

    Output
    ------
    (n_layers, embed_dim)  float32 — mean over 80 time tokens per norm layer
    """
    captured: list[torch.Tensor] = []
    hooks = []

    def _hook(module, inp, out):
        # out: (B=1, 80, D=11*embed_dim) — mean-pool over time tokens
        captured.append(out.mean(dim=1).detach().cpu())   # (1, D)

    for norm_layer in model.norm_layers:
        hooks.append(norm_layer.register_forward_hook(_hook))

    x    = torch.from_numpy(window).unsqueeze(0).to(device)        # (1, 22, 1280)
    mask = torch.zeros(
        1, N_BIPOLAR, WINDOW_TARGET_SAMPLES, dtype=torch.bool, device=device
    )                                                               # no masking

    with torch.no_grad():
        model(x, mask)   # return value discarded; hooks capture layer outputs

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
    femba_repo: Path,
    out_dir: Path,
    device: torch.device,
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
    print(f"  FEMBA-{size} (TUSL)  |  blocks={cfg['num_blocks']}  "
          f"D={GRID_ROWS * cfg['embed_dim']}  W={n_win}  S={len(subjects_to_run)}"
          + ("  [TEST MODE]" if test else ""))
    print(f"{'='*64}")

    t0 = time.time()
    model, n_layers, embed_dim = load_model(size, femba_repo, device)
    print(f"  Model ready in {time.time()-t0:.1f} s")

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
                model, window, n_layers, embed_dim, device
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
        embeddings    = embeddings,
        window_starts = window_starts,
        subjects      = np.array(subjects_to_run),
        channel_names = np.array(BIPOLAR_NAMES),
        size          = size,
        dataset       = HF_WEIGHTS_DATASET,
        n_layers      = n_layers,
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
        description="Extract FEMBA layerwise embeddings from CineBrain Season 7.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--femba-repo", required=True, metavar="PATH",
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
             "Saves femba_{size}_tusl_test.npz with shape (n_layers, 50, 1, D).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    femba_repo = Path(args.femba_repo).expanduser().resolve()
    out_dir    = Path(args.out_dir).expanduser().resolve()

    if not femba_repo.exists():
        raise FileNotFoundError(f"BioFoundation repo not found: {femba_repo}")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device   : {device}")
    print(f"Output   : {out_dir}")
    print(f"Windows  : {N_WINDOWS} per subject × {len(SUBJECTS)} subjects × 5 s each")
    print(f"Channels : {N_BIPOLAR} bipolar pairs (TUEG double-banana, TP9≈A1, TP10≈A2)")
    print(f"Tokens   : {GRID_COLS} time tokens per window "
          f"(patch ({PATCH_ROWS},{PATCH_COLS}) @ {TARGET_FS} Hz)")
    print(f"Weights  : {HF_WEIGHTS_REPO} / {HF_WEIGHTS_DATASET}")

    if args.test:
        print("TEST MODE — 50 random windows, sub-0001 only, all requested sizes.")

    sizes = [args.size] if args.size else list(MODEL_CONFIGS)
    for size in sizes:
        run(size, femba_repo, out_dir, device, test=args.test)

    print("\nDone.")


if __name__ == "__main__":
    main()
