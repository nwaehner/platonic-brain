"""
extract_neurolm.py

Extract layerwise NeuroLM embeddings from CineBrain Season 7 EEG data.

Data source  : HuggingFace  Fudan-fMRI/CineBrain  (downloaded and cached automatically)
Model weights: HuggingFace  Weibang/NeuroLM        (downloaded and cached automatically)
Season 7     : segments 0–13499 per subject (first 10 episodes, shared by all 6 subjects)
Window       : 8 s = 10 × 0.8 s segments = 8000 samples @ 1000 Hz
               → resample to 1600 samples @ 200 Hz → 8 patches of 200 samples per channel
Tokenisation : 64 channels × 8 patches = 512 valid tokens; zero-padded to BLOCK_SIZE=1024
Normalisation: resample_poly() then ×1e4  (Volts → µV ÷ 100, NeuroLM training range)
Output       : embeddings/neurolm_{b|l|xl}_layerwise.npz
               embeddings    (n_layers, W=1350, S=6, D)  float32
               window_starts (W,)     int64  starting segment index for each window
               subjects      (S,)     str
               size          str
               n_layers      int

Architecture (Table 8, NeuroLM paper):
    NeuroLM-B  : 12 GPT-2 blocks, D=768,  254M params
    NeuroLM-L  : 24 GPT-2 blocks, D=1024, 500M params
    NeuroLM-XL : 48 GPT-2 blocks, D=1600, 1696M params

Usage:
    # Test (50 random windows, sub-0001, base model):
    python extract_neurolm.py --neurolm-repo /path/to/NeuroLM --size base --test

    # Full run, all three sizes sequentially:
    python extract_neurolm.py --neurolm-repo /path/to/NeuroLM

    # Single size, explicit output dir and device:
    python extract_neurolm.py --neurolm-repo /path/to/NeuroLM --size large \\
        --out-dir /data/embeddings --device cuda:0

Requirements:
    pip install huggingface_hub scipy torch numpy

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
HF_WEIGHTS_REPO       = "Weibang/NeuroLM"
SUBJECTS              = [f"sub-{i:04d}" for i in range(1, 7)]

MAX_SEGMENT           = 13500   # exclusive; Season 7 = segments 0–13499
SEG_SAMPLES           = 800     # samples per .npy segment @ 1000 Hz
SOURCE_FS             = 1000    # CineBrain sampling rate (Hz)
TARGET_FS             = 200     # NeuroLM sampling rate (Hz)
SEGS_PER_WINDOW       = 10      # 10 × 0.8 s = 8 s window
WINDOW_SAMPLES_RAW    = SEGS_PER_WINDOW * SEG_SAMPLES               # 8000 @ 1000 Hz
WINDOW_SAMPLES_TARGET = WINDOW_SAMPLES_RAW * TARGET_FS // SOURCE_FS # 1600 @ 200 Hz
PATCH_SIZE            = 200     # 1 s patch @ 200 Hz  (paper: P = 200)
N_PATCHES_PER_CH      = WINDOW_SAMPLES_TARGET // PATCH_SIZE          # 8
N_EEG                 = 64      # EEG channels (first 64 of 69 stored in the raw tars)
BLOCK_SIZE            = 1024    # NeuroLM / GPT-2 context size
UNIT_SCALE            = 1e4     # Volts → µV ÷ 100  (paper: "values divided by 100")
N_WINDOWS             = MAX_SEGMENT // SEGS_PER_WINDOW               # 1350

# ── NeuroLM model registry ────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "base":  {"lm_ckpt": "checkpoints/NeuroLM-B.pt",  "out_stem": "neurolm_b"},
    "large": {"lm_ckpt": "checkpoints/NeuroLM-L.pt",  "out_stem": "neurolm_l"},
    "xl":    {"lm_ckpt": "checkpoints/NeuroLM-XL.pt", "out_stem": "neurolm_xl"},
}
VQ_CHECKPOINT = "checkpoints/VQ.pt"

# ── Channel mapping (inlined from channel_map.py for self-containment) ────────
# NeuroLM standard_1020 channel list (from NeuroLM/dataset.py)
_NEUROLM_STANDARD_1020 = [
    'FP1','FPZ','FP2',
    'AF9','AF7','AF5','AF3','AF1','AFZ','AF2','AF4','AF6','AF8','AF10',
    'F9','F7','F5','F3','F1','FZ','F2','F4','F6','F8','F10',
    'FT9','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','FT10',
    'T9','T7','C5','C3','C1','CZ','C2','C4','C6','T8','T10',
    'TP9','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','TP10',
    'P9','P7','P5','P3','P1','PZ','P2','P4','P6','P8','P10',
    'PO9','PO7','PO5','PO3','PO1','POZ','PO2','PO4','PO6','PO8','PO10',
    'O1','OZ','O2','O9','CB1','CB2',
    'IZ','O10','T3','T5','T4','T6','M1','M2','A1','A2',
    'CFC1','CFC2','CFC3','CFC4','CFC5','CFC6','CFC7','CFC8',
    'CCP1','CCP2','CCP3','CCP4','CCP5','CCP6','CCP7','CCP8',
    'T1','T2','FTT9h','TTP7h','TPP9h','FTT10h','TPP8h','TPP10h',
    'FP1-F7','F7-T7','T7-P7','P7-O1','FP2-F8','F8-T8','T8-P8','P8-O2',
    'FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2',
    'pad','I1','I2',
]
PAD_CHAN_IDX = _NEUROLM_STANDARD_1020.index('pad')

# GSN-HydroCel-64 (E1..E64) → nearest NeuroLM 10-20 name (built via MNE montage matching)
_GSN64_TO_1020 = {
    'E1':'AF8', 'E2':'AF4', 'E3':'F2',  'E4':'FCZ', 'E5':'FP2', 'E6':'AFZ',
    'E7':'FC1', 'E8':'AFZ', 'E9':'F1',  'E10':'AF3','E11':'AF3','E12':'F3',
    'E13':'F3', 'E14':'F3', 'E15':'FC3','E16':'FC1','E17':'AF7','E18':'F7',
    'E19':'FC5','E20':'C3', 'E21':'C1', 'E22':'C3', 'E23':'FT9','E24':'T7',
    'E25':'C5', 'E26':'CP5','E27':'CP5','E28':'CP3','E29':'TP9','E30':'P7',
    'E31':'P3', 'E32':'P9', 'E33':'P1', 'E34':'CPZ','E35':'O1', 'E36':'POZ',
    'E37':'OZ', 'E38':'P2', 'E39':'O2', 'E40':'P4', 'E41':'C2', 'E42':'CP4',
    'E43':'P10','E44':'P8', 'E45':'CP6','E46':'CP6','E47':'TP10','E48':'C6',
    'E49':'C4', 'E50':'C4', 'E51':'FC2','E52':'T8', 'E53':'FC4','E54':'FC2',
    'E55':'FT10','E56':'FC6','E57':'F4','E58':'F8', 'E59':'F4', 'E60':'F4',
    'E61':'F10','E62':'AF10','E63':'AF9','E64':'F9',
}
# CineBrain raw axis-0 indices 0..63 correspond to channels E1..E64
_CINEBRAIN_TO_NEUROLM_IDX: list[int] = [
    _NEUROLM_STANDARD_1020.index(_GSN64_TO_1020[f'E{i}'])
    for i in range(1, 65)
]


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


def _download_checkpoint(filename: str) -> Path:
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
    Yield (start_segment, raw_eeg) for non-overlapping 8 s windows covering
    Season 7 (segments 0–13499).

    Since SEGS_PER_WINDOW=10 divides MAX_SEGMENT=13500 evenly, windows are
    strictly segment-aligned: window i starts at segment i*10.

    Parameters
    ----------
    window_indices : sorted list of window indices in [0, N_WINDOWS) to yield,
                     or None for all 1350 in order.

    Yields
    ------
    start_segment : int
        Index of the first of the 10 segments in this window.
    raw_eeg : np.ndarray  (69, 8000)  float64
        Raw concatenated EEG at 1000 Hz (all 69 stored channels).
    """
    selected = set(window_indices) if window_indices is not None else None
    last_idx = max(window_indices) if window_indices is not None else N_WINDOWS - 1

    with tarfile.open(tar_path, "r") as tar:
        for w_idx in range(N_WINDOWS):
            if w_idx > last_idx:
                break
            if selected is not None and w_idx not in selected:
                continue
            start_seg = w_idx * SEGS_PER_WINDOW
            pieces = [_load_segment(tar, seg_id)
                      for seg_id in range(start_seg, start_seg + SEGS_PER_WINDOW)]
            raw = np.concatenate(pieces, axis=1)   # (69, 8000) @ 1000 Hz
            yield start_seg, raw


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_window(raw: np.ndarray) -> np.ndarray:
    """
    Preprocess one raw 8 s EEG window to NeuroLM input format.

    Input  : (69, 8000) float64  Volts  @ 1000 Hz
    Output : (64, 1600) float32  normalised  @ 200 Hz

    Steps (consistent with NeuroLM paper, Section 3.2 Data Preprocessing):
      1. Drop channels 64–68 → (64, 8000)
      2. Resample 1000 → 200 Hz via scipy resample_poly() (anti-aliasing built in)
      3. Scale ×1e4  (Volts → µV ÷ 100, matching ±100 µV training range)
    """
    eeg    = raw[:N_EEG, :].astype(np.float64)
    eeg_ds = resample_poly(eeg, up=TARGET_FS, down=SOURCE_FS, axis=1)
    eeg_ds = eeg_ds[:, :WINDOW_SAMPLES_TARGET].astype(np.float32)   # (64, 1600)
    eeg_ds *= UNIT_SCALE
    return eeg_ds


# ─────────────────────────────────────────────────────────────────────────────
# Input tensor construction
# ─────────────────────────────────────────────────────────────────────────────

def build_input_tensors(
    window: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a (64, 1600) window into the four NeuroLM input tensors.

    Token layout — time-major, channels within each time step:
      tokens   0– 63  = all 64 channels at patch 0
      tokens  64–127  = all 64 channels at patch 1
      ...
      tokens 448–511  = all 64 channels at patch 7   (512 valid tokens)
      tokens 512–1023 = zero padding                 (512 pad tokens)

    Returns
    -------
    X           : (1, 1024, 200)  float32  patch values
    input_chans : (1, 1024)       int32    NeuroLM standard_1020 channel index
    input_time  : (1, 1024)       int32    time-step index (0–7 for valid, 0 for pad)
    input_mask  : (1, 1024)       float32  1=valid token, 0=padding
    """
    n_ch     = window.shape[0]   # 64
    patches  = window.reshape(n_ch, N_PATCHES_PER_CH, PATCH_SIZE)
    patches  = np.transpose(patches, (1, 0, 2)).reshape(-1, PATCH_SIZE)  # (512, 200)
    n_tokens = patches.shape[0]  # 512

    X = np.zeros((1, BLOCK_SIZE, PATCH_SIZE), dtype=np.float32)
    X[0, :n_tokens] = patches

    chan_idx    = np.array(_CINEBRAIN_TO_NEUROLM_IDX, dtype=np.int32)   # (64,)
    input_chans = np.full((1, BLOCK_SIZE), PAD_CHAN_IDX, dtype=np.int32)
    input_chans[0, :n_tokens] = np.tile(chan_idx, N_PATCHES_PER_CH)     # (512,)

    input_time = np.zeros((1, BLOCK_SIZE), dtype=np.int32)
    input_time[0, :n_tokens] = np.repeat(np.arange(N_PATCHES_PER_CH), n_ch)  # (512,)

    input_mask = np.zeros((1, BLOCK_SIZE), dtype=np.float32)
    input_mask[0, :n_tokens] = 1.0

    return (
        torch.from_numpy(X),
        torch.from_numpy(input_chans),
        torch.from_numpy(input_time),
        torch.from_numpy(input_mask),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(size: str, neurolm_repo: Path, device: torch.device):
    """
    Instantiate NeuroLM, load checkpoint from HuggingFace, return
    (model, n_layers, embed_dim).
    """
    if str(neurolm_repo) not in sys.path:
        sys.path.insert(0, str(neurolm_repo))

    from model.model_neurolm import NeuroLM, GPTConfig  # type: ignore

    cfg     = MODEL_CONFIGS[size]
    lm_ckpt = _download_checkpoint(cfg["lm_ckpt"])
    vq_ckpt = _download_checkpoint(VQ_CHECKPOINT)

    print("  Loading checkpoint...")
    ckpt    = torch.load(lm_ckpt, map_location="cpu", weights_only=False)
    gptconf = GPTConfig(**ckpt["model_args"])

    # NeuroLM.__init__ calls torch.load(vq_ckpt) without map_location;
    # patch temporarily so CPU-only servers don't fail on GPU-saved tensors.
    _orig = torch.load
    def _cpu_load(*a, **kw):
        kw.setdefault("map_location", "cpu")
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _cpu_load
    try:
        model = NeuroLM(gptconf, tokenizer_ckpt_path=str(vq_ckpt), init_from="scratch")
    finally:
        torch.load = _orig

    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")

    model.to(device).eval()
    n_layers  = len(model.GPT2.transformer.h)
    embed_dim = gptconf.n_embd
    print(f"  GPT-2 blocks: {n_layers}  embed_dim: {embed_dim}")
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
    Run one preprocessed window through NeuroLM and capture the mean-pooled
    hidden state after each GPT-2 transformer block.

    Input  : (64, 1600) float32
    Output : (n_layers, embed_dim) float32

    Mean-pooling is over valid (non-padded) token positions only, consistent
    with the original embed_window() in extract_embeddings.py.
    """
    X, input_chans, input_time, input_mask = build_input_tensors(window)
    X           = X.to(device)
    input_chans = input_chans.to(device)
    input_time  = input_time.to(device)
    input_mask  = input_mask.to(device)
    mask_4d     = input_mask.unsqueeze(1).repeat(1, BLOCK_SIZE, 1).unsqueeze(1)

    captured: list[torch.Tensor] = []
    hooks = []
    for block in model.GPT2.transformer.h:
        hooks.append(block.register_forward_hook(
            lambda m, inp, out, _c=captured: _c.append(out)
        ))

    with torch.no_grad():
        x_eeg = model.tokenizer(X, input_chans, input_time, mask_4d,
                                 return_all_tokens=True)
        x_eeg = model.encode_transform_layer(x_eeg)
        x_eeg = x_eeg + model.pos_embed(input_chans)
        model.GPT2(x_eeg, None, None, None, input_time, None, None, lm_head=False)

    for h in hooks:
        h.remove()

    mask_w  = input_mask.unsqueeze(-1)            # (1, 1024, 1)
    n_valid = mask_w.sum(dim=1).clamp(min=1.0)    # (1, 1)

    result = np.zeros((n_layers, embed_dim), dtype=np.float32)
    for layer_idx, hidden in enumerate(captured):
        pooled = (hidden * mask_w).sum(dim=1) / n_valid   # (1, D)
        result[layer_idx] = pooled.squeeze(0).cpu().numpy()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    size: str,
    neurolm_repo: Path,
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
    n_win           = 50 if test else N_WINDOWS
    test_indices    = sorted(
        np.random.default_rng(0).choice(N_WINDOWS, n_win, replace=False).tolist()
    ) if test else None

    print(f"\n{'='*64}")
    print(f"  NeuroLM-{size}  |  W={n_win}  S={len(subjects_to_run)}"
          + ("  [TEST MODE]" if test else ""))
    print(f"{'='*64}")

    t0 = time.time()
    model, n_layers, embed_dim = load_model(size, neurolm_repo, device)
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
        for w_idx, (start_seg, raw) in enumerate(iter_windows(tar_path, test_indices)):
            window    = preprocess_window(raw)
            layer_emb = embed_window_layerwise(
                model, window, n_layers, embed_dim, device
            )
            embeddings[:, w_idx, s_idx, :] = layer_emb

            if not starts_saved:
                window_starts[w_idx] = start_seg

            if (w_idx + 1) % 100 == 0:
                elapsed = time.time() - t_sub
                rate    = (w_idx + 1) / elapsed
                eta     = (n_win - w_idx - 1) / rate
                print(f"    {w_idx+1}/{n_win} windows  "
                      f"{rate:.2f} win/s  ETA {eta/60:.1f} min")

        starts_saved = True
        print(f"  [{subject}] done in {(time.time()-t_sub)/60:.1f} min")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=embeddings,
        window_starts=window_starts,
        subjects=np.array(subjects_to_run),
        size=size,
        n_layers=n_layers,
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
        description="Extract NeuroLM layerwise embeddings from CineBrain Season 7.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--neurolm-repo", required=True, metavar="PATH",
        help="Path to local clone of github.com/935963004/NeuroLM "
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
             "Saves neurolm_{b|l|xl}_test.npz with shape (n_layers, 50, 1, D).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    neurolm_repo = Path(args.neurolm_repo).expanduser().resolve()
    out_dir      = Path(args.out_dir).expanduser().resolve()

    if not neurolm_repo.exists():
        raise FileNotFoundError(f"NeuroLM repo not found: {neurolm_repo}")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")
    print(f"Output : {out_dir}")
    print(f"Windows: {N_WINDOWS} per subject × {len(SUBJECTS)} subjects × 8 s each")

    if args.test:
        print("TEST MODE — 50 random windows, sub-0001 only, all requested sizes.")

    sizes = [args.size] if args.size else list(MODEL_CONFIGS)
    for size in sizes:
        run(size, neurolm_repo, out_dir, device, test=args.test)

    print("\nDone.")


if __name__ == "__main__":
    main()
