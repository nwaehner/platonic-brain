"""
extract_reve.py

Extract layerwise REVE embeddings from CineBrain Season 7 EEG data.

Data source  : HuggingFace  Fudan-fMRI/CineBrain          (downloaded and cached automatically)
Model weights: HuggingFace  brain-bzh/reve-{base,large}   (gated — accept Responsible Use
               Agreement at https://huggingface.co/brain-bzh/reve-base before first run)
               HuggingFace  brain-bzh/reve-positions       (not gated)
Season 7     : segments 0–13499 per subject (first 10 episodes, shared by all 6 subjects)
Window       : 10 s = 2.5 clips = 12.5 × 0.8 s segments = 10000 samples @ 1000 Hz
               → resample to 2000 samples @ 200 Hz → 11 patches (w=200, o=20) — exact fit,
               zero remainder. Formula: (2000 − 200) / 180 = 10 exactly.
Channels     : 64 EEG channels (GSN-HydroCel-64 10-20 layout). All 64 names confirmed
               present in brain-bzh/reve-positions.
Normalisation: (1) resample 1000 → 200 Hz; (2) bandpass 0.5–99 Hz (sosfiltfilt, 4th order
               Butterworth; capped at 99 Hz to avoid near-Nyquist instability; paper uses
               99.5 Hz — inert here since CineBrain is already low-passed at 30 Hz);
               (3) z-score per channel across the full session (13500 × 800 = 10,800,000
               samples), clip ±15 σ. Matches REVE pretraining pipeline (§3.1.1).
               Requires a streaming first pass over the tar to compute session statistics
               before window extraction (two passes per subject).
Output       : embeddings/reve_{base|large}_layerwise.npz
               embeddings    (n_layers=22, W=1080, S=6, D)  float32
               window_starts (W,)  int64  global sample index @ 1000 Hz
               subjects      (S,)  str
               channel_names (64,) str    10-20 electrode names
               size          str
               n_layers      int

Architecture (REVE paper, NeurIPS 2025 — El Ouahidi et al.):
    REVE-Base : 22 transformer layers, D=512,  69M params, n_freq=4
    REVE-Large: 22 transformer layers, D=1250, 408M params, n_freq=5
    Patch: w=200 samples (1 s), overlap=20 samples (0.1 s), stride=180.
    4D Fourier positional encoding: (x, y, z) from electrode positions + t from patch index.
    Forward hook captures the output of each transformer block; mean-pooled over all
    channel × patch tokens (64 × 11 = 704) to produce one D-dim vector per layer per window.

Usage:
    # Test (50 random windows, sub-0001, base model):
    python extract_reve.py --size base --test

    # Full run, both sizes sequentially:
    python extract_reve.py

    # Single size, explicit output dir and device:
    python extract_reve.py --size large --out-dir /data/embeddings --device cuda:0

Requirements:
    pip install huggingface_hub transformers scipy torch numpy

Disk space:
    ~12 GB per subject tar (6 subjects ≈ 72 GB cached in HF_HOME).
    Set HF_HOME env variable to redirect the HuggingFace cache.

HuggingFace token:
    If the dataset or model is gated, set HF_TOKEN env variable before running.
"""

from __future__ import annotations

import argparse
import gc
import io
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

# ── Dataset constants ─────────────────────────────────────────────────────────
HF_DATASET_REPO    = "Fudan-fMRI/CineBrain"
HF_REVE_POSITIONS  = "brain-bzh/reve-positions"
SUBJECTS           = [f"sub-{i:04d}" for i in range(1, 7)]

MAX_SEGMENT           = 13500     # Season 7 segments per subject (exclusive)
SEG_SAMPLES           = 800       # samples per segment @ 1000 Hz
SOURCE_FS             = 1000      # CineBrain sampling rate (Hz)
TARGET_FS             = 200       # REVE required sampling rate (Hz)
WINDOW_SOURCE_SAMPLES = 10_000    # 10 s @ 1000 Hz
WINDOW_TARGET_SAMPLES = 2_000     # 10 s @ 200 Hz
N_EEG                 = 64        # EEG channels (first 64 of 69 stored in the raw tars)
N_WINDOWS             = (MAX_SEGMENT * SEG_SAMPLES) // WINDOW_SOURCE_SAMPLES  # 1080

# ── REVE patch constants ──────────────────────────────────────────────────────
PATCH_SIZE    = 200                              # 1 s @ 200 Hz
PATCH_OVERLAP = 20                               # 0.1 s @ 200 Hz
PATCH_STRIDE  = PATCH_SIZE - PATCH_OVERLAP       # 180 samples
N_PATCHES     = (WINDOW_TARGET_SAMPLES - PATCH_SIZE) // PATCH_STRIDE + 1  # 11 (exact)
N_TOKENS      = N_EEG * N_PATCHES                # 64 × 11 = 704 tokens per window

# ── CineBrain channel names (GSN-HydroCel-64, axis-0 indices 0..63) ──────────
# Ordered to match the raw EEG array stored in the CineBrain tar files.
# All 64 names are present in brain-bzh/reve-positions (case-sensitive, mixed case).
CH_NAMES_EEG: list[str] = [
    "Fp1", "Fp2", "F7",  "F3",  "Fz",  "F4",  "F8",  "FC5",
    "FC1", "FC2", "FC6", "T7",  "C3",  "Cz",  "C4",  "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7",  "P3",
    "Pz",  "P4",  "P8",  "PO9", "O1",  "Oz",  "O2",  "PO10",
    "AF7", "AF3", "AF4", "AF8", "F5",  "F1",  "F2",  "F6",
    "FT9", "FT7", "FC3", "FC4", "FT8", "FT10", "C5",  "C1",
    "C2",  "C6",  "TP7", "CP3", "CPz", "CP4", "TP8", "P5",
    "P1",  "P2",  "P6",  "PO7", "PO3", "POz", "PO4", "PO8",
]

# ── REVE model registry ───────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "base":  {
        "hf_repo":  "brain-bzh/reve-base",
        "n_layers": 22,
        "embed_dim": 512,
        "out_stem": "reve_base",
    },
    "large": {
        "hf_repo":  "brain-bzh/reve-large",
        "n_layers": 22,
        "embed_dim": 1250,
        "out_stem": "reve_large",
    },
}

# ── Bandpass filter (built once at import time, reused across all windows) ────
# 0.5–99 Hz at 200 Hz (paper uses 99.5 Hz; capped at 99 to avoid near-Nyquist
# instability — inert for CineBrain which is already low-passed at 30 Hz).
_SOS_BP = butter(4, [0.5, 99.0], btype="bandpass", fs=TARGET_FS, output="sos")


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace access
# ─────────────────────────────────────────────────────────────────────────────

# Token file search order (first match wins):
#   1. HF_TOKEN environment variable
#   2. HF_TOKEN_FILE environment variable (explicit path to .txt)
#   3. hf_token.txt next to this script
#   4. hf_token.txt in the user's home directory
_TOKEN_SEARCH_PATHS: list[Path] = [
    Path(__file__).parent / "hf_token.txt",
    Path.home() / "hf_token.txt",
]

def _get_hf_token() -> str | None:
    """
    Resolve HuggingFace token from env var or a token file.
    Returns the token string, or None if not found (huggingface_hub will then
    fall back to its own cached login if available).
    """
    if token := os.environ.get("HF_TOKEN"):
        return token.strip()
    if path_str := os.environ.get("HF_TOKEN_FILE"):
        p = Path(path_str)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        raise FileNotFoundError(f"HF_TOKEN_FILE set but file not found: {p}")
    for p in _TOKEN_SEARCH_PATHS:
        if p.exists():
            print(f"  [token] Reading HF token from {p}")
            return p.read_text(encoding="utf-8").strip()
    return None


def get_tar_path(subject: str) -> Path:
    """Download (and cache) the subject's EEG tar from HuggingFace. ~12 GB each."""
    from huggingface_hub import hf_hub_download
    print(f"  [{subject}] Fetching EEG tar from HuggingFace "
          f"(cached after first run)...")
    return Path(hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=f"{subject}/EEG_preprocessed_data.tar",
        repo_type="dataset",
        token=_get_hf_token(),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Segment loading
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


# ─────────────────────────────────────────────────────────────────────────────
# Window iteration  (rolling buffer — 10 s = 12.5 segments, non-integer boundary)
# ─────────────────────────────────────────────────────────────────────────────

def iter_windows(tar_path: Path, window_indices: list[int] | None = None):
    """
    Yield (sample_start, eeg_64ch) for non-overlapping 10 s windows covering
    Season 7 (segments 0–13499).

    Uses a sample-level rolling buffer because 10 s / 0.8 s = 12.5 segments per
    window (non-integer boundary). Each yield covers exactly 10000 @ 1000 Hz samples.
    1080 windows exhaust the 10,800,000-sample season exactly.

    Parameters
    ----------
    window_indices : sorted list of global window indices to yield, or None for all.

    Yields
    ------
    sample_start : int
        Global sample index @ 1000 Hz of the first sample in the window.
    window : np.ndarray  (64, 10000)  float64
        Raw 64-channel EEG at 1000 Hz.
    """
    selected  = set(window_indices) if window_indices is not None else None
    last_idx  = max(window_indices) if window_indices is not None else N_WINDOWS - 1

    buffer        = np.empty((N_EEG, 0), dtype=np.float64)
    sample_offset = 0
    w_counter     = 0

    with tarfile.open(tar_path, "r") as tar:
        for seg_id in range(MAX_SEGMENT):
            seg    = _load_segment(tar, seg_id)[:N_EEG, :].astype(np.float64)
            buffer = np.concatenate([buffer, seg], axis=1)

            while buffer.shape[1] >= WINDOW_SOURCE_SAMPLES:
                if selected is None or w_counter in selected:
                    yield sample_offset, buffer[:, :WINDOW_SOURCE_SAMPLES].copy()
                buffer        = buffer[:, WINDOW_SOURCE_SAMPLES:]
                sample_offset += WINDOW_SOURCE_SAMPLES
                w_counter     += 1
                if w_counter > last_idx:
                    return


# ─────────────────────────────────────────────────────────────────────────────
# Session statistics  (streaming pass — required for z-score across session)
# ─────────────────────────────────────────────────────────────────────────────

def compute_session_stats(tar_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std across the full Season 7 session
    (13500 segments × 800 samples = 10,800,000 samples per channel).

    REVE pretraining used z-score statistics computed across recording sessions,
    not per-window (§3.1.1). This requires a dedicated streaming pass over the tar
    before window extraction — one extra full read per subject.

    Returns
    -------
    mean : (64,) float32
    std  : (64,) float32   channels with std < 1e-10 are set to 1.0 (flat channel guard)
    """
    n         = MAX_SEGMENT * SEG_SAMPLES           # 10,800,000
    total_sum = np.zeros(N_EEG, dtype=np.float64)
    total_sq  = np.zeros(N_EEG, dtype=np.float64)

    with tarfile.open(tar_path, "r") as tar:
        for seg_id in range(MAX_SEGMENT):
            seg = _load_segment(tar, seg_id)[:N_EEG, :].astype(np.float64)
            total_sum += seg.sum(axis=1)
            total_sq  += (seg ** 2).sum(axis=1)

    mean = total_sum / n
    var  = np.maximum(total_sq / n - mean ** 2, 0.0)
    std  = np.sqrt(var)
    std  = np.where(std < 1e-10, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_window(
    raw: np.ndarray,
    ch_mean: np.ndarray,
    ch_std:  np.ndarray,
) -> np.ndarray:
    """
    Preprocess one raw 10 s EEG window to REVE input format.

    Input  : (64, 10000) float64  Volts @ 1000 Hz
    Output : (64, 2000)  float32  z-scored, clipped @ 200 Hz

    Steps (matching REVE pretraining pipeline, §3.1.1):
      1. Resample 1000 → 200 Hz via scipy resample_poly() (anti-aliasing built in)
      2. Bandpass 0.5–99 Hz (sosfiltfilt, numerically stable near Nyquist)
      3. Z-score per channel using session statistics; clip ±15 σ
    """
    eeg = resample_poly(raw, up=TARGET_FS, down=SOURCE_FS, axis=1)
    eeg = eeg[:, :WINDOW_TARGET_SAMPLES]                         # (64, 2000)
    eeg = sosfiltfilt(_SOS_BP, eeg, axis=1)
    eeg = (eeg - ch_mean[:, None]) / ch_std[:, None]
    eeg = np.clip(eeg, -15.0, 15.0).astype(np.float32)
    return eeg


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_transformer_layers(model, n_layers: int) -> list:
    """
    Locate the n_layers repeated transformer block modules in a REVE model
    loaded via trust_remote_code.

    Strategy:
      1. Try common HuggingFace attribute path conventions.
      2. Fall back to a full recursive search over all named ModuleLists
         whose length equals n_layers — finds the blocks regardless of naming.
    """
    for attr_path in (
        "layers",
        "encoder.layers",
        "blocks",
        "transformer.h",
        "encoder.blocks",
        "model.layers",
        "model.encoder.layers",
    ):
        obj = model
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj) == n_layers:
            print(f"  [layers] Using model.{attr_path} (len={n_layers})")
            return list(obj)

    # Recursive search: walk all named modules, collect every ModuleList of length n_layers
    import torch.nn as nn
    candidates: list[tuple[str, list]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == n_layers:
            candidates.append((name, list(module)))

    if len(candidates) == 1:
        name, layers = candidates[0]
        print(f"  [layers] Found transformer blocks at model.{name} (len={n_layers})")
        return layers
    if len(candidates) > 1:
        # Prefer the deepest path that contains the word "layer" or "block"
        preferred = [(n, l) for n, l in candidates
                     if any(k in n.lower() for k in ("layer", "block", "transformer"))]
        name, layers = preferred[0] if preferred else candidates[0]
        print(f"  [layers] Multiple ModuleLists of len {n_layers} found; "
              f"using model.{name}")
        return layers

    raise RuntimeError(
        f"Could not locate {n_layers} transformer blocks in the loaded REVE model.\n"
        f"Paste the output of print(model) and the attribute path will be clear."
    )


def load_model(size: str, device: torch.device):
    """
    Load REVE encoder and position bank from HuggingFace.

    Returns
    -------
    model              : REVE encoder on device, eval mode
    positions_tensor   : (1, 64, 3) float32 on device — precomputed electrode positions
    transformer_layers : list of 22 transformer block modules (for forward hooks)
    n_layers           : 22
    embed_dim          : 512 (base) or 1250 (large)
    """
    from transformers import AutoModel

    cfg   = MODEL_CONFIGS[size]
    token = _get_hf_token()

    print("  Loading reve-positions (not gated)...")
    pos_bank = AutoModel.from_pretrained(HF_REVE_POSITIONS, trust_remote_code=True)
    pos_bank.eval()

    print(f"  Loading reve-{size} (gated — token required if not already logged in)...")
    model = AutoModel.from_pretrained(
        cfg["hf_repo"],
        trust_remote_code=True,
        token=token,
    )
    model.to(device).eval()

    # Precompute electrode positions for the 64 CineBrain channels.
    # Constant across all windows — computed once and reused.
    with torch.no_grad():
        pos = pos_bank(CH_NAMES_EEG)               # (64, 3)
    positions_tensor = pos.unsqueeze(0).to(device)  # (1, 64, 3)

    n_layers  = cfg["n_layers"]
    embed_dim = cfg["embed_dim"]

    transformer_layers = _find_transformer_layers(model, n_layers)
    print(f"  REVE-{size}: {n_layers} transformer layers, D={embed_dim}, "
          f"{N_TOKENS} tokens/window ({N_EEG} ch × {N_PATCHES} patches)")

    return model, positions_tensor, transformer_layers, n_layers, embed_dim


# ─────────────────────────────────────────────────────────────────────────────
# Layerwise forward pass
# ─────────────────────────────────────────────────────────────────────────────

def embed_window_layerwise(
    model,
    positions: torch.Tensor,
    transformer_layers: list,
    window: np.ndarray,
    n_layers: int,
    embed_dim: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run one preprocessed 10 s window through REVE and capture mean-pooled
    hidden state after each of the 22 transformer blocks.

    Input  : (64, 2000) float32  z-scored EEG @ 200 Hz
    Output : (n_layers, embed_dim) float32

    Forward hooks are attached to each transformer block. Each block output has
    shape (1, N_TOKENS, D) = (1, 704, D); we mean-pool over the token dimension
    to get one D-dim vector per layer, consistent with other extractors.
    """
    # Each "block" in REVE is ModuleList([Attention, FeedForward]); the transformer
    # forward does:  x = attn(x) + x;  x = ff(x) + x.  Because ModuleList is never
    # __call__'d, hooks on the block itself never fire — but hooks on its children
    # (Attention, FeedForward) do. Capture post-block hidden as ff_input + ff_output:
    #   ff_input  = attn(x_prev) + x_prev   (post-attention residual)
    #   ff_output = ff_delta
    #   post_block = ff_input + ff_output   (post-FF residual = block output)
    captured_in:  list[torch.Tensor | None] = [None] * n_layers
    captured_out: list[torch.Tensor | None] = [None] * n_layers
    hooks = []
    for i, block in enumerate(transformer_layers):
        ff = block[1]   # FeedForward
        def make_pre(idx):
            def pre(_m, inp, _idx=idx):
                captured_in[_idx] = inp[0] if isinstance(inp, tuple) else inp
            return pre
        def make_post(idx):
            def post(_m, _i, out, _idx=idx):
                captured_out[_idx] = out[0] if isinstance(out, tuple) else out
            return post
        hooks.append(ff.register_forward_pre_hook(make_pre(i)))
        hooks.append(ff.register_forward_hook(make_post(i)))

    x = torch.from_numpy(window).unsqueeze(0).to(device)   # (1, 64, 2000)

    with torch.no_grad():
        out = model(x, positions)

    for h in hooks:
        h.remove()

    # ── DIAGNOSTICS (printed once per process) ────────────────────────────────
    if not getattr(embed_window_layerwise, "_diag_done", False):
        embed_window_layerwise._diag_done = True
        n_in_fired  = sum(1 for t in captured_in  if t is not None)
        n_out_fired = sum(1 for t in captured_out if t is not None)
        print(f"\n  [diag] FF hooks: {n_in_fired}/{n_layers} pre fired, "
              f"{n_out_fired}/{n_layers} post fired")
        if captured_in[0] is not None and captured_out[0] is not None:
            ti, to = captured_in[0], captured_out[0]
            print(f"  [diag] block[0]  ff_in shape={tuple(ti.shape)} ||x||={float(ti.float().norm()):.4f}"
                  f"  ff_out shape={tuple(to.shape)} ||x||={float(to.float().norm()):.4f}")
            tiL, toL = captured_in[-1], captured_out[-1]
            print(f"  [diag] block[-1] ff_in shape={tuple(tiL.shape)} ||x||={float(tiL.float().norm()):.4f}"
                  f"  ff_out shape={tuple(toL.shape)} ||x||={float(toL.float().norm()):.4f}")
        if isinstance(out, torch.Tensor):
            print(f"  [diag] model() returned Tensor shape={tuple(out.shape)} "
                  f"||x||={float(out.float().norm()):.4f}")
    # ──────────────────────────────────────────────────────────────────────────

    assert all(t is not None for t in captured_in) and all(t is not None for t in captured_out), (
        f"FF hooks did not fire on every block "
        f"(pre={sum(t is not None for t in captured_in)}, "
        f"post={sum(t is not None for t in captured_out)}, expected={n_layers})."
    )

    result = np.zeros((n_layers, embed_dim), dtype=np.float32)
    for layer_idx in range(n_layers):
        # post-block hidden = ff_input + ff_output, then mean-pool over tokens
        post = captured_in[layer_idx] + captured_out[layer_idx]    # (1, N_TOKENS, D)
        pooled = post.mean(dim=1)                                  # (1, D)
        result[layer_idx] = pooled.squeeze(0).cpu().float().numpy()

    if not np.isfinite(result).all() or np.linalg.norm(result) == 0.0:
        raise RuntimeError(
            f"embed_window_layerwise produced an all-zero or non-finite result "
            f"(||result||={np.linalg.norm(result):.4e})."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    size: str,
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
    print(f"  REVE-{size}  |  layers={cfg['n_layers']}  D={cfg['embed_dim']}  "
          f"W={n_win}  S={len(subjects_to_run)}"
          + ("  [TEST MODE]" if test else ""))
    print(f"{'='*64}")

    t0 = time.time()
    model, positions, transformer_layers, n_layers, embed_dim = load_model(size, device)
    print(f"  Model ready in {time.time() - t0:.1f} s")

    embeddings    = np.zeros(
        (n_layers, n_win, len(subjects_to_run), embed_dim), dtype=np.float32
    )
    window_starts = np.zeros(n_win, dtype=np.int64)
    starts_saved  = False

    for s_idx, subject in enumerate(subjects_to_run):
        print(f"\n  [{subject}]")
        tar_path = get_tar_path(subject)

        print(f"  Computing session z-score statistics (streaming pass 1/2)...")
        t_stat = time.time()
        ch_mean, ch_std = compute_session_stats(tar_path)
        print(f"  Stats done in {time.time() - t_stat:.1f} s  "
              f"| mean ∈ [{ch_mean.min():.3e}, {ch_mean.max():.3e}]  "
              f"std ∈ [{ch_std.min():.3e}, {ch_std.max():.3e}]")

        print(f"  Extracting embeddings (streaming pass 2/2)...")
        t_sub = time.time()
        w_idx = -1
        for w_idx, (sample_start, raw_window) in enumerate(
            iter_windows(tar_path, test_indices)
        ):
            window    = preprocess_window(raw_window, ch_mean, ch_std)
            layer_emb = embed_window_layerwise(
                model, positions, transformer_layers,
                window, n_layers, embed_dim, device,
            )
            embeddings[:, w_idx, s_idx, :] = layer_emb

            if not starts_saved:
                window_starts[w_idx] = sample_start

            if (w_idx + 1) % 100 == 0:
                elapsed = time.time() - t_sub
                rate    = (w_idx + 1) / elapsed
                eta     = (n_win - w_idx - 1) / rate
                print(f"    {w_idx+1}/{n_win} windows  "
                      f"{rate:.2f} win/s  ETA {eta/60:.1f} min")

        starts_saved = True
        n_yielded = w_idx + 1
        print(f"  [{subject}] done in {(time.time() - t_sub)/60:.1f} min  "
              f"(windows yielded by iter_windows: {n_yielded}/{n_win})")
        if n_yielded == 0:
            raise RuntimeError(
                f"iter_windows yielded ZERO windows for {subject}. The streaming "
                f"reader is not producing data — check tar segment naming, the "
                f"test_indices subset, or get_tar_path()."
            )
        # Verify this subject's slice is non-zero
        subj_norm = float(np.linalg.norm(embeddings[:, :n_yielded, s_idx, :]))
        print(f"  [{subject}] ||embeddings[:, :, {s_idx}, :]|| = {subj_norm:.4f}")
        if subj_norm == 0.0:
            raise RuntimeError(
                f"Embeddings for {subject} are all zero after the inner loop ran "
                f"({n_yielded} windows). Inspect the [diag] line printed above."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings    = embeddings,
        window_starts = window_starts,
        subjects      = np.array(subjects_to_run),
        channel_names = np.array(CH_NAMES_EEG),
        size          = size,
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
        description="Extract REVE layerwise embeddings from CineBrain Season 7.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--size", choices=list(MODEL_CONFIGS), default=None,
        help="Model size to run. Omit to run both (base then large) sequentially.",
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
             "Saves reve_{base|large}_test.npz with shape (22, 50, 1, D).",
    )
    return ap.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    device  = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Device   : {device}")
    print(f"Output   : {out_dir}")
    print(f"Windows  : {N_WINDOWS} per subject × {len(SUBJECTS)} subjects × 10 s "
          f"(2.5 clips × 4 s)")
    print(f"Channels : {N_EEG} (all confirmed in brain-bzh/reve-positions)")
    print(f"Patches  : {N_PATCHES}/channel × {N_EEG} channels = {N_TOKENS} tokens/window")
    print(f"Access   : reve-base and reve-large are gated. Accept the Responsible Use")
    print(f"           Agreement at https://huggingface.co/brain-bzh/reve-base")
    print(f"           and set HF_TOKEN env variable before running.")
    print(f"Note     : Each subject requires two streaming passes over the tar (~12 GB)")
    print(f"           to compute session z-score statistics before extraction.")

    if args.test:
        print("TEST MODE — 50 random windows, sub-0001 only.")

    sizes = [args.size] if args.size else list(MODEL_CONFIGS)
    for size in sizes:
        run(size, out_dir, device, test=args.test)

    print("\nDone.")


if __name__ == "__main__":
    main()
