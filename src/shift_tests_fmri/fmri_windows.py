"""
fmri_windows.py — the CineBrain fMRI as a window "embedding".

Turns the per-subject raw TR matrices built by `extraction_scripts/extract_fmri.py`
into the exact (L, W, S, D) tensor every `_common` / `shift_tests` primitive expects,
so the fMRI can be fed to the circular-shift machinery as if it were a foundation
model's embedding. It is not one — it is the brain state itself — but the geometry
question ("do these two spaces agree about which windows are neighbours?") is
identical, and that is all mKNN needs.

  L = 1   fMRI has no layers. Every primitive here (precompute_knn, mknn_matrix,
          precompute_fft) already handles L=1, so nothing is special-cased.
  W       10800 s / win_sec. For the clip4s grid (4 s) that is 2700 — one window per
          Season-7 clip, the stimulus-native tiling.
  S       however many fmri_raw_*.npy files exist in FMRI_DIR. sub-0005 is an upstream
          duplicate of sub-0001 (byte-identical payload in the HF tar), so dropping
          that one file is all it takes to run at n=5 — no code change.
  D       18946 masked voxels, already z-scored upstream.

`load_fmri_raw` and `window_pool` are ported from `alignment_plots/fmri_vs_all.py`
so the two analyses pool TRs into windows in exactly the same way.

NOTE on the two different "shifts" in this package:
  * `shift_sec` here is the HRF shift — a real, physical realignment of the fMRI
    against the stimulus, applied while pooling TRs. Kept at 0 by default.
  * the circular shift `d` used by the null is a pure index rotation applied LATER,
    to the vision side only. They are unrelated; do not conflate them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

FMRI_DIR = Path(os.environ.get("FMRI_DIR", _PROJECT_ROOT / "data" / "fmri"))
ALL_SUBJECTS = [f"sub-{i:04d}" for i in range(1, 7)]

TR_SECONDS = 0.8
SEASON7_SEC = 10800
EXPECTED_D = 18946
EXPECTED_TR = 13500


def load_fmri_raw(fmri_dir: Path = None, subjects=None):
    """Memmap every per-subject (13500, D) matrix that exists. Returns (mms, subs).

    Memmapping keeps the raw stack off the heap — only the pooled window embedding is
    ever materialised. Missing subjects are skipped with a warning rather than an
    error: that is the mechanism by which sub-0005 is excluded.
    """
    d = Path(fmri_dir) if fmri_dir is not None else FMRI_DIR
    subs_want = subjects if subjects is not None else ALL_SUBJECTS
    mms, subs = [], []
    for s in subs_want:
        p = d / f"fmri_raw_{s}.npy"
        if not p.exists():
            print(f"  [skip] {p.name} not in {d} — excluded from this run")
            continue
        mms.append(np.load(p, mmap_mode="r"))
        subs.append(s)
    if not mms:
        raise SystemExit(f"No fmri_raw_*.npy in {d}. Run extract_fmri.py first.")
    T, D = mms[0].shape
    for m, s in zip(mms, subs):
        if m.shape != (T, D):
            raise ValueError(f"{s}: shape {m.shape} != {(T, D)} — inconsistent matrices")
    if T != EXPECTED_TR:
        print(f"  [warn] T={T}, expected {EXPECTED_TR}")
    if D != EXPECTED_D:
        print(f"  [warn] D={D}, expected {EXPECTED_D}")
    return mms, subs


def window_pool(raw_mms, win_sec, shift_sec=0.0):
    """list of S (T, D) memmaps -> (1, W, S, D) float32.

    Mean-pools the TRs whose CENTRE time falls in the HRF-shifted window
    [w*L + h, (w+1)*L + h). Reads only the pooled rows off disk per window.
    """
    S = len(raw_mms)
    T, D = raw_mms[0].shape
    W = int(SEASON7_SEC // win_sec)
    centres = (np.arange(T) + 0.5) * TR_SECONDS
    out = np.zeros((1, W, S, D), dtype=np.float32)
    for w in range(W):
        lo = w * win_sec + shift_sec
        hi = (w + 1) * win_sec + shift_sec
        sel = np.where((centres >= lo) & (centres < hi))[0]
        if sel.size == 0:                       # shifted past the end
            sel = np.array([min(T - 1, int(round(lo / TR_SECONDS)))])
        for s in range(S):
            out[0, w, s] = np.asarray(raw_mms[s][sel, :]).mean(axis=0)
    return out


def fmri_embedding(win_sec, shift_sec=0.0, fmri_dir=None, subjects=None, verbose=True):
    """Convenience: load + pool in one call. Returns (emb (1,W,S,D), subs)."""
    mms, subs = load_fmri_raw(fmri_dir, subjects)
    if verbose:
        print(f"  fMRI: {len(subs)} subjects {subs}, raw {mms[0].shape} (mmap)")
    emb = window_pool(mms, win_sec, shift_sec)
    if verbose:
        print(f"  fMRI window embedding: {emb.shape}  "
              f"(win={win_sec}s, HRF shift={shift_sec}s, {emb.nbytes/1e9:.2f} GB)")
    return emb, subs
