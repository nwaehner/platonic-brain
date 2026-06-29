"""
eeg_features.py — NICE-style EEG-derived features per window, from the RAW CineBrain EEG.

Motivation: vision/language stimulus features are not linearly decodable from EEG foundation-
model embeddings (grid-search R² ≈ 0). EEG-DERIVED markers (the classic NICE pipeline:
band power, spectral entropy, permutation entropy, Hjorth, complexity) come from the same
signal the model encoded, so the embedding should carry them — giving positive R² and making
the alignment↔decodability question meaningful for the brain models.

Reuses the raw-EEG access pattern of extraction_scripts/extract_femba.py (copied minimally so
this needs NO torch / model): per-subject tar `sub-XXXX/EEG_preprocessed_data.tar` of
`eeg_02/<seg>.npy` segments (69×800, first 64 = EEG channels) @ 1000 Hz, Season-7 segments
0–13499 (= 10 800 s), 6 subjects. CineBrain is filtered 0.1–30 Hz + 50 Hz notch → no usable
gamma, so bands stop at β (13–30 Hz).

Features (core set; median over the 64 channels → one scalar per window; wSMI deferred):
  abs/rel power δ θ α β (8) · spectral entropy · median freq · spectral edge 90 (3) ·
  permutation entropy broadband/θ/α (3) · Hjorth mobility/complexity (2) ·
  Lempel-Ziv complexity (1)   → 17 scalars.

Windows match each EEG model's grid exactly (non-overlapping tiling from t=0, win_sec from the
model's window scheme): femba/luna 5 s, neurolm 8 s, steegformer 6 s, reve 10 s. One job per
SUBJECT computes all distinct grids from a single tar read (peak ~3 GB raw in RAM). Subject
features are averaged later (combine_eeg_features.py) to match the subject-mean embeddings.

Output → outputs/eeg_features/raw/<subject>__<grid>__eegfeat.npz  (feat (W,F), feat_names).

Usage:
  python src/interp/eeg_features.py --subject sub-0001            # all grids, one subject
  python src/interp/eeg_features.py --subject sub-0001 --test     # 40 windows, quick check
"""

from __future__ import annotations

import argparse
import io
import math
import os
import tarfile
from pathlib import Path

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, resample_poly

import _interp_common as IC  # only for OUT_DIR + token resolution

HF_DATASET_REPO = "Fudan-fMRI/CineBrain"
SUBJECTS = [f"sub-{i:04d}" for i in range(1, 7)]
MAX_SEGMENT = 13500
SEG_SAMPLES = 800
FS = 1000              # CineBrain source sampling rate
FEAT_FS = 128         # decimate to 128 Hz before features: signal is ≤30 Hz (0.1–30 filter),
RES_UP, RES_DOWN = 16, 125   # 1000 × 16/125 = 128 Hz exactly. 8× less data → fast serial compute.
N_CH = 64

# distinct window schemes (grid name → win_sec). luna shares femba's 5 s (copied at combine).
GRID_WIN_SEC = {"femba": 5, "neurolm": 8, "steegformer": 6, "reve": 10}
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
SPEC_LO, SPEC_HI = 1.0, 30.0

OUT_RAW = IC.OUT_DIR / "eeg_features" / "raw"


# ── raw-EEG access — STREAM the tar over HTTP (no 12 GB written to disk) ───────────
# /scratch is quota-limited, so we never download the full tar. tarfile "r|" reads members
# sequentially from a streamed file object; we index each segment into the preallocated
# raw array by its id, so storage order is irrelevant. Peak disk ≈ 0, peak RAM ≈ 2.8 GB.
def load_subject_raw(subject, max_segment=MAX_SEGMENT):
    """(64, max_segment*800) float32 @ 1000 Hz, streamed from HuggingFace (no disk cache).
    Uses requests stream=True → tarfile 'r|' so data flows at network speed, never to disk."""
    import requests
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(HF_DATASET_REPO, f"{subject}/EEG_preprocessed_data.tar", repo_type="dataset")
    headers = {}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    N = max_segment * SEG_SAMPLES
    raw = np.empty((N_CH, N), dtype=np.float32)
    seen = 0
    print(f"  [{subject}] streaming EEG tar over HTTP (no disk download) …", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=120,
                      allow_redirects=True) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|") as tar:     # streaming, forward-only
            for member in tar:
                name = member.name
                if not (name.endswith(".npy") and "/" in name):
                    continue
                base = name.split("/")[-1][:-4]
                if not base.isdigit():
                    continue
                seg_id = int(base)
                if seg_id >= max_segment:
                    continue
                arr = np.load(io.BytesIO(tar.extractfile(member).read()))
                raw[:, seg_id * SEG_SAMPLES:(seg_id + 1) * SEG_SAMPLES] = arr[:N_CH, :SEG_SAMPLES]
                seen += 1
                if seen == 1 or seen % 1000 == 0:
                    print(f"    [{subject}] streamed {seen} segments (latest id {seg_id})", flush=True)
                if seen >= max_segment:
                    break
    if seen < max_segment:
        print(f"  [{subject}] WARNING: streamed {seen}/{max_segment} segments", flush=True)
    return raw


# ── NICE feature computation (per window, median over channels) ───────────────────
def _bandpass(x, lo, hi, fs=FEAT_FS):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _perm_entropy(sig, order=3, delay=1):
    """Normalised permutation entropy of a 1-D signal (Bandt–Pompe)."""
    n = len(sig)
    npat = n - (order - 1) * delay
    if npat <= 1:
        return np.nan
    # ordinal patterns via argsort of each embedded vector
    idx = np.arange(order) * delay
    patterns = np.argsort(np.stack([sig[i:i + npat] for i in idx], axis=1), axis=1)
    # hash each permutation to an int
    mult = (order ** np.arange(order))
    codes = (patterns * mult).sum(1)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(order)))


def _hjorth(sig):
    """(mobility, complexity) Hjorth parameters of a 1-D signal."""
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    v0 = np.var(sig) + 1e-20
    v1 = np.var(d1) + 1e-20
    v2 = np.var(d2) + 1e-20
    mob = np.sqrt(v1 / v0)
    com = np.sqrt(v2 / v1) / mob
    return float(mob), float(com)


def _lziv(sig):
    """Normalised Lempel-Ziv complexity of the median-binarised 1-D signal."""
    b = (sig > np.median(sig)).astype(np.uint8)
    s = b.tobytes()
    i, k, l = 0, 1, 1
    c, n = 1, len(b)
    kmax = 1
    while True:
        if l + k > n:
            c += 1
            break
        if s[i:i + k] == s[l:l + k]:
            k += 1
            if k > kmax:
                kmax = k
        else:
            i += 1
            if i == l:
                c += 1
                l += kmax
                i = 0
                k = 1
                kmax = 1
            else:
                k = 1
    return float(c * np.log(n) / n) if n > 1 else 0.0


def nice_features(win, fs=FEAT_FS):
    """Dict of NICE scalars for one (64, T) window (median over channels)."""
    nperseg = int(min(win.shape[1], fs * 2))
    f, Pxx = welch(win, fs=fs, nperseg=nperseg, axis=1)        # (64, nf)
    sband = (f >= SPEC_LO) & (f <= SPEC_HI)
    Ptot = Pxx[:, sband].sum(1) + 1e-20
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp = Pxx[:, m].sum(1)
        out[f"abs_{name}"] = float(np.median(np.log(bp + 1e-20)))
        out[f"rel_{name}"] = float(np.median(bp / Ptot))
    Pn = Pxx[:, sband] / Ptot[:, None]
    se = -(Pn * np.log(Pn + 1e-20)).sum(1) / np.log(int(sband.sum()))
    out["spec_entropy"] = float(np.median(se))
    fb = f[sband]
    cum = np.cumsum(Pxx[:, sband], axis=1) / (Pxx[:, sband].sum(1, keepdims=True) + 1e-20)
    mfreq = [fb[min(int(np.searchsorted(cum[c], 0.5)), len(fb) - 1)] for c in range(N_CH)]
    edge = [fb[min(int(np.searchsorted(cum[c], 0.9)), len(fb) - 1)] for c in range(N_CH)]
    out["median_freq"] = float(np.median(mfreq))
    out["spec_edge90"] = float(np.median(edge))
    out["pe_broad"] = float(np.median([_perm_entropy(win[c]) for c in range(N_CH)]))
    th = _bandpass(win, 4, 8); al = _bandpass(win, 8, 13)
    out["pe_theta"] = float(np.median([_perm_entropy(th[c]) for c in range(N_CH)]))
    out["pe_alpha"] = float(np.median([_perm_entropy(al[c]) for c in range(N_CH)]))
    hj = [_hjorth(win[c]) for c in range(N_CH)]
    out["hjorth_mobility"] = float(np.median([h[0] for h in hj]))
    out["hjorth_complexity"] = float(np.median([h[1] for h in hj]))
    return out  # NB: Lempel-Ziv/KC complexity (O(n²)) deferred to the wSMI fast-follow


def compute_grid(raw, win_sec, max_windows=None):
    """raw is already decimated to FEAT_FS. Serial loop (per-window compute is cheap at
    128 Hz; avoids loky worker-import hangs on the cluster)."""
    win = win_sec * FEAT_FS
    W = raw.shape[1] // win
    if max_windows:
        W = min(W, max_windows)
    rows = []
    for w in range(W):
        rows.append(nice_features(raw[:, w * win:(w + 1) * win]))
        if w % 400 == 0:
            print(f"      window {w}/{W}", flush=True)
    names = list(rows[0])
    M = np.array([[r[n] for n in names] for r in rows], dtype=np.float32)
    return M, names


def run(subject, n_jobs, test=False):
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    max_seg = 400 if test else MAX_SEGMENT
    raw = load_subject_raw(subject, max_segment=max_seg)
    print(f"  [{subject}] raw loaded {raw.shape} @1000Hz; decimating → {FEAT_FS}Hz …", flush=True)
    raw = resample_poly(raw, RES_UP, RES_DOWN, axis=1).astype(np.float32)
    print(f"  [{subject}] decimated {raw.shape} ({raw.nbytes/1e9:.2f} GB)", flush=True)
    mw = 40 if test else None
    for grid, ws in GRID_WIN_SEC.items():
        M, names = compute_grid(raw, ws, max_windows=mw)
        p = OUT_RAW / f"{subject}__{grid}{'__test' if test else ''}__eegfeat.npz"
        np.savez_compressed(p, feat=M, feat_names=np.array(names), subject=subject,
                            grid=grid, win_sec=ws)
        print(f"  [{subject}] {grid}: feat {M.shape} → {p.name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="NICE EEG-derived features from raw CineBrain EEG.")
    ap.add_argument("--subject", required=True, choices=SUBJECTS)
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    run(args.subject, args.n_jobs, test=args.test)


if __name__ == "__main__":
    main()
