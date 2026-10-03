#!/usr/bin/env python
"""
compute_mknn_cyclic.py — cross-architecture mKNN under the CYCLIC null (see CYCLIC_NULL.md).

Rebuilds the y axis of RESULTS.md panel (b) — mean calibrated mKNN to models of other
families — with the null-calibration framework of aristotelian.pdf, but with the
permutation group Pi_n set to the CYCLIC group C_n (circular shifts) rather than the full
symmetric group S_n. Our samples are consecutive windows of one recording, so they are not
exchangeable and the paper's Assumption 3.1 fails under a free permutation; C_n is the
subgroup that leaves the temporal autocorrelation intact.

WHAT THE PUBLISHED PIPELINE ACTUALLY DOES (verified, and not what the write-up implies):
`eeg_feature_alignment.py` builds MK as

    emb  = IC.load_eeg_emb(m, s)             # (L,W,S,D) -> .mean(axis=2), SUBJECTS AVERAGED
    ce   = to_common(emb, win_sec)           # -> the shared 10 s / 1080-bin grid
    knn  = C.precompute_knn_layers(ce, k)
    mk   = IC.best_layer_pair(knn_i, knn_j)[2]   # max over layer pairs, RAW

so MK is (a) computed on subject-mean embeddings — there is no subject axis anywhere on
the alignment side, which is why eeg_alignment and eeg_alignment_persubj carry a
byte-identical MK — and (b) NOT calibrated at all. No permutations, no null. This script
supplies the first null that axis has had.

Reproduced here exactly: subject-mean embeddings, to_common downsampling, cosine kNN with
k=5, max over all layer pairs. Only the calibration is added.

COST NOTE. mKNN under rotation is a circular cross-correlation over neighbour offsets, so
one FFT per layer yields ALL W rotations at once (shift_tests/fast_shift.py). Getting the
complete C_n null therefore costs about the same as getting a handful of shifts.

Saved (small on purpose):
    cyclic_mknn.npz
        curves  (n_pairs, W)  float32   mKNN at every rotation, index 0 == d=0
        pairs   (n_pairs, 2)  int       indices into `keys`
        keys, family, n_layers, win_sec, k_mknn, W
    knn_common.npz
        <key>   (L, 1080, 5)  int32     so any kNN-based redo needs no embeddings

From `curves` you can afterwards derive any far-tail cutoff, any alpha, and the
siblings-included variant with no recomputation.

Usage:
    python compute_mknn_cyclic.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))
sys.path.insert(0, str(_HERE.parent / "shift_tests"))

import _common as C  # noqa: E402
import fast_shift as FS  # noqa: E402

CACHE = _HERE / "cache"
REPO_EEG = "nitrox639/platonic-embeddings"
COMMON_WIN = 10          # seconds per common bin   (eeg_feature_alignment.COMMON_WIN)
N_COMMON = 1080          # bins                     (eeg_feature_alignment.N_COMMON)


def _tok():
    p = _HERE.parents[1] / "tokens" / "hf_token.txt"
    return p.read_text().strip() if p.exists() else None


def to_common(emb, win_sec, common_win=COMMON_WIN, n_common=N_COMMON):
    """(L,W,D) on a model's native grid -> (L,n_common,D) on the shared 10 s grid.

    Verbatim from eeg_feature_alignment.to_common: each native window is assigned to the
    bin containing its CENTRE time, and bins average the windows that land in them.
    Necessary because the five families tile Season 7 differently (5/6/8/10 s), and mKNN
    compares row i of one model to row i of another — they must mean the same instant.
    """
    L, W, D = emb.shape
    centers = (np.arange(W) + 0.5) * win_sec
    b = np.clip((centers // common_win).astype(int), 0, n_common - 1)
    out = np.zeros((L, n_common, D), dtype=np.float64)
    cnt = np.zeros(n_common)
    for w in range(W):
        out[:, b[w]] += emb[:, w]
        cnt[b[w]] += 1
    out /= np.maximum(cnt, 1)[None, :, None]
    return out


def phase1_knn(keys, k, overwrite=False):
    """Load each embedding once, subject-mean it, downsample, kNN, free. -> knn_common.npz"""
    out = CACHE / "knn_common.npz"
    if out.exists() and not overwrite:
        print(f"[phase 1] {out.name} exists, reusing")
        d = np.load(out, allow_pickle=True)
        return {kk: d[kk] for kk in keys if kk in d.files}
    from huggingface_hub import hf_hub_download
    CACHE.mkdir(parents=True, exist_ok=True)
    knn = {}
    for key in keys:
        m, s = key.rsplit("-", 1)
        t0 = time.time()
        emb = np.load(hf_hub_download(REPO_EEG, C.EEG[m]["fname"](s), repo_type="dataset",
                                      token=_tok()), allow_pickle=True)["embeddings"]
        mean_emb = emb.mean(axis=2)                     # SUBJECTS AVERAGED, as published
        del emb
        ce = to_common(mean_emb, C.EEG_WINDOWS[m]["win_sec"])
        del mean_emb
        knn[key] = C.precompute_knn_layers(ce, k).astype(np.int32)
        print(f"  [{key:18}] L={knn[key].shape[0]:2d} -> knn {knn[key].shape} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del ce
    np.savez_compressed(out, **knn)
    print(f"[phase 1] saved {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return knn


def phase1_knn_persubj(keys, k, overwrite=False):
    """Same as phase1_knn but WITHOUT averaging subjects: kNN inside each subject.

    This is the quantity the published pipeline never computes. `load_eeg_emb` averages
    the subject axis before anything else, so `kNN(mean of embeddings)` is what MK holds;
    here we build `kNN(each subject)` so the mean of per-subject alignment can be taken
    afterwards. The two are not equal, and averaging first is the one that inflates:
    subject-idiosyncratic variance is cut ~6x before the neighbourhoods are formed.

    -> {key: (S, L, n_common, k) int32}
    """
    out = CACHE / "knn_common_persubj.npz"
    if out.exists() and not overwrite:
        print(f"[phase 1] {out.name} exists, reusing")
        d = np.load(out, allow_pickle=True)
        return {kk: d[kk] for kk in keys if kk in d.files}
    from huggingface_hub import hf_hub_download
    CACHE.mkdir(parents=True, exist_ok=True)
    knn = {}
    for key in keys:
        m, s = key.rsplit("-", 1)
        t0 = time.time()
        emb = np.load(hf_hub_download(REPO_EEG, C.EEG[m]["fname"](s), repo_type="dataset",
                                      token=_tok()), allow_pickle=True)["embeddings"]
        L, W, S, D = emb.shape
        win = C.EEG_WINDOWS[m]["win_sec"]
        per = []
        for si in range(S):
            ce = to_common(emb[:, :, si, :], win)
            per.append(C.precompute_knn_layers(ce, k).astype(np.int32))
            del ce
        knn[key] = np.stack(per)                       # (S, L, n_common, k)
        del emb, per
        print(f"  [{key:18}] {knn[key].shape} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(out, **knn)
    print(f"[phase 1] saved {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return knn


def phase2_curves_persubj(knn, keys, k):
    """All pairs x all rotations, INSIDE each subject. -> curves (n_pairs, S, W).

    Subjects are the outer loop so only one subject's FFT stack (~1.1 GB) is resident;
    each model's transform is still built exactly once per subject.
    """
    W = N_COMMON
    S = knn[keys[0]].shape[0]
    n = len(keys)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    curves = np.zeros((len(pairs), S, W), dtype=np.float32)
    for si in range(S):
        t0 = time.time()
        F = {key: FS.precompute_fft(knn[key][si], W) for key in keys}
        for pi, (i, j) in enumerate(pairs):
            curves[pi, si] = FS.max_mknn_all_shifts(F[keys[i]], F[keys[j]], W, k)
        del F
        print(f"  [subject {si+1}/{S}] {len(pairs)} pairs ({time.time()-t0:.0f}s)", flush=True)
    return np.array(pairs), curves, W


def phase2_curves(knn, keys, k):
    """All 91 pairs, every rotation, via FFT. -> curves (n_pairs, W)."""
    W = N_COMMON
    t0 = time.time()
    F = {}
    for key in keys:
        F[key] = FS.precompute_fft(knn[key], W)
        print(f"  [fft {key:18}] {F[key].shape} "
              f"({sum(v.nbytes for v in F.values())/1e9:.2f} GB resident)", flush=True)
    pairs, curves = [], []
    n = len(keys)
    done = 0
    total = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            curves.append(FS.max_mknn_all_shifts(F[keys[i]], F[keys[j]], W, k).astype(np.float32))
            pairs.append((i, j))
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [pairs] {done}/{total} ({time.time()-t0:.0f}s)", flush=True)
    return np.array(pairs), np.stack(curves), W


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--per-subject", action="store_true",
                    help="kNN inside each subject instead of on the subject-mean "
                         "embedding. Writes cyclic_mknn_persubj.npz.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    C.HF_TOKEN_CACHE = _tok()

    print("── self-test (synthetic, no network) ──")
    if not FS.self_test():
        raise SystemExit("fast_shift self-test FAILED — refusing to run on real data.")

    keys = [f"{m}-{s}" for m in C.EEG for s in C.EEG[m]["sizes"]]
    fam = [k.rsplit("-", 1)[0] for k in keys]
    mode = "per-subject" if args.per_subject else "subject-mean"
    print(f"\n[phase 1] kNN on the common {COMMON_WIN}s / {N_COMMON}-bin grid, "
          f"{mode} embeddings")
    knn = (phase1_knn_persubj if args.per_subject else phase1_knn)(
        keys, args.k_mknn, args.overwrite)
    keys = [k for k in keys if k in knn]

    print(f"\n[phase 2] {len(keys)*(len(keys)-1)//2} pairs x {N_COMMON} rotations (FFT)")
    pairs, curves, W = (phase2_curves_persubj if args.per_subject else phase2_curves)(
        knn, keys, args.k_mknn)

    out = CACHE / ("cyclic_mknn_persubj.npz" if args.per_subject else "cyclic_mknn.npz")
    np.savez_compressed(out, curves=curves, pairs=pairs, keys=np.array(keys),
                        family=np.array(fam),
                        n_layers=np.array([knn[k].shape[-3] for k in keys]),
                        win_sec=np.array([C.EEG_WINDOWS[k.rsplit('-', 1)[0]]["win_sec"]
                                          for k in keys]),
                        k_mknn=args.k_mknn, W=W, common_win=COMMON_WIN)
    print(f"\nsaved {out}  curves={curves.shape}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
