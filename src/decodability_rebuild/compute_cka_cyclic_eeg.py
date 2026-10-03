#!/usr/bin/env python
"""
compute_cka_cyclic_eeg.py — cross-architecture EEG x EEG linear CKA at every rotation.

The last cell of the 2x2. Panel (b) is plotted from mKNN and panel (c) from CKA, so
metric and axis have been confounded throughout: any difference between the panels could
be the modality pair OR the metric. Filling in EEG x EEG with CKA separates them.

    axis            mKNN                       CKA
    (b) EEG x EEG   cyclic-cal +0.727 p=.003   <- THIS SCRIPT
    (c) fMRI x EEG  cyclic-cal +0.134 p=.648   cyclic-cal +0.196 p=.503

Same pipeline as compute_mknn_cyclic.py — subject-mean embeddings (as
`eeg_feature_alignment.py` does via `load_eeg_emb` -> `.mean(axis=2)`), to_common
downsampling to the shared 10 s / 1080-bin grid, max over all layer pairs — with CKA
substituted for mKNN and calibrated against the cyclic group C_n (CYCLIC_NULL.md).

`eeg_feature_alignment.py` builds CK with `best_cka(grams_i, grams_j)`: raw, no null,
exactly as it builds MK. So this adds the first calibration that axis has had, too.

CKA under rotation is a circular cross-correlation over the Gram's DIAGONALS — the
identity derived and self-tested in compute_fmri_cka_cyclic.py — so one rFFT per layer
gives every rotation. Unlike the fMRI case both sides have many layers here, so the max
runs over L_A x L_B pairs and is re-taken at every rotation.

Saved -> cache/cyclic_cka_eeg[_persubj].npz
    curves (n_pairs[, S], W) float32, pairs, keys, family

Usage:
    python compute_cka_cyclic_eeg.py
    python compute_cka_cyclic_eeg.py --per-subject
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import _common as C  # noqa: E402
from compute_mknn_cyclic import to_common, COMMON_WIN, N_COMMON, REPO_EEG, _tok  # noqa: E402
from compute_fmri_cka_cyclic import norm_gram, diag_rows, self_test  # noqa: E402

CACHE = _HERE / "cache"


def grams_fft(ce, W):
    """(L,W,D) -> (L, W, W//2+1) complex64: rFFT of each layer's Gram diagonals."""
    out = np.empty((ce.shape[0], W, W // 2 + 1), dtype=np.complex64)
    for l in range(ce.shape[0]):
        G = norm_gram(C.l2(ce[l]).astype(np.float64))
        out[l] = np.fft.rfft(diag_rows(G), axis=1).astype(np.complex64)
        del G
    return out


def max_cka_all_shifts(Fa, Fb, W):
    """Max over all (a-layer, b-layer) pairs, at every rotation. Fa (LA,W,F), Fb (LB,W,F).

    The pair is re-maximised independently at each shift, so the null keeps the same
    optimisation advantage as d=0 — the aggregation-aware requirement of the paper.
    """
    best = np.full(W, -np.inf)
    for la in range(Fa.shape[0]):
        for lb in range(Fb.shape[0]):
            cur = np.fft.irfft((Fa[la] * np.conj(Fb[lb])).sum(axis=0), n=W)
            np.maximum(best, cur, out=best)
    return best


def load_ce(key, subject=None):
    """to_common'd embedding: subject-mean (default) or one subject."""
    from huggingface_hub import hf_hub_download
    m, s = key.rsplit("-", 1)
    emb = np.load(hf_hub_download(REPO_EEG, C.EEG[m]["fname"](s), repo_type="dataset",
                                  token=_tok()), allow_pickle=True)["embeddings"]
    src = emb.mean(axis=2) if subject is None else emb[:, :, subject, :]
    del emb
    return to_common(src, C.EEG_WINDOWS[m]["win_sec"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-subject", action="store_true")
    args = ap.parse_args()

    print("── self-test ──")
    if not self_test():
        raise SystemExit("cka_all_shifts self-test FAILED.")
    C.HF_TOKEN_CACHE = _tok()

    keys = [f"{m}-{s}" for m in C.EEG for s in C.EEG[m]["sizes"]]
    fam = [k.rsplit("-", 1)[0] for k in keys]
    W, n = N_COMMON, len(keys)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    S = 6 if args.per_subject else 1
    curves = np.zeros((len(pairs), S, W), dtype=np.float32)

    for si in range(S):
        t0 = time.time()
        F = {}
        for key in keys:
            ce = load_ce(key, si if args.per_subject else None)
            F[key] = grams_fft(ce, W)
            del ce
            print(f"  [{'sub%d ' % (si+1) if args.per_subject else ''}{key:18}] "
                  f"{F[key].shape}  "
                  f"({sum(v.nbytes for v in F.values())/1e9:.2f} GB)", flush=True)
        for pi, (i, j) in enumerate(pairs):
            curves[pi, si] = max_cka_all_shifts(F[keys[i]], F[keys[j]], W)
            if (pi + 1) % 20 == 0:
                print(f"    [pairs] {pi+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)
        del F
        print(f"  [view {si+1}/{S}] done ({time.time()-t0:.0f}s)\n", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / ("cyclic_cka_eeg_persubj.npz" if args.per_subject else "cyclic_cka_eeg.npz")
    np.savez_compressed(out, curves=curves if args.per_subject else curves[:, 0],
                        pairs=np.array(pairs), keys=np.array(keys), family=np.array(fam),
                        W=W, common_win=COMMON_WIN)
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
