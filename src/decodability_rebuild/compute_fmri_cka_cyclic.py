#!/usr/bin/env python
"""
compute_fmri_cka_cyclic.py — fMRI x EEG linear CKA at EVERY circular rotation.

The y axis of RESULTS.md panel (c), calibrated with the cyclic null of CYCLIC_NULL.md
(aristotelian.pdf Eqs. 15-17 with Pi_n = C_n instead of S_n).

CKA UNDER ROTATION IS A CIRCULAR CROSS-CORRELATION — the identity that makes the exact
test affordable, and the CKA analogue of what fast_shift.py does for mKNN.

Rotating one side by d permutes its Gram's rows AND columns:  G'[i,j] = G[i-d, j-d].
So with  C(d) = sum_ij A[i,j] B[(i-d)%W, (j-d)%W]  and the substitution i'=i-d, j'=j-d,

    C(d) = sum_{i',j'} A[i'+d, j'+d] B[i', j']

Both indices shift together, so a pair can only ever meet along the SAME circular
diagonal k = i-j. Writing D_X[k, i] = X[i, (i-k) % W],

    C(d) = sum_k ( D_A[k,:] (x) D_B[k,:] )(d)

one 1-D circular cross-correlation per diagonal. One rFFT of each (W,W) diagonal matrix
gives every rotation at once, at O(W^2 log W) instead of O(W^3) for the naive sweep.
`self_test()` checks this against a literal recount before any real data is touched.

Departures from the published panel (c), both deliberate and both explained in
compute_fmri_cka.py: n=5 subjects (sub-0005's fMRI duplicates sub-0001 upstream) and
subjects matched BY ID rather than by array position.

Saved -> cache/cyclic_cka.npz
    <key>__curves   (S, W)  float32   CKA at every rotation, index 0 == d=0
    <key>__subjects (S,)              which subjects, in row order
    keys, family, win_sec, n_layers

Usage:
    python compute_fmri_cka_cyclic.py
    python compute_fmri_cka_cyclic.py --self-test-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))
sys.path.insert(0, str(_HERE.parent / "shift_tests_fmri"))

import _common as C  # noqa: E402
import fmri_windows as FW  # noqa: E402

CACHE = _HERE / "cache"
REPO_EEG = "nitrox639/platonic-embeddings"


def _tok():
    p = _HERE.parents[1] / "tokens" / "hf_token.txt"
    return p.read_text().strip() if p.exists() else None


def norm_gram(X):
    """Feature-centred, Frobenius-normalised Gram — verbatim from fmri_vs_all.norm_gram."""
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T
    return G / (np.linalg.norm(G) + 1e-20)


def diag_rows(G):
    """(W,W) -> (W,W) with row k the k-th circular diagonal: D[k,i] = G[i, (i-k) % W]."""
    W = G.shape[0]
    i = np.arange(W)[None, :]
    k = np.arange(W)[:, None]
    return G[i, (i - k) % W]


def cka_all_shifts(G_a, G_b):
    """CKA at every rotation d of B. Returns (W,), index d meaning B'[i,j]=B[i-d, j-d]."""
    W = G_a.shape[0]
    Fa = np.fft.rfft(diag_rows(G_a), axis=1)
    Fb = np.fft.rfft(diag_rows(G_b), axis=1)
    return np.fft.irfft((Fa * np.conj(Fb)).sum(axis=0), n=W)


def _brute(G_a, G_b, d):
    """Literal recount at one rotation, independent of the transform above."""
    W = G_a.shape[0]
    idx = (np.arange(W) - d) % W
    return float((G_a * G_b[np.ix_(idx, idx)]).sum())


def self_test(verbose=True):
    rng = np.random.default_rng(0)
    ok = True
    for W, D in [(37, 5), (64, 9), (129, 7)]:
        A = norm_gram(rng.normal(size=(W, D)))
        B = norm_gram(rng.normal(size=(W, D)))
        fast = cka_all_shifts(A, B)
        ds = [0, 1, 2, 7, W // 3, W - 1]
        ref = np.array([_brute(A, B, d) for d in ds])
        good = np.allclose(fast[ds], ref, atol=1e-9)
        ok &= good
        if verbose:
            print(f"  W={W:4d}: FFT == brute force at d={ds} : "
                  f"{'OK' if good else 'MISMATCH'}  (max err {np.abs(fast[ds]-ref).max():.2e})")
    if verbose:
        print(f"── cka_all_shifts self-test {'PASSED' if ok else 'FAILED'} ──")
    return bool(ok)


def run_model(model, size, fmri_cache, store, overwrite=False):
    key = f"{model}-{size}"
    if f"{key}__curves" in store and not overwrite:
        print(f"[{key}] cached, skipping")
        return
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    z = np.load(hf_hub_download(REPO_EEG, C.EEG[model]["fname"](size),
                                repo_type="dataset", token=_tok()), allow_pickle=True)
    emb = z["embeddings"]
    eeg_subs = [str(s) for s in z["subjects"]]
    L, We, _, D = emb.shape
    win = C.EEG_WINDOWS[model]["win_sec"]
    fm_emb, fm_subs = fmri_cache[win]
    W = min(We, fm_emb.shape[1])
    pairs = [(fm_subs.index(s), eeg_subs.index(s)) for s in fm_subs if s in eeg_subs]
    print(f"[{key}] L={L} W={W} D={D} win={win}s | {len(pairs)} subjects", flush=True)

    curves = np.zeros((len(pairs), W), dtype=np.float32)
    for si, (fi, ei) in enumerate(pairs):
        ts = time.time()
        Gf = norm_gram(C.l2(fm_emb[0, :W, fi, :]).astype(np.float64))
        Ff = np.fft.rfft(diag_rows(Gf), axis=1)
        del Gf
        best = np.full(W, -np.inf)
        for l in range(L):
            Ge = norm_gram(C.l2(emb[l, :W, ei, :]).astype(np.float64))
            Fe = np.fft.rfft(diag_rows(Ge), axis=1)
            del Ge
            np.maximum(best, np.fft.irfft((Ff * np.conj(Fe)).sum(axis=0), n=W), out=best)
            del Fe
        curves[si] = best
        print(f"    {fm_subs[fi]}: d0={best[0]:.4f} best_other={best[1:].max():.4f} "
              f"({time.time()-ts:.0f}s)", flush=True)
        del Ff
    store[f"{key}__curves"] = curves
    store[f"{key}__subjects"] = np.array([fm_subs[i] for i, _ in pairs])
    print(f"[{key}] done ({time.time()-t0:.0f}s)\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    all_keys = [f"{m}-{s}" for m in C.EEG for s in C.EEG[m]["sizes"]]
    ap.add_argument("--models", nargs="+", default=all_keys, choices=all_keys)
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    print("── self-test (synthetic, no network) ──")
    if not self_test():
        raise SystemExit("self-test FAILED — refusing to run on real data.")
    if args.self_test_only:
        return
    C.HF_TOKEN_CACHE = _tok()

    wins = sorted({C.EEG_WINDOWS[k.rsplit("-", 1)[0]]["win_sec"] for k in args.models})
    print(f"\npooling fMRI at window lengths {wins}s ...")
    fmri_cache = {w: FW.fmri_embedding(w, shift_sec=0.0) for w in wins}

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "cyclic_cka.npz"
    store = dict(np.load(out, allow_pickle=True)) if out.exists() else {}
    print()
    for k in args.models:
        m, s = k.rsplit("-", 1)
        run_model(m, s, fmri_cache, store)
        store["keys"] = np.array(args.models)
        store["family"] = np.array([kk.rsplit("-", 1)[0] for kk in args.models])
        np.savez_compressed(out, **store)
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
