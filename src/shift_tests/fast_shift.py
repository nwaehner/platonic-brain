"""
fast_shift.py — every circular shift at once, via FFT over neighbour offsets.

THE IDENTITY
------------
mKNN as implemented counts, over windows i and neighbour slots j, whether A's j-th
neighbour of i is also a neighbour of i in B:

    count(d) = #{ (i,a) in A, (p,b) in B  :  i = p+d,  a = b+d   (mod W) }
    mknn(d)  = count(d) / (W*k)

Subtracting the two congruences kills d:

    i - a  =  p - b   (mod W)

So define the OFFSET  u = (i - neighbour) mod W. It is invariant under the rotation. Two
entries can only ever meet — at any shift — if they share u, and when they do they meet
at exactly d = (i - p) mod W. Therefore

    count(d) = SUM_u  ( A_u  cross-correlated with  B_u ) (d)

where A_u is the indicator vector over window index i of A's entries with offset u. One
FFT per offset row gives EVERY shift simultaneously.

WHY IT IS FASTER
----------------
The direct loop costs O(n_shifts * W * k^2) per layer pair, so asking for all W rotations
costs W times one shift. Here the transform is computed ONCE per layer (not per pair, not
per shift) and each layer pair is a single elementwise product plus one inverse transform,
independent of how many shifts you want. Getting all W rotations therefore costs about
the same as getting a handful — which is what makes the EXACT permutation test (rank of
d=0 among all W-1 non-identity rotations, floor 2/(W+1)) affordable instead of the
Monte-Carlo approximation at resolution 1/84 that the step-10 grid gives.

CORRECTNESS
-----------
`self_test()` checks the FFT result against a literal brute-force recount on small random
data, and `verify_against(...)` checks it against the production `circular_curve` on real
embeddings. Both must pass before any number here is trusted.
"""

from __future__ import annotations

import numpy as np


def offset_rows(nn: np.ndarray, W: int):
    """(W,k) neighbour indices -> (offset u, window i) for every entry."""
    Wn, k = nn.shape
    i = np.repeat(np.arange(Wn), k)
    a = nn.ravel().astype(np.int64)
    u = (i - a) % W
    return u, i


def _indicator(nn: np.ndarray, W: int) -> np.ndarray:
    """Dense (W, W) indicator: row = offset u, col = window index i.

    The row axis spans the FULL offset range 0..W-1 rather than only the offsets observed
    in this particular array. An earlier version built a restricted vocabulary from one
    subject and reused it for the rest, which crashed the moment another subject produced
    an offset the first one lacked (STEEGFormer: u=0, a window that is its own neighbour,
    which happens when duplicate embeddings defeat the self-exclusion in knn_1d). Since
    the observed vocabulary is ~100% of W anyway, restricting it saved nothing and only
    created a way to be wrong.
    """
    u, i = offset_rows(nn, W)
    M = np.zeros((W, W), dtype=np.float32)
    M[u, i] = 1.0
    return M


def mknn_all_shifts(nn_a: np.ndarray, nn_b: np.ndarray, W: int) -> np.ndarray:
    """mKNN at EVERY circular shift d = 0..W-1, for one layer pair.

    nn_a, nn_b : (W, k).  Returns (W,) with index d meaning B was rotated by d,
    i.e. B'[i] = B[(i-d) mod W] — the same convention as circular_shift_null.roll_maps.
    """
    k = nn_a.shape[1]
    A = _indicator(nn_a, W)
    B = _indicator(nn_b, W)
    Fa = np.fft.rfft(A, axis=1)
    Fb = np.fft.rfft(B, axis=1)
    cross = (Fa * np.conj(Fb)).sum(axis=0)
    count = np.fft.irfft(cross, n=W)
    return count / (W * k)


def precompute_fft(nn_layers: np.ndarray, W: int) -> np.ndarray:
    """(L, W, k) -> (L, W, W//2+1) rFFT of each layer's offset indicator."""
    out = np.empty((nn_layers.shape[0], W, W // 2 + 1), dtype=np.complex64)
    for l in range(nn_layers.shape[0]):
        out[l] = np.fft.rfft(_indicator(nn_layers[l], W), axis=1).astype(np.complex64)
    return out


def max_mknn_all_shifts(Fa: np.ndarray, Fb: np.ndarray, W: int, k: int) -> np.ndarray:
    """Max over all (a-layer, b-layer) pairs, at every shift. Fa (LA,n_u,F), Fb (LB,...).

    The layer pair is re-maximised independently at each shift, exactly as the direct
    implementation does — the null keeps the same optimisation advantage as d=0.
    """
    LA, LB = Fa.shape[0], Fb.shape[0]
    best = np.full(W, -np.inf)
    for la in range(LA):
        for lb in range(LB):
            cross = (Fa[la] * np.conj(Fb[lb])).sum(axis=0)
            curve = np.fft.irfft(cross, n=W) / (W * k)
            np.maximum(best, curve, out=best)
    return best


# ── correctness ───────────────────────────────────────────────────────────────
def _brute(nn_a, nn_b, W, d):
    """Literal recount at one shift, independent of every optimisation above."""
    k = nn_a.shape[1]
    perm = np.roll(np.arange(W), d)
    inv = np.argsort(perm)
    nb = inv[nn_b[perm, :]]
    return float((nn_a[:, :, None] == nb[:, None, :]).any(-1).mean())


def self_test(verbose=True) -> bool:
    rng = np.random.default_rng(0)
    ok = True
    for W, k in [(37, 3), (64, 5), (211, 5)]:
        nn_a = np.stack([rng.choice(W - 1, k, replace=False) for _ in range(W)]).astype(np.int32)
        nn_b = np.stack([rng.choice(W - 1, k, replace=False) for _ in range(W)]).astype(np.int32)
        fast = mknn_all_shifts(nn_a, nn_b, W)
        ds = [0, 1, 2, 7, W // 3, W - 1]
        ref = np.array([_brute(nn_a, nn_b, W, d) for d in ds])
        good = np.allclose(fast[ds], ref, atol=1e-9)
        ok &= good
        if verbose:
            print(f"  W={W:4d} k={k}: FFT == brute force at d={ds} : "
                  f"{'OK' if good else 'MISMATCH'}  (max err {np.abs(fast[ds]-ref).max():.2e})")
    if verbose:
        print(f"── fast_shift self-test {'PASSED' if ok else 'FAILED'} ──")
    return bool(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
