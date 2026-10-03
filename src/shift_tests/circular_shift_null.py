"""
circular_shift_null.py — CIRCULAR-shift null for NeuroLM × LLM-caption alignment.

Scope (deliberately narrow, this is a testbed): NeuroLM sizes only, LLM captions only,
UNCALIBRATED mKNN only. No permutation calibration, no block-permutation null, no other
EEG family. Those come later once the shift machinery is trusted.

WHY CIRCULAR INSTEAD OF TRUNCATING
----------------------------------
The existing shift diagnostics (notebooks/analyze_mknn_hf.ipynb §7d and
alignment_plots/_common.shift_mknn) pair EEG[i] with LLM[i-d] and then TRUNCATE to the
overlap region, recomputing kNN on the shortened arrays. Two problems:

  (1) W changes with d.  mKNN as implemented (`mknn_1d`) has chance level k/(W-1), so a
      shift that drops 50 windows silently raises the floor. Measured: 5/1349 = 0.003706
      at d=0 vs 5/1299 = 0.003849 at |d|=50 — about 10% of the effect sizes being tested,
      systematically favouring the shifted (null) pairings.
  (2) The within-modality geometry changes.  Shrinking the candidate pool changes each
      modality's OWN nearest neighbours, so d=0 is compared against nulls computed on a
      different point set.

A circular shift fixes both. LLM'[i] = LLM[(i-d) mod W] keeps all W windows at every d,
and rotating index labels is an isometry of the point set: each modality's neighbour
structure is exactly PRESERVED, merely relabelled. The only thing varying across the
curve is the cross-modal correspondence — which is precisely what a null should vary.

Cost: because rotation is a relabelling, kNN is computed ONCE per modality and every
shift is a cheap index remap (the same trick `_common.aristotelian_cross` already uses
for its permutation null, with perm = np.roll(arange(W), d)). 101 shifts therefore cost
~1 kNN build, not 101.

Caveat, stated plainly: rotation splices the end of Season 7 onto its beginning, so at
shift d exactly |d| of the W windows are paired across that seam. This is benign for
mKNN — it reads only embedding geometry, and the seam perturbs neither modality's
internal neighbour structure — but it is why the null is a genuine scramble there.

WHAT IT DOES NOT FIX
--------------------
A signed monotone drift across d (alignment declining from d=-500 to d=+500) would mean
the shifts are not exchangeable, and ranking d=0 inside a trending set is not a valid
permutation test. With constant W and preserved geometry the truncation explanation is
gone, so a drift that survives here is real (a lag, or nonstationarity across the
season). That is why this script reports TWO p-values:

  p_all   d=0 ranked against every shift in the grid          (the old convention)
  p_tail  d=0 ranked against only |d| >= --tail-min           (drift-robust)

plus the tail mean/sd and a z-score, so the flatness of the far tail is visible rather
than assumed. Both use the standard (1 + #{>= obs}) / (n + 1) convention, which counts
d=0 in its own null (Phipson & Smyth 2010).

Usage:
    python circular_shift_null.py --self-test          # offline correctness check, no HF
    python circular_shift_null.py                      # all 3 sizes x all available LLMs
    python circular_shift_null.py --sizes l --stems bloomz-7b1
    python circular_shift_null.py --shifts -500:501:10 --tail-min 100
    python circular_shift_null.py --also-truncate      # old method alongside, for contrast
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# The Xet transfer backend fails on multi-GB files over a flaky link
# ("CAS Client Error: ... error decoding response body"). The classic HTTP path is
# slower but resumable and far more reliable, which matters for the 2.5 GB NeuroLM-XL
# file. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the project's loaders / registries / kNN primitives.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alignment_plots"))
import _common as C  # noqa: E402


def load_retry(path: str, attempts: int = 4, base_wait: float = 5.0):
    """C.load_npz with backoff. Multi-GB HF downloads fail transiently often enough
    that one network hiccup should not discard an hour of completed cells."""
    for a in range(1, attempts + 1):
        try:
            return C.load_npz(path)
        except Exception as e:
            if a == attempts:
                raise
            wait = base_wait * (2 ** (a - 1))
            print(f"    [retry {a}/{attempts - 1}] {path}: {type(e).__name__}: "
                  f"{str(e)[:100]} — waiting {wait:.0f}s")
            time.sleep(wait)

OUT = Path(__file__).resolve().parent / "outputs_vs_llm"

# The complete LLM-caption set for every EEG grid lives in triniborrell's dataset;
# nitrox639 only ever had llms/neurolm/. _common.fetch routes llms/ to C.LLM_REPO.
DEFAULT_LLM_REPO = "triniborrell/platonic-embeddings"


# ── circular-shift primitives ─────────────────────────────────────────────────
def roll_maps(W: int, d: int):
    """Index maps for LLM'[i] = LLM[(i - d) mod W].

    perm[i] = (i - d) % W   so  emb[perm] is the rotated embedding matrix.
    inv     = argsort(perm) maps an OLD index to its NEW position.
    """
    perm = np.roll(np.arange(W), d)
    inv = np.argsort(perm)
    return perm, inv


def relabel_nn(nn: np.ndarray, perm: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """Neighbour sets of the rotated point set, WITHOUT recomputing kNN.

    The point now sitting at index i is old point perm[i]; its neighbours are
    nn[perm[i]] in old labels, which become inv[...] in new labels. Distances are
    unchanged by relabelling, so ordering is preserved too.
      nn : (L, W, k) -> (L, W, k)
    """
    return inv[nn[:, perm, :]]


def mknn_matrix(nn_a: np.ndarray, nn_b: np.ndarray) -> np.ndarray:
    """mKNN for every (a-layer, b-layer) pair. nn_a (LA,W,k), nn_b (LB,W,k) -> (LA,LB).

    Identical semantics to `_common.mknn_1d` (mean over the W*k neighbour slots), just
    evaluated for the whole layer grid at once.
    """
    LA = nn_a.shape[0]
    LB = nn_b.shape[0]
    out = np.empty((LA, LB), dtype=np.float64)
    for la in range(LA):
        hit = (nn_a[la][None, :, :, None] == nn_b[:, :, None, :]).any(-1)  # (LB,W,k)
        out[la] = hit.mean(axis=(1, 2))
    return out


def circular_curve(nn_eeg: np.ndarray, nn_llm: np.ndarray, shifts) -> np.ndarray:
    """Uncalibrated max-over-layer-pairs mKNN at every circular shift.

      nn_eeg : (S, LE, W, k)   per-subject EEG neighbour sets (computed once)
      nn_llm : (LV, W, k)      LLM neighbour sets (computed once)
    Returns (S, n_shifts).

    The layer pair is re-maximised at every d — the null gets the same optimisation
    advantage as d=0, so the comparison stays fair. Fixing the pair at d=0's argmax
    would hand d=0 a selection bias.
    """
    S, _, W, _ = nn_eeg.shape
    out = np.empty((S, len(shifts)), dtype=np.float64)
    for di, d in enumerate(shifts):
        perm, inv = roll_maps(W, int(d))
        nn_d = relabel_nn(nn_llm, perm, inv)
        for s in range(S):
            out[s, di] = mknn_matrix(nn_eeg[s], nn_d).max()
    return out


def truncating_curve(eeg_emb: np.ndarray, llm_emb: np.ndarray, k: int, shifts) -> np.ndarray:
    """The OLD method, kept for contrast: truncate to the overlap and rebuild kNN.
    eeg_emb (LE,W,S,D), llm_emb (LV,W,D) -> (S, n_shifts). Expensive."""
    S = eeg_emb.shape[2]
    out = np.empty((S, len(shifts)), dtype=np.float64)
    for di, d in enumerate(shifts):
        for s in range(S):
            out[s, di] = C.shift_mknn(eeg_emb[:, :, s, :], llm_emb, k, int(d))
    return out


# ── statistics ────────────────────────────────────────────────────────────────
def shift_pvalues(shifts: np.ndarray, curve_mean: np.ndarray, tail_min: int) -> dict:
    """p_all over the whole grid, p_tail over |d| >= tail_min. Both include d=0 in the
    null (the +1/+1 convention), so the floor is 1/(n+1)."""
    shifts = np.asarray(shifts)
    obs = float(curve_mean[shifts == 0][0])

    p_all = float((1 + (curve_mean >= obs).sum()) / (len(curve_mean) + 1))

    tail = np.abs(shifts) >= tail_min
    tail_vals = curve_mean[tail]
    n_tail = int(tail.sum())
    p_tail = float((1 + (tail_vals >= obs).sum() + 1) / (n_tail + 2)) if n_tail else np.nan
    # (+1 in numerator/denominator for d=0 itself, which is excluded from `tail`.)

    tail_mu = float(tail_vals.mean()) if n_tail else np.nan
    tail_sd = float(tail_vals.std(ddof=1)) if n_tail > 1 else np.nan
    z_tail = (obs - tail_mu) / tail_sd if n_tail > 1 and tail_sd > 0 else np.nan

    # Signed drift: slope of the curve vs d, and Spearman-free monotonicity proxy.
    slope = float(np.polyfit(shifts, curve_mean, 1)[0])

    return dict(obs=obs, p_all=p_all, p_tail=p_tail, n_tail=n_tail,
                tail_mean=tail_mu, tail_sd=tail_sd, z_tail=float(z_tail), slope=slope)


# ── offline self-test ─────────────────────────────────────────────────────────
def self_test() -> bool:
    """Verify the relabelling identity and the vectorised mKNN against brute force.
    Pure synthetic data — runs with no HuggingFace access."""
    print("── self-test (synthetic, no network) ──")
    rng = np.random.default_rng(0)
    L, W, D, k = 3, 200, 16, 5
    emb = rng.normal(size=(L, W, D)).astype(np.float32)
    nn_orig = np.stack([C.knn_1d(C.l2(emb[l]), k) for l in range(L)])

    ok = True
    for d in (0, 1, 7, -3, 97, -150):
        perm, inv = roll_maps(W, d)
        # ground truth: actually rotate the embeddings and recompute kNN
        rolled = emb[:, perm, :]
        nn_direct = np.stack([C.knn_1d(C.l2(rolled[l]), k) for l in range(L)])
        nn_relab = relabel_nn(nn_orig, perm, inv)
        same = all(set(nn_direct[l, i]) == set(nn_relab[l, i])
                   for l in range(L) for i in range(W))
        print(f"  d={d:+5d}  relabel == recomputed kNN : {'OK' if same else 'MISMATCH'}")
        ok &= same

    # rotation must preserve the number of windows
    perm, _ = roll_maps(W, 37)
    ok &= len(np.unique(perm)) == W
    print(f"  rotation is a bijection over W={W} : {'OK' if len(np.unique(perm)) == W else 'FAIL'}")

    # vectorised mKNN grid == the project's scalar primitive
    emb_b = rng.normal(size=(4, W, D)).astype(np.float32)
    nn_b = np.stack([C.knn_1d(C.l2(emb_b[l]), k) for l in range(4)])
    M = mknn_matrix(nn_orig, nn_b)
    ref = np.array([[C.mknn_1d(nn_orig[a], nn_b[b]) for b in range(4)] for a in range(L)])
    close = np.allclose(M, ref)
    print(f"  mknn_matrix == _common.mknn_1d grid : {'OK' if close else 'MISMATCH'}")
    ok &= close

    mx = M.max()
    ref_mx = C.max_mknn_over_layers(nn_orig, nn_b)
    print(f"  max == _common.max_mknn_over_layers : "
          f"{'OK' if np.isclose(mx, ref_mx) else 'MISMATCH'}  ({mx:.6f} vs {ref_mx:.6f})")
    ok &= np.isclose(mx, ref_mx)

    # d=0 must reproduce the unshifted score exactly
    c0 = circular_curve(nn_orig[None], nn_b, [0])[0, 0]
    print(f"  circular d=0 == unshifted mKNN      : "
          f"{'OK' if np.isclose(c0, ref_mx) else 'MISMATCH'}")
    ok &= np.isclose(c0, ref_mx)

    print(f"── self-test {'PASSED' if ok else 'FAILED'} ──\n")
    return bool(ok)


# ── plotting ──────────────────────────────────────────────────────────────────
def _stem_colors(stems):
    bloom = [s for s in stems if s.startswith("bloomz")]
    other = [s for s in stems if not s.startswith("bloomz")]
    col = {}
    for i, s in enumerate(bloom):
        col[s] = plt.cm.Blues(0.35 + 0.6 * i / max(1, len(bloom) - 1))
    for i, s in enumerate(other):
        col[s] = plt.cm.Oranges(0.35 + 0.6 * i / max(1, len(other) - 1))
    return col


def plot_curves(shifts, results, sizes, stems, tail_min, path, model="neurolm",
                xlim=None, title_extra=""):
    col = _stem_colors(stems)
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.2 * len(sizes), 5.0), squeeze=False)
    for ax, sz in zip(axes[0], sizes):
        for st in stems:
            key = (sz, st)
            if key not in results:
                continue
            m = results[key]["curve"].mean(axis=0)
            ax.plot(shifts, m, color=col[st], lw=1.4, label=st, zorder=3)
            i0 = int(np.where(shifts == 0)[0][0])
            ax.plot([0], [m[i0]], marker="*", ms=13, color=col[st],
                    mec="crimson", mew=1.2, zorder=6)
        ax.axvline(0, color="crimson", lw=0.8, ls="--", alpha=0.6, zorder=1)
        for sgn in (-1, 1):
            ax.axvline(sgn * tail_min, color="#999999", lw=0.7, ls=":", zorder=1)
        ax.set_title(f"{C.DISPLAY[model]}-{sz.upper()} ({C.PARAMS[model][sz]}M)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("circular shift d (windows)", fontsize=9)
        ax.set_ylabel("mKNN (uncalibrated, max over layer pairs)", fontsize=9)
        ax.grid(alpha=0.25)
        if xlim:
            ax.set_xlim(*xlim)
    axes[0][-1].legend(fontsize=7, ncol=1, loc="upper right", framealpha=0.9)
    fig.suptitle(
        f"Circular-shift null — {C.DISPLAY[model]} × LLM captions "
        f"(k=5, uncalibrated){title_extra}\n"
        f"star = d=0 (true pairing) · dotted = |d| = {tail_min} far-tail boundary · "
        f"W constant at every shift",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eeg-model", default="neurolm", choices=list(C.EEG.keys()))
    ap.add_argument("--sizes", nargs="+", default=None,
                    help="sizes of --eeg-model (default: all of them)")
    ap.add_argument("--stems", nargs="+", default=None,
                    help="LLM stems (default: every bloom + openllama stem available). "
                         "llama-13b is excluded by default: it is the only llama size on "
                         "HF, so it adds an isolated point with no within-family scaling.")
    ap.add_argument("--llm-repo", default=DEFAULT_LLM_REPO,
                    help="HF dataset holding llms/<eeg_model>/")
    ap.add_argument("--llm-grid", default=None,
                    help="read llms/<this>/ instead of llms/<eeg_model>/. FEMBA and LUNA "
                         "share the same 5s/2160-window tiling and their caption files are "
                         "byte-identical (verified by sha256), so `--eeg-model luna "
                         "--llm-grid femba` reuses the cache instead of re-downloading.")
    ap.add_argument("--shifts", default="-500:501:10", help="start:stop:step")
    ap.add_argument("--tail-min", type=int, default=100,
                    help="|d| >= this defines the drift-robust far tail")
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--also-truncate", action="store_true",
                    help="additionally run the old truncating method (slow)")
    ap.add_argument("--self-test", action="store_true", help="offline checks, then exit")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not self_test():
        sys.exit("self-test failed — refusing to run on real data")

    a, b, st = (int(v) for v in args.shifts.split(":"))
    shifts = np.array(sorted(set(list(range(a, b, st)) + [0])))
    stems = args.stems or (C.LLM["bloom"] + C.LLM["openllama"])

    model = args.eeg_model
    llm_grid = args.llm_grid or model
    if C.EEG_WINDOWS[llm_grid]["n_windows"] != C.EEG_WINDOWS[model]["n_windows"]:
        sys.exit(f"--llm-grid {llm_grid} has W="
                 f"{C.EEG_WINDOWS[llm_grid]['n_windows']} but {model} has W="
                 f"{C.EEG_WINDOWS[model]['n_windows']} — grids must match.")
    all_sizes = C.EEG[model]["sizes"]
    sizes = args.sizes or all_sizes
    bad = [s for s in sizes if s not in all_sizes]
    if bad:
        sys.exit(f"unknown size(s) {bad} for {model}; valid: {all_sizes}")

    C.LLM_REPO = args.llm_repo          # route llms/ reads (fetch reads this at call time)
    C.HF_TOKEN_CACHE = C.resolve_hf_token(args.hf_token)
    if not C.HF_TOKEN_CACHE:
        sys.exit("No HuggingFace token. Set HF_TOKEN, write tokens/hf_token.txt, "
                 "or pass --hf-token.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W_expect = C.EEG_WINDOWS[model]["n_windows"]
    if np.abs(shifts).max() >= W_expect // 2:
        print(f"WARN: |d|max={np.abs(shifts).max()} >= W/2={W_expect // 2}; circular shifts "
              f"have period W={W_expect}, so far shifts start mirroring each other.")

    print(f"\nEEG model     : {model} ({C.EEG_WINDOWS[model]['win_sec']}s windows, "
          f"W={W_expect})")
    print(f"sizes         : {sizes}")
    print(f"LLM stems     : {stems}   (repo: {C.LLM_REPO})")
    print(f"shifts        : {len(shifts)} values, {shifts.min()}..{shifts.max()} step {st}")
    print(f"k             : {args.k_mknn}   (uncalibrated mKNN only)\n")

    # LLM neighbour sets — computed once per stem, reused for every size and shift.
    nn_llm, llm_emb = {}, {}
    for stem in stems:
        try:
            t0 = time.time()
            emb = load_retry(C.llm_path(llm_grid, stem))["embeddings"].astype(np.float32)
        except Exception as e:
            print(f"  [skip] {stem}: {type(e).__name__}: {str(e)[:120]}")
            continue
        nn_llm[stem] = C.precompute_knn_layers(emb, args.k_mknn)
        print(f"  llm  {stem:16s} {emb.shape}  kNN in {time.time() - t0:.1f}s")
        if args.also_truncate:
            llm_emb[stem] = emb
        else:
            del emb
    if not nn_llm:
        sys.exit("No LLM embeddings could be loaded.")
    stems = [s for s in stems if s in nn_llm]

    results, payload = {}, {"shifts": shifts}
    npz_path = out_dir / f"circular_shift_null__{model}.npz"

    def checkpoint():
        """Persist after every cell. A multi-hour sweep over a flaky link must never
        discard completed work because a later download failed."""
        np.savez_compressed(npz_path, **payload)

    for sz in sizes:
        t0 = time.time()
        try:
            emb = load_retry(C.EEG[model]["fname"](sz))["embeddings"].astype(np.float32)
        except Exception as e:
            print(f"  [skip] {model}-{sz}: {type(e).__name__}: {str(e)[:140]}")
            continue
        LE, W, S, _ = emb.shape
        nn_eeg = C.precompute_knn(emb, args.k_mknn).transpose(1, 0, 2, 3)  # (S,LE,W,k)
        print(f"\n  eeg  {model}-{sz:6s} {emb.shape}  kNN in {time.time() - t0:.1f}s")
        eeg_emb = emb if args.also_truncate else None
        del emb

        for stem in stems:
            assert nn_llm[stem].shape[1] == W, \
                f"window mismatch: eeg W={W} vs {stem} W={nn_llm[stem].shape[1]}"
            t1 = time.time()
            curve = circular_curve(nn_eeg, nn_llm[stem], shifts)
            stats = shift_pvalues(shifts, curve.mean(axis=0), args.tail_min)
            results[(sz, stem)] = dict(curve=curve, **stats)
            payload[f"{sz}__{stem}__curve"] = curve
            for kk, vv in stats.items():
                payload[f"{sz}__{stem}__{kk}"] = vv
            print(f"    {stem:16s} d0={stats['obs']:.5f}  p_all={stats['p_all']:.3f}  "
                  f"p_tail={stats['p_tail']:.3f}  z_tail={stats['z_tail']:+.2f}  "
                  f"slope={stats['slope']:+.2e}  ({time.time() - t1:.1f}s)")
            checkpoint()

            if args.also_truncate:
                t2 = time.time()
                tc = truncating_curve(eeg_emb, llm_emb[stem], args.k_mknn, shifts)
                ts = shift_pvalues(shifts, tc.mean(axis=0), args.tail_min)
                payload[f"{sz}__{stem}__trunc_curve"] = tc
                print(f"      [truncating] d0={ts['obs']:.5f}  p_all={ts['p_all']:.3f}  "
                      f"({time.time() - t2:.1f}s)")

    checkpoint()
    print(f"\n  Saved {npz_path}")
    if not results:
        sys.exit("No cells completed — nothing to plot.")

    sizes_done = [s for s in sizes if any(k[0] == s for k in results)]
    plot_curves(shifts, results, sizes_done, stems, args.tail_min,
                out_dir / f"circular_shift__{model}_llm.png", model=model)
    near = min(200, int(np.abs(shifts).max()))
    plot_curves(shifts, results, sizes_done, stems, args.tail_min,
                out_dir / f"circular_shift__{model}_llm__zoom.png", model=model,
                xlim=(-near, near), title_extra="  [zoom near d=0]")

    print("\n── summary (subject-mean, uncalibrated) ──")
    print(f"{'size':5s} {'stem':16s} {'d=0':>8s} {'tail mu':>8s} {'tail sd':>8s} "
          f"{'z':>6s} {'p_all':>6s} {'p_tail':>7s}")
    for sz in sizes_done:
        for stem in stems:
            r = results.get((sz, stem))
            if r:
                print(f"{sz:5s} {stem:16s} {r['obs']:8.5f} {r['tail_mean']:8.5f} "
                      f"{r['tail_sd']:8.5f} {r['z_tail']:+6.2f} {r['p_all']:6.3f} "
                      f"{r['p_tail']:7.3f}")


if __name__ == "__main__":
    main()
