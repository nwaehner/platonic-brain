"""
circular_shift_vision_full.py — EVERY circular rotation, every EEG family x every vision model.

Replaces the step-10 sample with the exact permutation test. For each EEG family the
number of rotations is its own W (it varies: 1080 / 1350 / 1800 / 2160), so the null is
the complete set of W-1 non-identity rotations rather than 82 sampled ones, and the
p-value floor drops from 2/84 = 0.024 to 2/(W-199+2).

This is only affordable because of `fast_shift`: for a fixed layer pair, mKNN as a
function of the shift is a circular cross-correlation, so ONE FFT yields the whole curve.
The max over layer pairs is still taken independently at every shift — nothing is
skipped, the winning pair may differ from rotation to rotation, exactly as in the direct
implementation (verified to 1e-8 in bench_neurolm_dinov2.py, 188x faster there).

Outputs, one subfolder per EEG family:

    outputs_vs_vision/<model>/circular_shift_full__<model>.npz
    outputs_vs_vision/<model>/shift_full__<model>__<arch>.png      (one per architecture)

Each detail figure is rows = EEG size, cols = vision size, showing the full-resolution
curve with dashed lines at the episode boundaries (every 1080 s, which is a different
number of windows in each family) and the exact p annotated.

Usage:
    python circular_shift_vision_full.py                    # all five families
    python circular_shift_vision_full.py --eeg-model neurolm reve
    python circular_shift_vision_full.py --replot           # figures only, from saved npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import circular_shift_null as CS  # noqa: E402
import fast_shift as FS  # noqa: E402
import _common as C  # noqa: E402

OUT = _HERE / "outputs_vs_vision"
ARCHS = ["dinov2", "videomae", "videomae_ft", "vjepa2"]
EPISODE_S = 1080.0
TAIL_MIN = 100


def knn_excl(z, k, excl):
    """k nearest neighbours, forbidding anything within +-excl windows in time.

    Same result as `_common.knn_1d_excl` but without materialising the full W x W
    similarity matrix and sorting it: ask sklearn for k + 2*excl + 1 neighbours, which is
    guaranteed to leave at least k survivors after dropping the banned band, then filter.
    Verified against the reference implementation before use.
    """
    from sklearn.neighbors import NearestNeighbors
    W = len(z)
    m = min(k + 2 * excl + 1, W)
    _, idx = NearestNeighbors(n_neighbors=m, metric="cosine",
                              algorithm="brute").fit(z).kneighbors(z)
    i = np.arange(W)[:, None]
    keep = np.abs(idx - i) > excl
    out = np.zeros((W, k), dtype=np.int32)
    for r in range(W):
        out[r] = idx[r][keep[r]][:k]
    return out


def precompute_knn_excl(emb, k, excl):
    """EEG (n_layers, W, S, D) -> (n_layers, S, W, k), temporally excluded."""
    n_layers, W, Ns, _ = emb.shape
    idx = np.zeros((n_layers, Ns, W, k), dtype=np.int32)
    for l in range(n_layers):
        for s in range(Ns):
            idx[l, s] = knn_excl(C.l2(emb[l, :, s, :]), k, excl)
    return idx


def precompute_knn_layers_excl(emb, k, excl):
    """Vision/LLM (n_layers, W, D) -> (n_layers, W, k), temporally excluded."""
    return np.stack([knn_excl(C.l2(emb[l]), k, excl) for l in range(emb.shape[0])])


def curves_all_shifts(nn_eeg, nn_vis, W, k):
    """(S,LE,W,k) x (LV,W,k) -> (S,W) uncalibrated max-over-layer-pairs mKNN at every d.

    The vision transform is built once and reused across subjects; the EEG transform is
    built per subject and released, which keeps the peak under ~1.5 GB even for the
    48-layer NeuroLM-XL against 40-layer DINOv2-giant.
    """
    S = nn_eeg.shape[0]
    Fv = FS.precompute_fft(nn_vis, W)
    out = np.empty((S, W), dtype=np.float64)
    for s in range(S):
        Fe = FS.precompute_fft(nn_eeg[s], W)
        out[s] = FS.max_mknn_all_shifts(Fe, Fv, W, k)
        del Fe
    del Fv
    return out


def signed_shifts(W):
    d = np.arange(W)
    return np.where(d <= W // 2, d, d - W)


def stats_for(curve, d_signed, tail_min=TAIL_MIN):
    """Subject-MEAN statistics — what every figure and p-value in this project uses."""
    m = curve.mean(axis=0)
    tail = np.abs(d_signed) >= tail_min
    t = m[tail]
    z = (m[0] - t.mean()) / t.std(ddof=1)
    p = (1 + int((t >= m[0]).sum()) + 1) / (int(tail.sum()) + 2)
    return float(z), float(p), int(tail.sum())


def stats_per_subject(curve, d_signed, tail_min=TAIL_MIN):
    """Same test run INDEPENDENTLY inside each subject: their own d=0 against their own
    far-tail null. Stored alongside the mean so the between-subject question can be asked
    later without recomputing anything — the analysis here still reports the mean only.

    Worth keeping because the two can disagree sharply: a subject-mean z of +2.89 has been
    seen to come from subjects [+1.18, -0.42, +4.09, +0.66, +0.04, +2.66], i.e. carried by
    two of six. Averaging six noisy replicates shrinks the null spread by ~sqrt(6), so the
    mean-based z is legitimately larger than a typical individual one — but whether the
    effect is present in most individuals is a separate question these arrays can answer.
    """
    tail = np.abs(d_signed) >= tail_min
    n = int(tail.sum())
    zs, ps = [], []
    for c in curve:
        t = c[tail]
        zs.append((c[0] - t.mean()) / t.std(ddof=1))
        ps.append((1 + int((t >= c[0]).sum()) + 1) / (n + 2))
    return np.array(zs, dtype=np.float64), np.array(ps, dtype=np.float64)


def plot_arch(model, arch, z, d_signed, out_path):
    """rows = EEG size, cols = vision size of this architecture."""
    W = C.EEG_WINDOWS[model]["n_windows"]
    ws = C.EEG_WINDOWS[model]["win_sec"]
    ep_win = int(round(EPISODE_S / ws))
    ep_lines = [n * ep_win for n in range(-(W // 2) // ep_win, W // 2 // ep_win + 1) if n]

    sizes = C.EEG[model]["sizes"]
    vsizes = C.VISION[arch]["sizes"]
    order = np.argsort(d_signed)
    ds = d_signed[order]

    fig, axes = plt.subplots(len(sizes), len(vsizes),
                             figsize=(4.6 * len(vsizes), 3.2 * len(sizes)), squeeze=False)
    for r, sz in enumerate(sizes):
        for c, vs in enumerate(vsizes):
            ax = axes[r][c]
            key = f"{sz}__{arch}-{vs}__curve"
            if key not in z.files:
                ax.axis("off")
                continue
            curve = z[key]
            m = curve.mean(axis=0)[order]
            for j, x in enumerate(ep_lines):
                ax.axvline(x, color="#b2182b", lw=1.0, ls="--", alpha=0.55, zorder=2,
                           label=f"episode boundary (every {ep_win} win = 1080 s)"
                           if j == 0 else None)
            ax.plot(ds, m, color="#4477aa", lw=0.6, alpha=0.85, zorder=3,
                    label=f"all {W} rotations")
            ax.plot([0], [m[ds == 0]], "*", ms=15, color="crimson",
                    mec="black", mew=0.6, zorder=8, label="d = 0")
            zz, pp, ntail = stats_for(curve, d_signed)
            ax.set_title(f"{C.DISPLAY[model]}-{sz.upper()} × {C.VISION[arch]['display']}-{vs}",
                         fontsize=9.5, fontweight="bold", pad=15)
            ax.text(0.5, 1.012, f"z = {zz:+.2f}   p = {pp:.4f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color="#b2182b" if pp <= 0.05 else "#666666")
            ax.set_xlim(-W // 2, W // 2)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)
            if r == len(sizes) - 1:
                ax.set_xlabel("circular shift d (windows)", fontsize=9)
            if c == 0:
                ax.set_ylabel("mKNN (uncal.)", fontsize=9)
    axes[0][0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    _, _, ntail = stats_for(z[f"{sizes[0]}__{arch}-{vsizes[0]}__curve"], d_signed)
    fig.suptitle(
        f"{C.DISPLAY[model]} × {C.VISION[arch]['display']} — every one of the {W} circular "
        f"rotations ({ws}s windows)\n"
        f"Dashed red = episode boundaries, every {ep_win} windows = 1080 s = 18 min.  "
        f"Exact permutation p over {ntail} valid rotations (|d| ≥ {TAIL_MIN}), "
        f"floor {2 / (ntail + 2):.4f}.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path.name}")


def run_model(model, replot=False, excl=0, out_root=None):
    W = C.EEG_WINDOWS[model]["n_windows"]
    k = C.K_MKNN_DEFAULT
    fam = C.EEG[model]["family"]
    sizes = C.EEG[model]["sizes"]
    root = Path(out_root) if out_root else OUT
    sub = root / model
    sub.mkdir(parents=True, exist_ok=True)
    npz_path = sub / f"circular_shift_full__{model}.npz"
    d_signed = signed_shifts(W)

    if not replot:
        ex = f"  [temporal exclusion +-{excl} windows = "
        ex += f"{excl * C.EEG_WINDOWS[model]['win_sec']}s]" if excl else ""
        print(f"\n=== {C.DISPLAY[model]}  W={W}  ({C.EEG_WINDOWS[model]['win_sec']}s windows) "
              f"— {W - 1} non-identity rotations ==={ex if excl else ''}")
        nn_vis = {}
        for arch in ARCHS:
            for vs in C.VISION[arch]["sizes"]:
                emb = CS.load_retry(C.VISION[arch]["path"](vs, fam))["embeddings"].astype(np.float32)
                nn_vis[f"{arch}-{vs}"] = (precompute_knn_layers_excl(emb, k, excl) if excl
                                          else C.precompute_knn_layers(emb, k))
                del emb
        print(f"  {len(nn_vis)} vision variants loaded")

        payload = {"shifts": d_signed, "excl": excl}
        for sz in sizes:
            emb = CS.load_retry(C.EEG[model]["fname"](sz))["embeddings"].astype(np.float32)
            nn_eeg = (precompute_knn_excl(emb, k, excl) if excl
                      else C.precompute_knn(emb, k)).transpose(1, 0, 2, 3)
            LE = nn_eeg.shape[1]
            del emb
            for key, nv in nn_vis.items():
                if nv.shape[1] != W:
                    print(f"    [skip] {key}: W mismatch")
                    continue
                t0 = time.perf_counter()
                curve = curves_all_shifts(nn_eeg, nv, W, k)
                zz, pp, _ = stats_for(curve, d_signed)
                zsub, psub = stats_per_subject(curve, d_signed)
                # (S, W) curves are the complete per-subject record; z/p per subject are
                # stored too so later analysis needs no recomputation.
                payload[f"{sz}__{key}__curve"] = curve
                payload[f"{sz}__{key}__z_subj"] = zsub
                payload[f"{sz}__{key}__p_subj"] = psub
                payload[f"{sz}__{key}__z_mean"] = zz
                payload[f"{sz}__{key}__p_mean"] = pp
                print(f"    {model}-{sz:6s} × {key:16s} LE={LE:2d} LV={nv.shape[0]:2d}  "
                      f"z={zz:+5.2f} p={pp:.4f}  | subj z: "
                      f"{' '.join(f'{v:+.1f}' for v in zsub)}  ({time.perf_counter() - t0:4.1f}s)")
            del nn_eeg
        np.savez_compressed(npz_path, **payload)
        print(f"  Saved {npz_path}")

    z = np.load(npz_path, allow_pickle=True)
    for arch in ARCHS:
        plot_arch(model, arch, z, d_signed, sub / f"shift_full__{model}__{arch}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-model", nargs="+", default=list(C.EEG.keys()),
                    choices=list(C.EEG.keys()))
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--excl", type=int, default=0,
                    help="forbid neighbours within +-this many windows (temporal exclusion)")
    ap.add_argument("--out-root", default=None,
                    help="write elsewhere than outputs_vs_vision/")
    args = ap.parse_args()

    if not args.replot and not FS.self_test():
        sys.exit("fast_shift self-test failed")
    C.HF_TOKEN_CACHE = C.resolve_hf_token()

    t0 = time.perf_counter()
    for m in args.eeg_model:
        run_model(m, replot=args.replot, excl=args.excl, out_root=args.out_root)
    print(f"\nTOTAL {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
