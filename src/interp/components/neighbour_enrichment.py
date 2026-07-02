"""
neighbour_enrichment.py — Component 2: what mediates the neighbourhood (variance ratio).

EVERY window is a query (no anchors). A regime = a PAIR of model instances (A, B) on a common
grid — either adjacent sizes of one architecture (intramodal) or two different modalities
(cross-modal). For each window w the SHARED-neighbour set S_w = kNN_A(w) ∩ kNN_B(w) at k=10.
For each feature f and each window with |S_w|≥2:

    ratio_w(f) = var(f over S_w) / var(f over ALL windows)

ratio<1 ⇒ the shared neighbours are tighter on f than the grid as a whole ⇒ f is the content
that locally co-clusters / mediates the alignment. We test, per feature, whether the median
ratio is significantly below a matched-n permutation null (BH-FDR). The signed mean-shift used
before is dropped — it cancels when every window is a query.

  Regime        Grid          Embeddings (see plan)                    Pair
  eeg_only      EEG native    nitrox639/eeg                            adjacent EEG sizes
  vision_only   4 s (clip4s)  triniborrell/vision …__clip4s            adjacent vision sizes
  llm_only      4 s (clip4s)  triniborrell/llms/clip4s                 adjacent LLM stems
  intersection_eeg_vision   EEG native   eeg + nitrox639/vision …__<eeg_family>
  intersection_eeg_llm      EEG native   eeg + triniborrell/llms/<eeg_model>
  intersection_llm_vision   4 s (clip4s) triniborrell clip4s vision + llm

Output per regime per pair: outputs/enrichment/<grid>/varratio_<regime>_<label>.png
(three subplots — low / CLIP / semantic — one step-histogram of ratio_w per feature, line at 1)
and enrichment/enrich_<regime>.npz (median_ratio, p, q per feature, mKNN).

Usage:
  python src/interp/neighbour_enrichment.py --smoke
  python src/interp/neighbour_enrichment.py --regimes intersection_llm_vision --k 10
"""

from __future__ import annotations

# bootstrap: add interp root + all study subdirs to sys.path
import sys as _sys; from pathlib import Path as _Path
_INTERP = _Path(__file__).resolve().parent.parent
for _d in ([_INTERP] + [p for p in _INTERP.iterdir() if p.is_dir() and p.name[0] not in "._o"]):
    if str(_d) not in _sys.path: _sys.path.insert(0, str(_d))

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import _interp_common as IC
import _common as C
import interp_features as F

REGIMES = ["eeg_only", "vision_only", "llm_only",
           "intersection_eeg_vision", "intersection_eeg_llm", "intersection_llm_vision"]
TIER_KEYS = ["low", "clip", "sem"]
TIER_LABEL = {"low": "low-level visual", "clip": "high-level visual (CLIP)", "sem": "semantic"}


def _clean(M):
    M = np.asarray(M, dtype=np.float64).copy()
    if M.ndim == 1:
        M = M[:, None]
    cm_ = np.nanmean(M, axis=0)
    cm_[~np.isfinite(cm_)] = 0.0
    inds = np.where(np.isnan(M))
    M[inds] = np.take(cm_, inds[1])
    return M


def _tiers_of(names):
    return np.array([str(n).split(":")[0] for n in names])


# ── embeddings → layerwise kNN on a grid ─────────────────────────────────────────
def _knn(modality, ident, grid, family, k, W):
    if modality == "eeg":
        emb = IC.load_eeg_emb(ident[0], ident[1])
    elif modality == "vision":
        emb = IC.load_vision_grid(ident[0], ident[1], family)
    else:  # llm
        emb = IC.load_llm_grid(grid, ident)
    if emb is None:
        return None
    return C.precompute_knn_layers(emb[:, :W], k)


def shared_neighbours(specA, specB, k, W):
    """Per-window shared-neighbour sets (intersection of the two models' kNN at their best
    layer pair) over ALL windows. specA/specB = (modality, ident, grid, family). Returns
    (list of index arrays length W, mKNN score) or (None, None)."""
    nnA = _knn(*specA, k=k, W=W)
    nnB = _knn(*specB, k=k, W=W)
    if nnA is None or nnB is None:
        return None, None
    lA, lB, score = IC.best_layer_pair(nnA, nnB)
    a, b = nnA[lA], nnB[lB]
    return [np.intersect1d(a[w], b[w]) for w in range(W)], float(score)


# ── regime → list of (grid, label, specA, specB) pairs ───────────────────────────
def _eeg_archs(smoke):
    return {"reve": ["base", "large"]} if smoke else {m: C.EEG[m]["sizes"] for m in C.EEG}


def _vision_archs(smoke):
    return {"dinov2": ["small", "base"]} if smoke else {a: C.VISION[a]["sizes"] for a in C.VISION}


def _llm_stems(smoke):
    return {"bloom": ["bloomz-560m", "bloomz-1b1"]} if smoke else {f: s for f, s in C.LLM.items()}


def _adj(seq):
    return list(zip(seq[:-1], seq[1:]))                  # adjacent-size pairs


def iter_pairs(regime, smoke):
    """Yield (grid, label, specA, specB) for every pair in a regime."""
    if regime == "eeg_only":
        for arch, sizes in _eeg_archs(smoke).items():
            for s0, s1 in _adj(sizes):
                yield (arch, f"{arch}-{s0}__{arch}-{s1}",
                       ("eeg", (arch, s0), arch, None), ("eeg", (arch, s1), arch, None))
    elif regime == "vision_only":
        for arch, sizes in _vision_archs(smoke).items():
            for s0, s1 in _adj(sizes):
                yield ("clip4s", f"{arch}-{s0}__{arch}-{s1}",
                       ("vision", (arch, s0), "clip4s", "clip4s"),
                       ("vision", (arch, s1), "clip4s", "clip4s"))
    elif regime == "llm_only":
        for fam, stems in _llm_stems(smoke).items():
            for s0, s1 in _adj(stems):
                yield ("clip4s", f"{s0}__{s1}",
                       ("llm", s0, "clip4s", None), ("llm", s1, "clip4s", None))
    elif regime == "intersection_eeg_vision":
        for em, esz in _eeg_archs(smoke).items():
            fam = C.EEG[em]["family"]
            for va, vsz in _vision_archs(smoke).items():
                yield (em, f"{em}-{esz[-1]}__{va}-{vsz[-1]}",
                       ("eeg", (em, esz[-1]), em, None),
                       ("vision", (va, vsz[-1]), em, fam))
    elif regime == "intersection_eeg_llm":
        for em, esz in _eeg_archs(smoke).items():
            for fam, stems in _llm_stems(smoke).items():
                yield (em, f"{em}-{esz[-1]}__{stems[-1]}",
                       ("eeg", (em, esz[-1]), em, None), ("llm", stems[-1], em, None))
    else:  # intersection_llm_vision
        for va, vsz in _vision_archs(smoke).items():
            for fam, stems in _llm_stems(smoke).items():
                yield ("clip4s", f"{va}-{vsz[-1]}__{stems[-1]}",
                       ("vision", (va, vsz[-1]), "clip4s", "clip4s"),
                       ("llm", stems[-1], "clip4s", None))


# ── plot ─────────────────────────────────────────────────────────────────────────
def plot_varratio(res, names, title, path, max_legend=15):
    """Three subplots (low / CLIP / semantic): one step-histogram of ratio_w per feature
    (colour by median ratio), vertical line at ratio=1; q<0.05 solid, else dashed."""
    ratios, med, q = res["ratios"], res["median_ratio"], res["q"]
    tiers = _tiers_of(names)
    keys = [t for t in TIER_KEYS if (tiers == t).any()]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.2), squeeze=False)
    bins = np.linspace(0, 2, 31)
    for ax, t in zip(axes[0], keys):
        idx = np.where(tiers == t)[0]
        order = idx[np.argsort(med[idx])]                # most-localised first
        colors = cm.viridis(np.linspace(0, 1, len(order)))
        for rank, (fi, col) in enumerate(zip(order, colors)):
            vals = ratios[:, fi]
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                continue
            sig = np.isfinite(q[fi]) and q[fi] < 0.05
            lbl = f"{names[fi].split(':', 1)[-1]} ({med[fi]:.2f})" if rank < max_legend else None
            ax.hist(np.clip(vals, 0, 2), bins=bins, density=True, histtype="step",
                    color=col, lw=1.6 if sig else 0.8, ls="-" if sig else "--", label=lbl)
        ax.axvline(1, color="k", lw=1.2)
        ax.set_xlabel("var(neighbours) / var(all)")
        ax.set_title(TIER_LABEL[t], fontsize=9)
        ax.legend(fontsize=5, loc="upper right", title="feature (median)")
    fig.suptitle(title + "   (solid = q<0.05; colour = median ratio, dark→small)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    IC.savefig(fig, path)


# ── driver ────────────────────────────────────────────────────────────────────────
def run_regime(regime, smoke, k, R):
    out = {}
    for grid, label, specA, specB in iter_pairs(regime, smoke):
        feats = F.load_features(grid, smoke)
        if feats is None:
            print(f"  [skip] no feature cache for grid '{grid}' — build Comp 1 first")
            continue
        W = int(feats["W"]); feat = _clean(feats["feat"]); names = feats["feat_names"]
        sets, score = shared_neighbours(specA, specB, k, W)
        if sets is None:
            continue
        res = IC.varratio_test(feat, sets, R=R)
        if res is None:
            print(f"  [skip] {label}: <3 windows with ≥2 shared neighbours (k={k})")
            continue
        res["mKNN"] = score; res["feature_names"] = names
        ttl = f"{regime}: {label}  (mKNN={score:.3f}, {res['n_used']} usable windows)"
        plot_varratio(res, names, ttl, f"enrichment/{grid}/varratio_{regime}_{label}.png")
        out[label] = res
        cand = np.where(res["q"] < 0.05)[0]                # prefer significant features
        pool = cand if len(cand) else np.arange(len(names))
        i = int(pool[np.nanargmin(res["median_ratio"][pool])])
        print(f"  {regime} {label}: {res['n_used']} usable win · {len(cand)} feat q<0.05 · "
              f"most-localised{' sig' if len(cand) else ''} {names[i]}="
              f"{res['median_ratio'][i]:.2f}")
    return out


def save_regime(regime, out):
    if not out:
        return
    flat = {"feature_names": next(iter(out.values()))["feature_names"]}
    for label, r in out.items():
        for key in ["median_ratio", "p", "q"]:
            flat[f"{label}__{key}"] = r[key]
        flat[f"{label}__mKNN"] = np.array([r["mKNN"]])
    IC.save_npz(f"enrichment/enrich_{regime}.npz", **flat)


def main():
    ap = argparse.ArgumentParser(description="Component 2: neighbourhood variance ratio.")
    ap.add_argument("--regimes", nargs="*", default=REGIMES, choices=REGIMES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=10, help="shared-neighbour kNN size (mKNN).")
    ap.add_argument("--R", type=int, default=200, help="matched-perm null draws.")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    R = 50 if args.smoke else args.R
    for regime in args.regimes:
        print(f"\n=== {regime} ===")
        out = run_regime(regime, args.smoke, args.k, R)
        save_regime(regime, out)
    print("\nDone.")


if __name__ == "__main__":
    main()
