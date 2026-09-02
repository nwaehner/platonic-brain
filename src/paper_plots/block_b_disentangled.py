"""
block_b_disentangled.py — per-feature ("disentangled") version of block (b).

Block (b) plots one point per consecutive size pair with x = the *tier-mean* decodability
R². Here we open up each tier: one subplot per individual feature of that tier, laid out
on a near-square (n×n) grid. Within a subplot the 9 pairs are scattered as usual
(x = mean R² of the two sizes for that single feature, y = intramodal calibrated
alignment), with a Spearman ρ, raw p and BH-FDR q (corrected within tier × metric).

Grids: eeg 16 → 4×4 · low 12 → 4×3 · semantic 24 → 5×5.
One figure per (tier × metric); metrics are mKNN and CKA.

The `semantic` tier disaggregates over the top-24 PCA components of the ModernBERT space
(tier `semantic_pca` in the cache, see P.disagg_tier) — a raw BERT dimension has no standalone
meaning, so per-dimension panels would not be readable.

Outputs → data/paper-plots/intra-arch_same-modality_EEG__semantic-nestedcv/
  intra__disent__<tier>__{mknn,cka}.png  (6) · stats_block_b_disentangled.csv

Usage:  python src/paper_plots/block_b_disentangled.py
"""

from __future__ import annotations

import csv
import math

import numpy as np
import matplotlib.pyplot as plt

import _paper_common as P

METRICS = [("mk_cal", "mknn", "mKNN (%)"), ("ck_cal", "cka", "CKA (%)")]


def bh_fdr(pvals):
    """Benjamini–Hochberg q-values; NaNs pass through."""
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if idx.size == 0:
        return q
    order = idx[np.argsort(p[idx])]
    n = order.size
    ranked = p[order] * n / np.arange(1, n + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return q


def _short(name):
    """'clip:traffic light' → 'traffic light'; 'hjorth_mobility' stays."""
    return name.split(":", 1)[1] if ":" in name else name


def _pair_x(vmap, model, a, b):
    va, vb = vmap.get(f"{model}-{a}"), vmap.get(f"{model}-{b}")
    vals = [v for v in (va, vb) if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _grid_shape(n):
    """Near-square (rows, cols) with cols >= rows."""
    c = math.ceil(math.sqrt(n))
    r = math.ceil(n / c)
    return r, c


def _subplot(ax, xs, ys, models, name, rho, p, q, small):
    for x, y, m in zip(xs, ys, models):
        ax.scatter(x, y, s=22 if small else 60,
                   color=P.FAM_COLORS.get(m, "#333"), marker=P.FAM_MARKERS.get(m, "o"),
                   edgecolor="k", lw=0.3, zorder=3)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    msk = np.isfinite(xs) & np.isfinite(ys)
    if msk.sum() >= 2 and np.ptp(xs[msk]) > 0:
        b, a = np.polyfit(xs[msk], ys[msk], 1)
        xf = np.linspace(xs[msk].min(), xs[msk].max(), 20)
        ax.plot(xf, a + b * xf, "k--", lw=0.9, zorder=2)
    sig = P.stars(q) if np.isfinite(q) else ""
    ax.set_title(f"{_short(name)}\nρ={rho:+.2f} p={p:.2g} q={q:.2g}{sig}",
                 fontsize=6.5 if small else 9)
    ax.tick_params(labelsize=5 if small else 8)
    ax.grid(alpha=0.25)


def main():
    P.ensure_dirs()
    dec = P.load_decodability()
    intra = P.load_intramodal()
    pairs = [intra[k] for k in intra]
    models_present = {d["model"] for d in pairs}

    names = dec["feat_names"]
    perf = dec["perf"]
    handles = [plt.Line2D([], [], marker=P.FAM_MARKERS[m], ls="", color=P.FAM_COLORS[m],
                          label=m, markeredgecolor="k")
               for m in P.FAM_COLORS if m in models_present]

    stats_out = []
    for tier in P.TIERS:
        cols = np.flatnonzero(dec["feat_tier"] == P.disagg_tier(tier))
        # x per feature: {variant key → R²}
        xmaps = [{k: float(v) for k, v in zip(dec["keys"], perf[:, c])} for c in cols]
        xs_per_feat = [[_pair_x(xm, d["model"], d["a"], d["b"]) for d in pairs] for xm in xmaps]
        ms = [d["model"] for d in pairs]

        n = len(cols)
        rows, ncol = _grid_shape(n)
        small = n > 40

        for mkey, mslug, mlabel in METRICS:
            ys = [100.0 * float(np.nanmean(d[mkey])) for d in pairs]
            rp = [P.spearman(x, ys) for x in xs_per_feat]
            qs = bh_fdr([p for _, p in rp])

            fig, axes = plt.subplots(rows, ncol, figsize=(2.3 * ncol, 2.5 * rows),
                                     squeeze=False, sharey=True)
            flat = axes.ravel()
            for i, c in enumerate(cols):
                (rho, p), q = rp[i], qs[i]
                _subplot(flat[i], xs_per_feat[i], ys, ms, names[c], rho, p, q, small)
                stats_out.append(dict(tier=tier, feature=names[c], metric=mslug,
                                      rho=rho, p=p, q_bh=q,
                                      n=int(np.isfinite(np.asarray(xs_per_feat[i], float)).sum())))
            for ax in flat[n:]:
                ax.axis("off")
            for ax in axes[-1]:
                ax.set_xlabel("R²", fontsize=6 if small else 9)
            for ax in axes[:, 0]:
                ax.set_ylabel(mlabel, fontsize=6 if small else 9)

            fig.suptitle(f"Intra-architecture (EEG) — {P.TIER_LABEL[tier]}, per-feature: "
                         f"decodability vs intramodal alignment ({mlabel})\n"
                         f"{n} features · 9 size pairs each · q = BH-FDR within this panel set",
                         fontsize=12 if small else 13)
            fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                       fontsize=9, bbox_to_anchor=(0.5, -0.005))
            fig.tight_layout(rect=[0, 0.02, 1, 0.96 if small else 0.93])
            path = P.DIR_B_NESTED / f"intra__disent__{tier}__{mslug}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {path.name}  ({rows}×{ncol}, {n} features)")

    out = P.DIR_B_NESTED / "stats_block_b_disentangled.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tier", "feature", "metric", "rho", "p", "q_bh", "n"])
        w.writeheader()
        w.writerows(stats_out)
    print(f"  saved {out.name}  ({len(stats_out)} rows)")


if __name__ == "__main__":
    main()
