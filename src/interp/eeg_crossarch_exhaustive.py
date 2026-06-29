"""
eeg_crossarch_exhaustive.py — exhaustive robustness pass on the EEG cross-architectural
alignment ↔ decodability result (replot/recompute from outputs/eeg_alignment/summary.npz;
needs only numpy + scipy + matplotlib).

The single mean-over-all-features performance is dragged down by poorly-decoded markers
(pe_broad, pe_alpha). Here we recompute the per-model cross-arch MKNN/CKA-vs-performance
Spearman correlation under several performance definitions, each with a bootstrap 95% CI:

  all          mean R² over all features
  well_decoded mean R² over features with across-model mean R² > THRESH (default 0.3)
  spectral     band powers (abs/rel δθαβ) + spectral entropy / median freq / edge
  complexity   permutation entropy (broad/θ/α)
  hjorth       Hjorth mobility + complexity

Outputs:  crossarch_exhaustive.csv · crossarch_exhaustive.png  (in outputs/eeg_alignment/).

Usage:  python src/interp/eeg_crossarch_exhaustive.py [--thresh 0.3] [--boot 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent / "outputs" / "eeg_alignment"
FAM_COLORS = {"femba": "#1b9e77", "luna": "#d95f02", "neurolm": "#7570b3",
              "reve": "#e7298a", "steegformer": "#66a61e"}
FAM_MARKERS = {"femba": "o", "luna": "s", "neurolm": "^", "reve": "D", "steegformer": "P"}


def _is(prefixes, name):
    return any(name == p or name.startswith(p) for p in prefixes)


def feature_groups(names, perf, thresh):
    mean_r2 = np.nanmean(perf, axis=0)
    spectral = [n for n in names if n.startswith(("abs_", "rel_"))
                or n in ("spec_entropy", "median_freq", "spec_edge90")]
    complexity = [n for n in names if n.startswith("pe_")]
    hjorth = [n for n in names if n.startswith("hjorth_")]
    well = [n for i, n in enumerate(names) if mean_r2[i] > thresh]
    return {"all": list(names), "well_decoded": well, "spectral": spectral,
            "complexity": complexity, "hjorth": hjorth}


def perf_of(perf, names, group):
    cols = [i for i, n in enumerate(names) if n in set(group)]
    if not cols:
        return np.full(perf.shape[0], np.nan)
    return np.nanmean(perf[:, cols], axis=1)


def boot_ci(x, y, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 4:
        return (np.nan, np.nan)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(x[idx]) == 0 or np.ptp(y[idx]) == 0:
            continue
        rs.append(spearmanr(x[idx], y[idx]).statistic)
    if not rs:
        return (np.nan, np.nan)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    z = np.load(OUT / "summary.npz", allow_pickle=True)
    names = [str(x) for x in z["feat_names"]]
    fams = [str(x) for x in z["family"]]
    perf = z["perf"].astype(float)                    # (n_models, F)
    mk, ck = z["mean_mknn"].astype(float), z["mean_cka"].astype(float)
    groups = feature_groups(names, perf, args.thresh)

    rows = []
    print(f"{'performance':<14}{'nfeat':>6}  {'metric':<5}{'rho':>8}{'p':>9}   95% CI")
    for gname, gfeats in groups.items():
        p = perf_of(perf, names, gfeats)
        for metric, align in [("MKNN", mk), ("CKA", ck)]:
            m = np.isfinite(p) & np.isfinite(align)
            rho, pv = spearmanr(p[m], align[m]) if m.sum() >= 4 else (np.nan, np.nan)
            lo, hi = boot_ci(p, align, args.boot)
            rows.append(dict(performance=gname, n_features=len(gfeats), metric=metric,
                             rho=rho, p=pv, ci_lo=lo, ci_hi=hi))
            print(f"{gname:<14}{len(gfeats):>6}  {metric:<5}{rho:>+8.3f}{pv:>9.3g}   "
                  f"[{lo:+.2f}, {hi:+.2f}]")

    # save CSV
    import csv
    with open(OUT / "crossarch_exhaustive.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n  saved → {OUT/'crossarch_exhaustive.csv'}")

    _plot(rows, perf, names, mk, ck, fams, groups, args.thresh)


def _plot(rows, perf, names, mk, ck, fams, groups, thresh):
    gnames = list(groups)
    fig = plt.figure(figsize=(13, 5))
    # (left) ρ ± CI across performance definitions, MKNN vs CKA
    ax = fig.add_subplot(1, 2, 1)
    yy = np.arange(len(gnames))
    for off, metric, col in [(-0.18, "MKNN", "#2c7fb8"), (0.18, "CKA", "#d95f02")]:
        rr = [next(r for r in rows if r["performance"] == g and r["metric"] == metric) for g in gnames]
        rho = [r["rho"] for r in rr]
        lo = [r["rho"] - r["ci_lo"] for r in rr]; hi = [r["ci_hi"] - r["rho"] for r in rr]
        ax.errorbar(rho, yy + off, xerr=[lo, hi], fmt="o", color=col, label=metric, capsize=3)
        for r, y in zip(rr, yy):
            if np.isfinite(r["p"]) and r["p"] < 0.05:
                ax.text(r["rho"], y + off + 0.08, "*", ha="center", fontsize=11, color=col)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(gnames)
    ax.set_xlabel("Spearman ρ (performance vs cross-arch alignment)  ± bootstrap 95% CI")
    ax.set_title("Cross-arch alignment ↔ decodability\nby performance definition (* p<0.05)",
                 fontsize=10)
    ax.legend(fontsize=8)
    # (right) the well-decoded MKNN scatter
    ax2 = fig.add_subplot(1, 2, 2)
    p = perf_of(perf, names, groups["well_decoded"])
    for xi, yi, fam in zip(p, mk, fams):
        ax2.scatter(xi, yi, s=70, color=FAM_COLORS.get(fam, "#333"),
                    marker=FAM_MARKERS.get(fam, "o"), edgecolor="k", lw=0.4)
    m = np.isfinite(p) & np.isfinite(mk)
    rho, pv = spearmanr(p[m], mk[m])
    if np.ptp(p[m]) > 0:
        b, a = np.polyfit(p[m], mk[m], 1)
        xs = np.linspace(p[m].min(), p[m].max(), 20); ax2.plot(xs, a + b * xs, "k--", lw=1)
    ax2.set_xlabel(f"performance (mean R² over {len(groups['well_decoded'])} well-decoded features)")
    ax2.set_ylabel("mean cross-arch MKNN")
    ax2.set_title(f"Well-decoded performance vs MKNN\nρ={rho:+.3f}  p={pv:.3g}", fontsize=10)
    handles = [plt.Line2D([], [], marker=FAM_MARKERS[f], ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in FAM_COLORS if f in fams]
    ax2.legend(handles=handles, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "crossarch_exhaustive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {OUT/'crossarch_exhaustive.png'}")


if __name__ == "__main__":
    main()
