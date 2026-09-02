"""
block_a_combined.py — one combined curve pooling ALL EEG foundation models.

Unlike block_a (one panel per EEG model), here every EEG model's points are overlaid on a single
axis (marker+colour = EEG family, as in the cross-modal figures). x = LLM performance (1−BPB),
y = calibrated alignment. Two subplots: mKNN (top), CKA (bottom). One OLS fit across all points.

Three point definitions (three figures), each a 2-subplot combined curve:
  fully_disagg               — every (EEG variant × LLM) point (14×9 = 126).
  mean_EEG-fm_per-LLM        — per (EEG model × LLM), mean alignment over the model's sizes (5×9=45).
  largest-size_EEG-fm_per-LLM — per (EEG model × LLM), only the largest size (5×9 = 45).

Statistics per subplot: OLS β, R², the OLS p (naïve, treats all points as independent), and a
FPR-corrected **permutation p** — performance is shuffled across the 9 LLMs (all EEG points of an
LLM move together), so the effective sample is the 9 LLMs, not the 45/126 correlated points. Trust
the permutation p; the OLS p is anti-conservative because each EEG family contributes many points.

Output: data/paper-plots/cross-arch_cross-alignm_LLM-to-EEG_combined/{<case>.png} + stats + report.

Usage:  HF_HOME=$PWD/.hf_cache python src/paper_plots/block_a_combined.py
"""

from __future__ import annotations

import csv

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import _paper_common as P
from _paper_common import C

METRICS = [("mk_cal", "mKNN"), ("ck_cal", "CKA")]
K_PERM = 10000


def _index(rows):
    idx = {(r["eeg_model"], r["eeg_size"], r["partner"]): r for r in rows}
    stems = sorted([s for s in P.LLM_STEMS if any(r["partner"] == s for r in rows)], key=C.llm_x)
    models = [m for m in C.EEG if any(r["eeg_model"] == m for r in rows)]
    return idx, stems, models


def _points(case, mkey, idx, stems, models):
    """Return arrays (x=perf, y=alignment, model, llm_stem) for one case × metric."""
    xs, ys, ms, lls = [], [], [], []
    for m in models:
        sizes = C.EEG[m]["sizes"]
        for st in stems:
            if case == "fully_disagg":
                for sz in sizes:
                    r = idx.get((m, sz, st))
                    if r and np.isfinite(r[mkey]):
                        xs.append(C.llm_x(st)); ys.append(r[mkey]); ms.append(m); lls.append(st)
            else:
                use = sizes if case == "mean" else [sizes[-1]]      # mean-over-sizes | largest only
                vals = [idx[(m, sz, st)][mkey] for sz in use
                        if (m, sz, st) in idx and np.isfinite(idx[(m, sz, st)][mkey])]
                if vals:
                    xs.append(C.llm_x(st)); ys.append(float(np.mean(vals))); ms.append(m); lls.append(st)
    return np.array(xs), np.array(ys), np.array(ms), np.array(lls)


def _residualize(y, fams):
    """Δ = alignment minus its EEG-family mean — removes each family's baseline (the family
    confound), so all families share a common zero and a SINGLE line describes them together."""
    y = np.asarray(y, float).copy()
    for f in dict.fromkeys(fams.tolist()):
        sel = (fams == f) & np.isfinite(y)
        if sel.any():
            y[sel] = y[sel] - y[sel].mean()
    return y


def _fit(x, yres, llm_ids):
    """ONE OLS of Δalignment ~ perf across all models (family already removed). β equals the
    family-adjusted ANCOVA slope (perf ⊥ family); R² is the honest partial R². p(perm) shuffles
    perf across the 9 LLMs (calibrated)."""
    m = np.isfinite(x) & np.isfinite(yres)
    if m.sum() < 3 or np.ptp(x[m]) == 0:
        return np.nan, np.nan, np.nan, np.nan, int(m.sum())
    f = sm.OLS(yres[m], sm.add_constant(x[m])).fit()
    p_perm, _ = P.perm_p(yres[m], x[m], llm_ids[m], control_cats=[], K=K_PERM)
    return float(f.params[1]), float(f.pvalues[1]), float(p_perm), float(f.rsquared), int(m.sum())


def _draw(ax, x, yres, ms, stat, ylabel, stems, show_x):
    beta, p_ols, p_perm, r2, n = stat
    for m in dict.fromkeys(ms.tolist()):
        sel = ms == m
        ax.scatter(x[sel], yres[sel], s=46, color=P.FAM_COLORS.get(m, "#333"),
                   marker=P.FAM_MARKERS.get(m, "o"), edgecolor="k", lw=0.35, zorder=3)
    fin = np.isfinite(x) & np.isfinite(yres)
    if np.isfinite(beta) and fin.sum() >= 2:           # ONE line for all models
        a = yres[fin].mean() - beta * x[fin].mean()
        xf = np.linspace(x[fin].min(), x[fin].max(), 20)
        ax.plot(xf, a + beta * xf, "k--", lw=1.6, zorder=2)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_title(f"β={beta:+.3f}  R²={r2:.2f}  ·  p(OLS)={p_ols:.2g}  "
                 f"p(perm)={p_perm:.2g}{P.stars(p_perm)}  (n={n})", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12); ax.grid(alpha=0.3)
    if show_x:
        ax.set_xticks([C.llm_x(s) for s in stems])
        ax.set_xticklabels([P.LLM_DISPLAY.get(s, s) for s in stems], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("LLM performance", fontsize=11)


def main():
    rows = P.load_crossmodal_llm()
    idx, stems, models = _index(rows)
    out = P.DIR_A_COMBINED
    out.mkdir(parents=True, exist_ok=True)

    cases = [("fully_disagg", "fully_disagg"),
             ("mean", "mean_EEG-fm_per-LLM"),
             ("largest", "largest-size_EEG-fm_per-LLM")]
    stats = []
    for case, fname in cases:
        fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True, squeeze=False)
        for ri, (mkey, mlabel) in enumerate(METRICS):
            x, y, ms, lls = _points(case, mkey, idx, stems, models)
            yres = _residualize(y, ms)                       # Δ: remove EEG-family baseline
            stat = _fit(x, yres, lls)
            _draw(axes[ri][0], x, yres, ms, stat, f"Δ {mlabel}", stems, show_x=(ri == 1))
            stats.append(dict(case=fname, metric=mlabel, beta=stat[0], p_ols=stat[1],
                              p_perm=stat[2], r2=stat[3], n=stat[4]))
        handles = [plt.Line2D([], [], marker=P.FAM_MARKERS[m], ls="", color=P.FAM_COLORS[m],
                   label=m, markeredgecolor="k") for m in P.FAM_COLORS if m in models]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"EEG × LLM alignment vs LLM performance — {fname}", fontsize=12)
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        fig.savefig(out / f"{fname}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {fname}.png")

    with open(out / "stats_block_a_combined.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "metric", "beta", "p_ols", "p_perm", "r2", "n"])
        w.writeheader(); w.writerows(stats)

    lines = ["# Block (a) combined — one curve for all EEG models (family confound removed)\n",
             "x = LLM performance (1−BPB); **y = Δ alignment** = calibrated mKNN/CKA with each EEG "
             "family's baseline subtracted (residualised against EEG family), so all families share a "
             "common zero and a **single OLS line** describes them together — guaranteeing no EEG-"
             "family confound (à la the ΔR²/Δalign residual plots in `cross-modal_diff-arch_EEG-to-"
             "LLM`). Marker/colour = EEG family; two subplots (Δ mKNN, Δ CKA). β equals the family-"
             "adjusted slope (perf ⊥ family); R² is the honest partial R² (small).\n",
             "**p(perm)** shuffles performance across the 9 LLMs (all EEG points of an LLM move "
             "together) → effective sample = 9, not the 45/126 correlated points, correcting the FPR "
             "inflation from each family contributing many points. p(OLS) is the naïve, anti-"
             "conservative value.\n",
             "| case | metric | β | R² | p(OLS) | p(perm) | n |", "|---|---|---|---|---|---|---|"]
    for r in stats:
        lines.append(f"| {r['case']} | {r['metric']} | {r['beta']:+.3f} | {r['r2']:.2f} | "
                     f"{r['p_ols']:.2g} | {r['p_perm']:.2g} | {r['n']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"  block_a_combined → {out.name}: 3 figs + stats + report")


if __name__ == "__main__":
    main()
