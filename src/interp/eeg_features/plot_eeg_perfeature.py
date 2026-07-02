"""
plot_eeg_perfeature.py — per-feature cross-architectural alignment-vs-decodability scatters.

Standalone replot from outputs/eeg_alignment/summary.npz (needs only numpy + scipy + matplotlib,
none of the interp import chain). One panel PER NICE feature instead of the single mean-over-
features scatter: x = that feature's per-model decodability R², y = per-model mean cross-arch
MKNN (and a second figure for CKA), points coloured by EEG family, per-panel fit + Spearman ρ/p.

Usage:  python src/interp/plot_eeg_perfeature.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent.parent / "outputs" / "eeg_alignment"
FAM_COLORS = {"femba": "#1b9e77", "luna": "#d95f02", "neurolm": "#7570b3",
              "reve": "#e7298a", "steegformer": "#66a61e"}
FAM_MARKERS = {"femba": "o", "luna": "s", "neurolm": "^", "reve": "D", "steegformer": "P"}


def _grid(perf, align, fams, names, ylabel, path):
    F = len(names)
    nc = 4
    nr = int(np.ceil(F / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.5 * nc, 3.0 * nr), squeeze=False)
    for fi, name in enumerate(names):
        ax = axes[fi // nc][fi % nc]
        x = perf[:, fi].astype(float)
        y = align.astype(float)
        for xi, yi, fam in zip(x, y, fams):
            ax.scatter(xi, yi, s=46, color=FAM_COLORS.get(fam, "#333"),
                       marker=FAM_MARKERS.get(fam, "o"), edgecolor="k", lw=0.35)
        m = np.isfinite(x) & np.isfinite(y)
        rho, p = spearmanr(x[m], y[m]) if m.sum() >= 3 else (np.nan, np.nan)
        if m.sum() >= 2 and np.ptp(x[m]) > 0:
            b, a = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(x[m].min(), x[m].max(), 20)
            ax.plot(xs, a + b * xs, "k--", lw=1)
        sig = "*" if (np.isfinite(p) and p < 0.05) else ""
        ax.set_title(f"{name}{sig}\nρ={rho:+.2f}  p={p:.2g}", fontsize=8)
        ax.set_xlabel("R²", fontsize=7); ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(F, nr * nc):
        axes[j // nc][j % nc].axis("off")
    handles = [plt.Line2D([], [], marker=FAM_MARKERS[f], ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in FAM_COLORS if f in fams]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Per-feature: decodability vs {ylabel} (each point = one EEG model)",
                 y=1.0, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def main():
    z = np.load(OUT / "summary.npz", allow_pickle=True)
    names = [str(x) for x in z["feat_names"]]
    fams = [str(x) for x in z["family"]]
    _grid(z["perf"], z["mean_mknn"], fams, names, "mean cross-arch MKNN",
          OUT / "crossarch_perfeature_mknn.png")
    _grid(z["perf"], z["mean_cka"], fams, names, "mean cross-arch CKA",
          OUT / "crossarch_perfeature_cka.png")


if __name__ == "__main__":
    main()
