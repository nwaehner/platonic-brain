#!/usr/bin/env python
"""
prh_axes.py — the three-axis PRH summary figure (F1).

One panel per axis of the hypothesis, sharing the same story: better/larger EEG
foundation models converge.

  (a) Intra-model      — within a family, alignment between adjacent size pairs,
                         ordered small -> large.
  (b) Cross-model      — alignment of each EEG variant to all OTHER EEG
      (intra-modality)   architectures, vs its NICE-marker decodability.
  (c) Cross-modality   — alignment of each EEG variant to fMRI, vs the same.

Panels (b) and (c) are **family-centred**: the 5 EEG architectures sit at very
different baseline alignment levels, and pooling them raw hides the trend
(rho = +0.27, p = 0.34 pooled vs rho_adj = +0.82 corrected). Both axes are
mean-centred within family, which is the ANCOVA `~ x + C(family)` contrast
drawn as a partial-regression plot.

Inputs (all already on disk — no HF download, no model inference):
  outputs/intramodal.npz                      (tracked)
  outputs/fmri_vs_all__<family>.npz           (tracked)
  ../interp/outputs/eeg_alignment/summary.npz (GITIGNORED — produced by
      src/interp/eeg_features/eeg_feature_alignment.py; rerun the interp
      studies if it is missing)

Output: outputs/prh_axes_summary.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SUMMARY = os.path.join(HERE, "..", "interp", "outputs", "eeg_alignment", "summary.npz")

# Okabe-Ito subset, ordered so every adjacent pair clears the CVD separation check.
FAMILY_COLOR = {
    "femba": "#0072B2",
    "luna": "#E69F00",
    "neurolm": "#009E73",
    "reve": "#D55E00",
    "steegformer": "#CC79A7",
}
FAMILY_MARKER = {  # secondary encoding, so identity never rests on hue alone
    "femba": "o",
    "luna": "s",
    "neurolm": "^",
    "reve": "D",
    "steegformer": "v",
}
# ascending size order per family
SIZES = {
    "femba": ["femba-tiny", "femba-base", "femba-large"],
    "luna": ["luna-base", "luna-large", "luna-huge"],
    "neurolm": ["neurolm-b", "neurolm-l", "neurolm-xl"],
    "reve": ["reve-base", "reve-large"],
    "steegformer": ["steegformer-small", "steegformer-base", "steegformer-large"],
}

THEME = {
    "figure.dpi": 150,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
}


def centre_within_family(x, fam):
    """Mean-centre x within each family (the ANCOVA partial-regression contrast)."""
    x = np.asarray(x, float)
    out = x.copy()
    for f in set(fam):
        m = np.asarray(fam) == f
        out[m] = x[m] - x[m].mean()
    return out


def panel_intramodel(ax):
    """(a) Within-family alignment between adjacent size pairs, small -> large."""
    d = np.load(os.path.join(OUT, "intramodal.npz"), allow_pickle=True)
    pairs = {}
    for k in d.files:
        if not k.endswith("__s_cal"):
            continue
        fam, a, b, _ = k.split("__")
        pairs.setdefault(fam, []).append((f"{a}→{b}", float(d[k].mean()), float(d[k].std())))

    n_up = 0
    for fam, rows in sorted(pairs.items()):
        xs = np.arange(len(rows))
        ys = [r[1] for r in rows]
        es = [r[2] for r in rows]
        ax.errorbar(
            xs, ys, yerr=es, color=FAMILY_COLOR[fam], marker=FAMILY_MARKER[fam],
            markersize=7, linewidth=2, capsize=3, elinewidth=1, zorder=3,
        )
        ax.annotate(
            fam, (xs[-1], ys[-1]), textcoords="offset points", xytext=(7, 0),
            color=FAMILY_COLOR[fam], fontsize=9, va="center", weight="bold",
        )
        n_up += ys[-1] > ys[0]

    ax.axhline(0.003, color="0.45", linestyle=":", linewidth=1.2, zorder=1)
    ax.annotate("permutation null ≈ 0.003", (0.02, 0.003), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 5), fontsize=8, color="0.35")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["smaller pair", "larger pair"])
    ax.set_xlim(-0.25, 1.55)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("calibrated mKNN between adjacent sizes")
    ax.set_title(f"(a) Intra-model\nlarger sizes converge — {n_up}/{len(pairs)} families", loc="left")
    ax.grid(axis="y", zorder=0)


def _scatter_panel(ax, x, y, fam, labels, xlabel, ylabel, title):
    """Shared body for the two family-centred partial-regression panels."""
    xc, yc = centre_within_family(x, fam), centre_within_family(y, fam)
    for f in sorted(set(fam)):
        m = np.asarray(fam) == f
        ax.scatter(xc[m], yc[m], c=FAMILY_COLOR[f], marker=FAMILY_MARKER[f],
                   s=55, edgecolor="white", linewidth=1.0, zorder=3, label=f)
    if len(xc) > 2:
        b, a = np.polyfit(xc, yc, 1)
        xx = np.linspace(xc.min(), xc.max(), 50)
        ax.plot(xx, a + b * xx, color="0.25", linewidth=1.6, zorder=2)
    rho, p = spearmanr(xc, yc)
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.annotate(
        rf"$\rho_{{adj}}$ = {rho:+.2f}" + "\n" + (f"p = {p:.3f}" if p >= 1e-3 else "p < 0.001"),
        xy=(0.04, 0.90), xycoords="axes fraction", fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.8),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(zorder=0)
    return rho, p


def load_eeg_table():
    """NICE decodability + cross-FAMILY alignment per EEG variant.

    `summary.npz`'s own `mean_mknn` averages a model against every *other model*,
    which includes its own family's other sizes — and sibling alignment is ~5x
    higher than cross-family (0.50 vs 0.10), so it would let Axis 1 leak into
    Axis 2. We re-average the raw pairwise matrix with same-family pairs masked
    out, so this panel is strictly across architectures. (The trend holds either
    way: NICE rho_adj = +0.84 cross-family vs +0.82 including siblings.)
    """
    if not os.path.exists(SUMMARY):
        raise SystemExit(
            f"missing {SUMMARY}\n"
            "  -> produced by the interp studies (src/interp/outputs/ is gitignored).\n"
            "     Run: bash src/interp/run-extra-studies.sh"
        )
    s = np.load(SUMMARY, allow_pickle=True)
    keys = [str(k) for k in s["keys"]]
    fam = np.array([str(f) for f in s["family"]])
    mk = np.array(s["MK"], float)
    mk[fam[:, None] == fam[None, :]] = np.nan       # drop same-family pairs (and the diagonal)
    return keys, list(fam), np.asarray(s["perf_mean"], float), np.nanmean(mk, axis=1)


def load_fmri_eeg():
    """Per-EEG-variant alignment to fMRI, read off each family's fmri_vs_all npz."""
    mknn, cka = {}, {}
    for fam in SIZES:
        for cand in (fam, "femba_luna" if fam in ("femba", "luna") else fam):
            p = os.path.join(OUT, f"fmri_vs_all__{cand}.npz")
            if os.path.exists(p):
                break
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        for lab, kind, m, c in zip(d["labels"], d["kinds"], d["mknn_cal"], d["cka"]):
            if str(kind) == "eeg":
                mknn[str(lab)], cka[str(lab)] = float(m), float(c)
    return mknn, cka


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(OUT, "prh_axes_summary.png"))
    args = ap.parse_args()

    plt.rcParams.update(THEME)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    panel_intramodel(axes[0])

    keys, fam, perf, xarch = load_eeg_table()
    _scatter_panel(
        axes[1], perf, xarch, fam, keys,
        "NICE decodability (family-centred)",
        "mKNN to other EEG families (family-centred)",
        "(b) Cross-model, within EEG\nbetter models agree more with other architectures",
    )

    fmri_mknn, fmri_cka = load_fmri_eeg()
    keep = [i for i, k in enumerate(keys) if k in fmri_cka]
    _scatter_panel(
        axes[2], perf[keep], np.array([fmri_cka[keys[i]] for i in keep]),
        [fam[i] for i in keep], [keys[i] for i in keep],
        "NICE decodability (family-centred)",
        "linear CKA to fMRI (family-centred)",
        "(c) Cross-modality\nbetter models agree more with fMRI",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        "Platonic convergence in EEG foundation models: better models align more, on all three axes",
        fontsize=14, y=1.0,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
