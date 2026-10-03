#!/usr/bin/env python
"""
plot_axis1_calibrated.py — alignment_plots/f_axis1.png, on the cyclic-null axis.

AXIS 1: how much do two ADJACENT SIZES of the same EEG family agree with each other, and
does that agreement grow as the pair gets bigger? Each family with three sizes gives two
sibling pairs — (small, mid) and (mid, large) — plotted as "smaller pair" and "larger
pair". REVE has only two sizes, so it has a single pair and no slope; it is omitted, as
in the original.

WHAT CHANGED FROM THE ORIGINAL. The published figure plots the Aristotelian-calibrated
mKNN. Here y is the cyclic-null calibration used throughout this package,

    Delta = T(0) - mean over every non-identity rotation of T(d),

for BOTH metrics, so the two panels differ only in the metric and both sit on the same
null as every other figure in the set. Points are the mean over the six subjects of the
per-subject calibrated value (cache/cyclic_*_persubj.npz), bars +-1 SEM across subjects.

Note this is the one axis where SIBLING pairs are the point rather than the confound: the
cross-model figures mask them out precisely because two sizes of one architecture share
training and data, which is what Axis 1 sets out to measure.

Usage:
    python plot_axis1_calibrated.py
    python plot_axis1_calibrated.py --with-null
    python plot_axis1_calibrated.py --out ../alignment_plots/f_axis1_calibrated.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import _common as C  # noqa: E402
from plot_panel_b import CACHE, THEME, FAMILY_COLOR, FAMILY_MARKER  # noqa: E402

SRC = {"mknn": "cyclic_mknn_persubj.npz", "cka": "cyclic_cka_eeg_persubj.npz"}
LABEL = {"mknn": r"$\mathbf{mKNN}$", "cka": r"$\mathbf{CKA}$"}
INK = "#33475B"


def adjacent(metric):
    """-> {family: (delta (2, S), sizes)} for the two adjacent-size pairs of each family."""
    d = np.load(CACHE / SRC[metric], allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    idx = {k: i for i, k in enumerate(keys)}
    at = {(int(a), int(b)): n for n, (a, b) in enumerate(d["pairs"])}
    out = {}
    for m in C.EEG:
        sz = C.EEG[m]["sizes"]
        if len(sz) < 3:            # a single pair has no smaller/larger contrast
            continue
        rows = []
        for a, b in zip(sz, sz[1:]):
            i, j = idx[f"{m}-{a}"], idx[f"{m}-{b}"]
            c = np.asarray(d["curves"][at[(min(i, j), max(i, j))]], float)  # (S, W)
            rows.append(c[:, 0] - c[:, 1:].mean(axis=1))                    # (S,)
        out[m] = (np.stack(rows), sz)
    return out


def adjacent_null(metric):
    """-> {family: (null mean (2, S), sizes)} for the same adjacent-size pairs.

    Every entry is first averaged over the 1,079 non-identity cyclic rotations within
    each subject. Plot-level error bars can therefore use the same between-subject SEM
    as the calibrated scores.
    """
    d = np.load(CACHE / SRC[metric], allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    idx = {k: i for i, k in enumerate(keys)}
    at = {(int(a), int(b)): n for n, (a, b) in enumerate(d["pairs"])}
    out = {}
    for m in C.EEG:
        sz = C.EEG[m]["sizes"]
        if len(sz) < 3:
            continue
        rows = []
        for a, b in zip(sz, sz[1:]):
            i, j = idx[f"{m}-{a}"], idx[f"{m}-{b}"]
            c = np.asarray(d["curves"][at[(min(i, j), max(i, j))]], float)
            rows.append(c[:, 1:].mean(axis=1))
        out[m] = (np.stack(rows), sz)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-null", action="store_true",
                    help="Also plot each pair's cyclic-null mean with its across-subject SEM.")
    args = ap.parse_args()
    plt.rcParams.update({**THEME, "axes.ymargin": 0.12})

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    x = np.array([0.0, 1.0])
    for ax, metric in zip(axes, ["mknn", "cka"]):
        data = adjacent(metric)
        null_data = adjacent_null(metric) if args.with_null else {}
        for m, (D, sz) in data.items():
            mu = D.mean(axis=1)
            se = D.std(axis=1, ddof=1) / np.sqrt(D.shape[1])
            if args.with_null:
                N, _ = null_data[m]
                null_mu = N.mean(axis=1)
                null_se = N.std(axis=1, ddof=1) / np.sqrt(N.shape[1])
                ax.errorbar(x, null_mu, yerr=null_se, color=FAMILY_COLOR[m],
                            lw=1.8, ls="--", marker=FAMILY_MARKER[m], ms=8,
                            capsize=3, elinewidth=1.2, mfc="white", mec=FAMILY_COLOR[m],
                            mew=1.4, alpha=0.85, zorder=3)
            ax.errorbar(x, mu, yerr=se, color=FAMILY_COLOR[m], lw=2.4,
                        marker=FAMILY_MARKER[m], ms=10, capsize=4, elinewidth=1.4,
                        mec="white", mew=1.2, zorder=5)
            ax.annotate(C.DISPLAY[m], (x[1], mu[1]), textcoords="offset points",
                        xytext=(12, 0), va="center", ha="left", fontsize=14,
                        color=FAMILY_COLOR[m], fontweight="bold", clip_on=False)
        ax.set_xticks(x)
        ax.set_xticklabels(["smaller pair", "larger pair"], fontsize=20)
        ax.set_xlim(-0.28, 1.55)
        ax.set_ylabel(LABEL[metric], fontsize=28, color=INK, labelpad=10)
        ax.tick_params(axis="both", labelsize=20)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(0.7)
        ax.grid(zorder=0)
    if args.with_null:
        style_handles = [
            Line2D([0], [0], color="0.2", lw=2.4, marker="o", ms=8,
                   label=r"calibrated score ($\pm 1$ SEM)"),
            Line2D([0], [0], color="0.2", lw=1.8, ls="--", marker="o", ms=7,
                   mfc="white", mec="0.2", label=r"cyclic-null mean ($\pm 1$ SEM)"),
        ]
        fig.legend(handles=style_handles, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.01), fontsize=14, frameon=False)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
    else:
        fig.tight_layout()

    default_name = ("f_axis1_calibrated_with_null.png" if args.with_null
                    else "f_axis1_calibrated.png")
    out = Path(args.out) if args.out else _HERE.parent / "alignment_plots" / default_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}\n")
    for metric in ("mknn", "cka"):
        print(f"  {LABEL[metric].strip('$').replace(chr(92)+'mathbf','').strip('{}')}:")
        for m, (D, sz) in adjacent(metric).items():
            mu, se = D.mean(axis=1), D.std(axis=1, ddof=1) / np.sqrt(D.shape[1])
            pairs = " -> ".join(f"{C.PARAMS[m][a]}M/{C.PARAMS[m][b]}M"
                                for a, b in zip(sz, sz[1:]))
            print(f"    {C.DISPLAY[m]:12} {pairs:22} "
                  f"{mu[0]:+.5f}+-{se[0]:.5f}  ->  {mu[1]:+.5f}+-{se[1]:.5f}"
                  f"   ({'rises' if mu[1] > mu[0] else 'falls'})")
            if args.with_null:
                N, _ = adjacent_null(metric)[m]
                nmu = N.mean(axis=1)
                nse = N.std(axis=1, ddof=1) / np.sqrt(N.shape[1])
                print(f"      null mean{'':13} "
                      f"{nmu[0]:+.5f}+-{nse[0]:.5f}  ->  {nmu[1]:+.5f}+-{nse[1]:.5f}")


if __name__ == "__main__":
    main()
