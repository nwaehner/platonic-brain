#!/usr/bin/env python
"""
fmri_dissociation.py — how fMRI aligns with every other representation (F7).

One point per (model, window-grid). x = calibrated mKNN (LOCAL neighbourhood
agreement), y = linear CKA (GLOBAL similarity-matrix agreement), coloured by
the model's modality.

The two axes separate the modalities almost perfectly:

  * vision and language models sit bottom-right — they agree with fMRI about
    WHICH windows are alike (local structure), but not about the overall
    geometry.
  * EEG foundation models sit top-left — they share fMRI's global geometry
    (CKA 0.42-0.53) while their nearest-neighbour sets barely overlap.

fMRI is the pivot: it aligns with vision/language *locally* and with EEG
*globally*, so the two brain modalities are not interchangeable probes.

Input:  outputs/fmri_vs_all__<grid>.npz  (tracked; written by fmri_vs_all.py)
Output: outputs/fmri_dissociation.png
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

# Okabe-Ito; adjacent pairs clear the CVD separation check.
MODALITY = {
    "eeg": ("EEG foundation models", "#0072B2", "o"),
    "vision": ("vision models", "#D55E00", "s"),
    "llm": ("caption LLMs", "#009E73", "^"),
}

THEME = {
    "figure.dpi": 150,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
}


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "fmri_vs_all__*.npz"))):
        d = np.load(p, allow_pickle=True)
        grid = str(d["family"])
        for lab, kind, m, c in zip(d["labels"], d["kinds"], d["mknn_cal"], d["cka"]):
            rows.append((grid, str(lab), str(kind), float(m), float(c)))
    if not rows:
        raise SystemExit(f"no fmri_vs_all__*.npz in {OUT} — run fmri_vs_all.py first")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(OUT, "fmri_dissociation.png"))
    args = ap.parse_args()

    rows = load()
    plt.rcParams.update(THEME)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for kind, (label, color, marker) in MODALITY.items():
        sel = [r for r in rows if r[2] == kind]
        if not sel:
            continue
        ax.scatter(
            [r[3] for r in sel], [r[4] for r in sel],
            c=color, marker=marker, s=60, alpha=0.85,
            edgecolor="white", linewidth=1.0, zorder=3,
            label=f"{label}  (n={len(sel)})",
        )

    # Direct cluster labels — identity never rests on hue alone.
    for kind, dx, dy, ha in (("eeg", 0.030, -0.055, "left"), ("vision", -0.012, 0.03, "right")):
        sel = [r for r in rows if r[2] == kind]
        if sel:
            mx = np.mean([r[3] for r in sel])
            my = np.mean([r[4] for r in sel])
            ax.annotate(
                MODALITY[kind][0].replace(" ", "\n", 1),
                (mx + dx, my + dy), color=MODALITY[kind][1],
                fontsize=11, weight="bold", ha=ha, va="bottom", zorder=4,
            )

    ax.set_xlabel("calibrated mKNN with fMRI   →   local neighbourhood agreement")
    ax.set_ylabel("linear CKA with fMRI   →   global geometry agreement")
    ax.set_title(
        "fMRI aligns with vision/language locally, and with EEG globally\n"
        "one point per model × window grid",
        loc="left",
    )
    ax.grid(zorder=0)
    ax.legend(loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")

    for kind in MODALITY:
        sel = [r for r in rows if r[2] == kind]
        if sel:
            print(f"  {kind:<7} n={len(sel):<3} mKNN {np.mean([r[3] for r in sel]):.3f}"
                  f"  CKA {np.mean([r[4] for r in sel]):.3f}")


if __name__ == "__main__":
    main()
