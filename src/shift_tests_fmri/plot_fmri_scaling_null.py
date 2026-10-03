#!/usr/bin/env python
"""
plot_fmri_scaling_null.py — F1 re-laid-out in the style of shift_tests/plot_femba_dinov2.py.

Same data and same statistic as plot_fmri_scaling.fig_scaling_view; the layout, the theme
and the null are what change.

    fmri_<mod>_scaling_view.png    1 x N, a colourbar keyed to d, |d| >= 100 null
    fmri_scaling_satellites.png    2 x 3, this script, full C_n null

One panel per scaling family: vision architectures fill the first two columns, LLM
families the third. In each panel the coloured line is the true pairing (d = 0), the
faint cloud behind it is every circular rotation of the same cell, and the grey band is
the 5-95 percentile of that cloud at each size.

WHAT p IS. Nothing is annotated on the panels; both statistics go to stdout.
p tests whether the family CLEARS THE NULL — the same statistic
shift_tests/plot_femba_dinov2.py reports. Each size's curve is z-scored against its own
rotations, the z is averaged over the family's sizes, and the d = 0 mean-z is ranked
against the mean-z at every rotation:

    is the true pairing more aligned than a rotated one, across this family as a whole?

It says nothing about whether alignment RISES WITH SIZE. That is a different question with
a different answer, and the two disagree here: fMRI x DINOv2 clears the null at p = 0.001
while its slope across sizes is indistinguishable from a rotation's (slope p = 0.22). The
slope statistic is computed alongside it. Neither is drawn: at two to five points per
family every corner of a panel is near either the curve or the null band, so an annotation
lands on ink more often than not.

The null is the COMPLETE cyclic group by default (every d != 0, no far-tail cut), matching
CYCLIC_NULL.md and the decodability_rebuild figures; --tail-min 100 reproduces the far-tail
variant that fmri_<mod>_scaling_view.png uses.

Usage:
    python plot_fmri_scaling_null.py                       # 2x3, vision + language
    python plot_fmri_scaling_null.py --npz outputs_vs_vision/circular_shift_fmri_vision.npz
    python plot_fmri_scaling_null.py --hero                # add the net-alignment panel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import _common as C  # noqa: E402
from plot_fmri_scaling import group_title, load  # noqa: E402

# verbatim from shift_tests/plot_femba_dinov2.py
PANEL_BG = "#F7F7F7"
GRID_C = "#DDDDDD"
INK = "#33475B"
THEME = {
    "figure.dpi": 150, "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "mathtext.rm": "cmr10",
    "axes.formatter.use_mathtext": True,
    "axes.edgecolor": GRID_C, "axes.linewidth": 1.0,
    "grid.color": GRID_C, "grid.linewidth": 1.0,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.major.size": 0, "ytick.major.size": 0,
}
# One pair of sweeps per metric. The CKA files carry the mKNN files' cell tables
# verbatim, so both metrics label and order their cells identically.
NPZ = {"mknn": ["outputs_vs_vision/circular_shift_fmri_vision.npz",
                "outputs_vs_llm/circular_shift_fmri_llm.npz"],
       "cka":  ["outputs_vs_vision/circular_shift_fmri_vision_cka.npz",
                "outputs_vs_llm/circular_shift_fmri_llm_cka.npz"]}
METRIC_LABEL = {"mknn": r"$\mathbf{mKNN}$", "cka": r"$\mathbf{CKA}$"}
MOD_LABEL = {"vision": "vision", "llm": "language"}
# plot_fmri_scaling.group_title upper-cases anything outside C.VISION, which turns the
# LLM families into BLOOM / OPENLLAMA. House capitalisation instead, matching
# decodability_rebuild/_llm_grids.DISPLAY.
LANG_DISPLAY = {"bloom": "BLOOMZ", "openllama": "OpenLLaMA", "llama": "LLaMA"}
# One colour per modality, carried by the curves, the null cloud and the heading, so
# the heading reads as belonging to the panels under it. Okabe-Ito blue / vermillion.
BAND_COLOR = {"vision": "#0072B2", "llm": "#D55E00"}
BAND_LABEL = {"vision": "VISION", "llm": "LANGUAGE"}


def title_of(name, modality):
    return group_title(name) if modality == "vision" else LANG_DISPLAY.get(name, name)


def dress(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_C, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.7)


def collect(paths):
    """Merge several sweeps. -> shifts, [(group, modality, cells), ...]."""
    shifts, out = None, []
    for p in paths:
        sh, _swept, _excl, modality, groups = load(_HERE / p)
        if shifts is None:
            shifts = sh
        elif not np.array_equal(shifts, sh):
            raise SystemExit(f"{p}: different shift grid; cannot share one null mask")
        for g in groups:
            if len(groups[g]) >= 2:
                out.append((g, modality, groups[g]))
    return shifts, out


def colour_of(name, modality, i, n):
    """Colour by MODALITY, not by family: the panels group into vision and language and
    the headings say so, so a per-family palette would only compete with that."""
    return BAND_COLOR.get(modality, INK)


def params_of(name, modality, cells):
    if modality == "vision":
        return [C.VISION_PARAMS.get((name, k.rsplit("-", 1)[1])) for _, k, _, _ in cells]
    return [C.LLM_PARAMS.get(k) for _, k, _, _ in cells]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="mknn", choices=list(NPZ))
    ap.add_argument("--npz", nargs="+", default=None,
                    help="override the sweeps to read (default: whichever pair the "
                         "chosen --metric names)")
    ap.add_argument("--tail-min", type=int, default=0,
                    help="0 (DEFAULT) = every non-identity rotation, the full group C_n. "
                         "100 reproduces the far-tail null of fmri_<mod>_scaling_view.png.")
    ap.add_argument("--max-lines", type=int, default=400)
    ap.add_argument("--hero", action="store_true",
                    help="append the net-alignment panel (obs - mean null) on a shared "
                         "log-parameter x. Off by default: the satellites are the figure.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    npz = args.npz or NPZ[args.metric]
    shifts, fams = collect(npz)
    if not fams:
        raise SystemExit("no family has >=2 sizes to plot")
    i0 = int(np.where(shifts == 0)[0][0])
    tail = (shifts != 0) if args.tail_min <= 0 else (np.abs(shifts) >= args.tail_min)
    tail_idx = np.where(tail)[0]
    rng = np.random.default_rng(0)
    sub = rng.choice(tail_idx, min(args.max_lines, len(tail_idx)), replace=False)

    plt.rcParams.update(THEME)
    nrow = 2
    ncol = int(np.ceil(len(fams) / nrow)) + (1 if args.hero else 0)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 6.6), squeeze=False,
                            gridspec_kw=dict(wspace=0.38, hspace=0.30))
    if args.hero:
        for r in range(nrow):
            axs[r][-1].remove()
        hero = fig.add_subplot(1, ncol, ncol)
    else:
        hero = None

    stats, placed = {}, []
    for i, (name, modality, cells) in enumerate(fams):
        # COLUMN-major, so the two LLM families land in a column of their own rather
        # than being wrapped in beside the vision ones.
        ax = axs[i % nrow][i // nrow]
        placed.append((ax, modality, i // nrow))
        x = np.arange(len(cells), dtype=float)
        M = np.stack([c[3].mean(axis=0) for c in cells])       # (n_size, W) subject-mean
        cc = colour_of(name, modality, i, len(fams))

        obs = M[:, i0]
        nul = M[:, tail].mean(axis=1)
        lo = np.percentile(M[:, tail], 5, axis=1)
        hi = np.percentile(M[:, tail], 95, axis=1)
        # clearing the null: z each size against its own rotations, average over the
        # family, rank d=0 against every rotation. plot_femba_dinov2's statistic.
        Zc = (M - nul[:, None]) / M[:, tail].std(axis=1, ddof=1)[:, None]
        zbar = float(Zc[:, i0].mean())
        pval = (1 + int((Zc[:, tail].mean(axis=0) >= zbar).sum()) + 1) / (int(tail.sum()) + 2)
        s0 = np.polyfit(x, obs, 1)[0]
        st = np.array([np.polyfit(x, M[:, di], 1)[0] for di in tail_idx])
        p_slope = (1 + int((st >= s0).sum())) / (len(st) + 1)
        stats[name] = (obs, nul, pval, params_of(name, modality, cells), cc,
                       zbar, p_slope)

        dress(ax)
        seg = np.stack([np.broadcast_to(x, M[:, sub].T.shape), M[:, sub].T], axis=-1)
        ax.add_collection(LineCollection(seg, colors=cc, lw=0.5, alpha=0.05, zorder=2))
        ax.fill_between(x, lo, hi, color="#8494A8", alpha=0.35, lw=0, zorder=3)
        ax.plot(x, obs, color=cc, lw=2.2, marker="o", ms=6, zorder=6)
        ax.set_ylabel(title_of(name, modality), fontsize=23, color=INK)
        ax.set_xticks(x)
        # The sweep stores "SMALL\n22M" and "560M\n0.6B": a name line and a parameter
        # line. Keep only the parameter line — the names add nothing next to the counts,
        # and on the LLM side the name line IS a parameter count, so showing both
        # printed "560M" over "0.6B" for the same model.
        ax.set_xticklabels([c[2].split("\n")[-1] for c in cells], fontsize=19)
        # was -0.28/+0.28: a quarter of the panel was empty on each side. Kept
        # non-zero so the end markers do not sit on the frame.
        ax.set_xlim(-0.16, len(cells) - 0.84)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="y", labelsize=19)
    for j in range(len(fams), nrow * (ncol - (1 if args.hero else 0))):
        axs[j % nrow][j // nrow].set_visible(False)

    # Everything below needs settled limits and a renderer.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    for mod in dict.fromkeys(m for _, m, _ in placed):
        cols = {c for _, m, c in placed if m == mod}
        bbs = [a.get_tightbbox(rend).transformed(inv)
               for a, _, c in placed if c in cols]
        gx0, gx1 = min(b.x0 for b in bbs), max(b.x1 for b in bbs)
        gy1 = max(b.y1 for b in bbs)
        fig.text((gx0 + gx1) / 2, gy1 + 0.016, BAND_LABEL.get(mod, mod.upper()),
                 ha="center", va="bottom", fontsize=26,
                 color=BAND_COLOR.get(mod, INK))

    if hero is not None:
        dress(hero)
        net_all = []
        for name, modality, cells in fams:
            obs, nul, pval, pm, cc, zbar, p_slope = stats[name]
            if any(p is None for p in pm):
                continue
            net_all.append(obs - nul)
            hero.plot(pm, obs - nul, color=cc, lw=3.0, marker="o", ms=13, zorder=6,
                      label=title_of(name, modality))
        net = np.concatenate(net_all)
        ptp = float(np.ptp(net))
        hero.set_ylim(net.min() - 0.16 * ptp, net.max() + 0.30 * ptp)
        hero.set_xscale("log")
        hero.xaxis.set_major_locator(FixedLocator([30, 300, 3000]))
        hero.xaxis.set_minor_locator(FixedLocator([]))
        hero.xaxis.set_major_formatter(FixedFormatter(["30M", "300M", "3B"]))
        hero.tick_params(axis="x", labelsize=21)
        hero.yaxis.set_major_locator(MaxNLocator(nbins=6))
        hero.yaxis.tick_right()
        hero.yaxis.set_label_position("right")
        hero.tick_params(axis="y", labelsize=21)
        hero.set_xlabel("model size", fontsize=29, color=INK, labelpad=10)
        hero.set_ylabel(METRIC_LABEL[args.metric], fontsize=34, color=INK,
                        labelpad=10)
        hero.legend(fontsize=15, loc="lower right", framealpha=0.95, facecolor="white",
                    edgecolor="#BBBBBB", borderpad=0.7, labelspacing=0.55)

    stem = ("fmri_scaling_satellites" if hero is None else "fmri_scaling_null")
    out = Path(args.out) if args.out else \
        (_HERE / npz[0]).parent / f"{stem}_{args.metric}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote {out}\n")
    ttl = "full C_n" if args.tail_min <= 0 else f"|d| >= {args.tail_min}"
    print(f"null = {ttl}   ({int(tail.sum())} rotations)   subject-mean curves")
    print("p (annotated) = clears the null;  slope p = rises with size\n")
    print(f"{'family':14}{'net = obs - null':>42}{'mean z':>8}{'p':>9}{'slope p':>9}")
    for name, modality, cells in fams:
        obs, nul, pval, pm, cc, zbar, p_slope = stats[name]
        print(f"{name:14}{' '.join(f'{v:+.5f}' for v in obs - nul):>42}"
              f"{zbar:+8.2f}{pval:9.4f}{p_slope:9.4f}")


if __name__ == "__main__":
    main()
