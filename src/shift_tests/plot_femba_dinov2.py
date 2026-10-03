#!/usr/bin/env python
"""
plot_femba_dinov2.py — the null-correction argument, laid out like PRH Figure 3.

Hero panel (left): net alignment, obs - mean(null), one curve per DINOv2 size.
Satellites (2x2, right): the same cells uncalibrated -- the true pairing over the cloud
of every circular-shift null curve, with the 5-95 percentile band.

The pairing is the argument. In every satellite the observed line climbs with FEMBA size,
which is the scaling result as usually reported -- but the null cloud climbs with it,
because a larger model's embedding space is more clustered and that lifts k-NN overlap
against ANY temporally smooth signal, a rotated one included. The hero is what is left
once that is removed.

Usage:
    python plot_femba_dinov2.py
    python plot_femba_dinov2.py --eeg-model luna --arch vjepa2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))
import _common as C  # noqa: E402

PANEL_BG = "#F7F7F7"      # barely off-white: enough to separate the panel from the
                          # page, not enough to compete with the series colours
GRID_C = "#DDDDDD"        # the grid can no longer be white against this ground
INK = "#33475B"           # slate for the hero's axis labels
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


def dress(ax, y_side="left"):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID_C, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for side in (y_side, "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-model", default="femba", choices=list(C.EEG))
    ap.add_argument("--arch", default="dinov2", choices=list(C.VISION))
    ap.add_argument("--tail-min", type=int, default=0,
                    help="0 (default) = every non-identity rotation, the full group C_n.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    m, arch = args.eeg_model, args.arch

    z = np.load(_HERE / "outputs_vs_vision" / m / f"circular_shift_full__{m}.npz",
                allow_pickle=True)
    shifts = z["shifts"]
    i0 = int(np.where(shifts == 0)[0][0])
    tail = (shifts != 0) if args.tail_min <= 0 else (np.abs(shifts) >= args.tail_min)
    sizes = [s for s in C.EEG[m]["sizes"] if any(k.startswith(f"{s}__") for k in z.files)]
    vsizes = C.VISION[arch]["sizes"][:4]
    x = np.arange(len(sizes))
    ticks = [f"{C.PARAMS[m][s]}M" for s in sizes]
    disp = C.VISION[arch]["display"]

    plt.rcParams.update(THEME)
    # Nested rather than one 2x3: a single gridspec forces the same wspace between the
    # two satellite columns and between the satellites and the hero, and those two gaps
    # want different sizes — the satellites only need room for a y tick label, the hero
    # carries its axis on the far side and needs almost none.
    # The gutter between the satellite columns is set by the enlarged y label plus its
    # tick labels, not by taste: below ~0.30 "DINOv2-base" runs into the left panel.
    # Height and the satellite/hero width split do the shaping instead — taller canvas
    # and a narrower satellite block bring the four panels close to square.
    fig = plt.figure(figsize=(16.0, 8.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.62], wspace=0.06,
                             left=0.045, right=0.945, top=0.96, bottom=0.11)
    gsat = outer[0].subgridspec(2, 2, wspace=0.32, hspace=0.26)
    sat = [fig.add_subplot(gsat[i // 2, i % 2]) for i in range(4)]
    hero = fig.add_subplot(outer[1])

    col = C.VISION_SIZE_COLORS[arch]
    stats = {}
    for i, vs in enumerate(vsizes):
        ax = sat[i]
        cc = col.get(vs, plt.cm.viridis(0.15 + 0.7 * i / 3))
        obs, nul, zc, M, lo, hi, pcell = [], [], [], [], [], [], []
        for s in sizes:
            mm = z[f"{s}__{arch}-{vs}__curve"].mean(axis=0)
            t = mm[tail]
            obs.append(mm[i0]); nul.append(t.mean())
            lo.append(np.percentile(t, 5)); hi.append(np.percentile(t, 95))
            zc.append((mm - t.mean()) / t.std(ddof=1))
            M.append(mm)
            # p for THIS cell alone, the same test the grid figures put on each dot.
            # The panel annotation below is a different quantity: an aggregate over
            # the three EEG sizes, which can clear when one of its cells does not.
            pcell.append((1 + int((t >= mm[i0]).sum()) + 1) / (int(tail.sum()) + 2))
        M = np.vstack(M)
        Zc = np.vstack(zc)
        zbar = float(Zc[:, i0].mean())
        null = Zc[:, tail].mean(axis=0)
        pval = (1 + int((null >= zbar).sum()) + 1) / (int(tail.sum()) + 2)
        stats[vs] = (np.array(obs), np.array(nul), zbar, pval,
                     np.array(pcell) <= args.alpha)

        dress(ax)
        seg = np.stack([np.broadcast_to(x, M[:, tail].T.shape), M[:, tail].T], axis=-1)
        ax.add_collection(LineCollection(seg, colors=cc, lw=0.5, alpha=0.02, zorder=2))
        ax.fill_between(x, lo, hi, color="#8494A8", alpha=0.35, lw=0, zorder=3)
        ax.plot(x, obs, color=cc, lw=2.2, zorder=6)
        sig = np.array(pcell) <= args.alpha
        for mask, fc in ((sig, cc), (~sig, "white")):
            if mask.any():
                ax.plot(np.asarray(x)[mask], np.asarray(obs)[mask], ls="none",
                        marker="o", ms=8, mfc=fc, mec=cc, mew=1.8, zorder=7)
        ax.set_ylabel(f"{disp}-{vs}", fontsize=23, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(ticks, fontsize=18)
        ax.set_xlim(-0.28, len(sizes) - 0.72)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="y", labelsize=18)
        # No panel annotation: the aggregate p over the three EEG sizes is a different
        # test from the marker fills, and showing both invited them to be read as one.
        # It is still computed and printed to stdout.
        if i >= 2:
            ax.set_xlabel(f"{C.DISPLAY[m]} size", fontsize=27, color=INK, labelpad=6)

    dress(hero, y_side="right")
    for i, vs in enumerate(vsizes):
        obs, nul, zbar, pval, sig = stats[vs]
        cc = col.get(vs, plt.cm.viridis(0.15 + 0.7 * i / 3))
        hero.plot(x, obs - nul, color=cc, lw=3.0, zorder=6, label=f"{disp.lower()} {vs}")
        for mask, fc in ((sig, cc), (~sig, "white")):
            if mask.any():
                hero.plot(x[mask], (obs - nul)[mask], ls="none", marker="o", ms=14,
                          mfc=fc, mec=cc, mew=2.4, zorder=7)
    net = np.concatenate([stats[v][0] - stats[v][1] for v in vsizes])
    pad = 0.16 * float(np.ptp(net))
    hero.set_ylim(net.min() - pad, net.max() + pad)
    hero.set_xticks(x); hero.set_xticklabels(ticks, fontsize=25)
    hero.set_xlim(-0.22, len(sizes) - 0.78)
    hero.yaxis.set_major_locator(MaxNLocator(nbins=6))
    hero.yaxis.tick_right()
    hero.yaxis.set_label_position("right")
    hero.tick_params(axis="y", labelsize=25)
    hero.set_xlabel(f"{C.DISPLAY[m]} size", fontsize=31, color=INK, labelpad=10)
    # Bare metric name, matching grid_vision_net.png: the caption carries that this is
    # mKNN minus the cyclic-null mean, not raw mKNN.
    hero.set_ylabel(r"$\mathbf{mKNN}$", fontsize=36, color=INK, labelpad=10)
    hero.legend(fontsize=19, loc="upper center", framealpha=0.95, facecolor="white",
                edgecolor="#BBBBBB", borderpad=0.7, labelspacing=0.55)

    key = [plt.Line2D([], [], color="#555555", lw=0, marker="o", ms=13,
                      mfc="#555555", label=rf"$p \leq$ {args.alpha:g}"),
           plt.Line2D([], [], color="#555555", lw=0, marker="o", ms=13,
                      mfc="white", mew=2.0, label=rf"$p >$ {args.alpha:g}")]
    fig.legend(handles=key, loc="lower center", ncol=2, fontsize=18, frameon=False,
               bbox_to_anchor=(0.5, -0.045), handletextpad=0.4, columnspacing=2.4)

    out = Path(args.out) if args.out else \
        _HERE / "outputs_vs_vision" / f"{m}_{arch}_null.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}\n")
    print(f"{'vision':20s}{'obs':>26s}{'null':>26s}{'net':>26s}{'z':>7s}{'p':>8s}")
    for vs in vsizes:
        o, n, zb, pv, sg = stats[vs]
        print(f"{arch+'-'+vs:20s}{' '.join(f'{v:.4f}' for v in o):>26s}"
              f"{' '.join(f'{v:.4f}' for v in n):>26s}"
              f"{' '.join(f'{v:.5f}' for v in (o-n)):>26s}{zb:+7.2f}{pv:8.3f}")


if __name__ == "__main__":
    main()
