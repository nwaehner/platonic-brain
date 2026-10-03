#!/usr/bin/env python
"""
plot_grid_vision_net.py — net alignment for every EEG model x every partner model.

Rows = partner family (four vision architectures, then two LLM families), columns = EEG
family, x = EEG size, one line per partner size.
y = Delta = T(0) - mean over every non-identity rotation of T(d): the alignment that
survives the cyclic null, with the null itself not drawn because it is the zero line.

Each ROW gets its own y-axis, anchored at zero and shared across the five EEG families.
Rows differ by more than an order of magnitude -- DINOv2 reaches 0.068 in CKA while every
other partner stays near 0.01 -- so one axis for the whole grid flattened five of the six
rows into a line. The cost is that magnitudes are no longer comparable BETWEEN rows by
eye; read the tick values, not the heights, when comparing partners. `--shared-y` restores
the single common axis, which is the better choice when the point being made is how small
the net alignment is relative to the raw metric.

Filled marker = the cell clears alpha against its own rotation null; hollow = it does not.
Vision rows are drawn in a cool palette and language rows in a warm one, so the two
modalities separate at a glance; within a row, colour still encodes partner size rank.

TWO METRICS, ONE PIPELINE.
  --metric mknn (default) reads this package's own sweeps, which are PER-SUBJECT (6 curves
      averaged after the fact) on each EEG model's NATIVE window grid.
  --metric cka reads outputs_cka/, written by circular_shift_cka_full.py, which mirrors
      the mKNN sweep exactly: per subject, native grid, complete cyclic null, layer pair
      re-maximised at every rotation. The two metrics are therefore the same cells
      measured two ways, and differ only in the metric.

Usage:
    python plot_grid_vision_net.py
    python plot_grid_vision_net.py --metric cka
    python plot_grid_vision_net.py --tail-min 100 --root outputs_vs_vision_excl10
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))
import _common as C  # noqa: E402

PANEL_BG, GRID_C, INK = "#F7F7F7", "#DDDDDD", "#33475B"
KEY_X = 0.885
KEY_LINE = 0.013   # length of the key's line sample, in figure fractions
ARCHS = ["dinov2", "videomae", "videomae_ft", "vjepa2"]
LLMS = ["bloom", "openllama"]
MODELS = ["femba", "luna", "neurolm", "steegformer", "reve"]
LLM_DISPLAY = {"bloom": "BLOOMZ", "openllama": "OpenLLaMA"}
# Colour by size RANK, not by name: _common.VISION_SIZE_COLORS gives videomae_ft
# firebrick/darkorchid/darkorange while the other three run blue/green/orange/violet,
# so "the blue one" would mean the smallest in three rows and the middle one in the
# fourth. C.VISION[arch]["sizes"] is ascending, so position is the rank.
RANK_COLORS = ["steelblue", "seagreen", "darkorange", "mediumpurple"]
# Language rows get a warm ramp so the modality is legible before the row label is read;
# five entries because BLOOMZ has five sizes. Rank still runs dark -> light.
LANG_COLORS = ["#67001F", "#B2182B", "#D6604D", "#E58268", "#F4A582"]


def row_spec(name):
    """-> (member sizes, display name, palette) for a vision arch or an LLM family."""
    if name in C.VISION:
        return C.VISION[name]["sizes"], C.VISION[name]["display"], RANK_COLORS
    return C.LLM[name], LLM_DISPLAY.get(name, name.upper()), LANG_COLORS


def member_label(name, member):
    """Legend text for one partner size. Vision keeps its size word; an LLM stem is cut
    to its size token, so "open_llama_13b" reads 13B rather than filling the margin."""
    return member if name in C.VISION else re.split(r"[-_]", member)[-1].upper()


THEME = {
    "figure.dpi": 150, "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "mathtext.rm": "cmr10",
    "axes.formatter.use_mathtext": True,
    "axes.facecolor": PANEL_BG, "axes.axisbelow": True,
    "axes.edgecolor": GRID_C, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "grid.color": GRID_C, "grid.linewidth": 1.0,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.major.size": 0, "ytick.major.size": 0,
}


def load_stats(metric, root, tail_min, alpha):
    """-> stats[(eeg_model, eeg_size, row, member)] = (delta, p), sizes_of, ylabel."""
    stats, sizes_of = {}, {}

    def add(curve, key, i0, tail):
        c = np.asarray(curve, float)
        t = c[tail]
        p_ = (1 + int((t >= c[i0]).sum()) + 1) / (int(tail.sum()) + 2)
        stats[key] = (c[i0] - t.mean(), p_)

    if metric == "mknn":
        for sub, rows in ((root, ARCHS), (root.parent / "outputs_vs_llm", LLMS)):
            for m in MODELS:
                f = sub / m / f"circular_shift_full__{m}.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                sh = z["shifts"]
                i0 = int(np.where(sh == 0)[0][0])
                tail = (sh != 0) if tail_min <= 0 else (np.abs(sh) >= tail_min)
                sizes_of.setdefault(m, [s for s in C.EEG[m]["sizes"]
                                        if any(k.startswith(f"{s}__") for k in z.files)])
                for row in rows:
                    members, _, _ = row_spec(row)
                    for mem in members:
                        tag = f"{row}-{mem}" if row in C.VISION else mem
                        for sz in sizes_of[m]:
                            k = f"{sz}__{tag}__curve"
                            if k in z.files:
                                # per-subject curves, averaged before the null is taken
                                add(z[k].mean(axis=0), (m, sz, row, mem), i0, tail)
        return stats, sizes_of, r"$\mathbf{mKNN}$"

    # CKA: circular_shift_cka_full.py, same pipeline as the mKNN sweeps — per subject,
    # native grid, one file per EEG family holding both vision and language partners.
    for m in MODELS:
        f = _HERE / "outputs_cka" / m / f"circular_shift_cka__{m}.npz"
        if not f.exists():
            print(f"[skip] {m}: {f.name} missing — run circular_shift_cka_full.py")
            continue
        z = np.load(f, allow_pickle=True)
        sh = z["shifts"]
        i0 = int(np.where(sh == 0)[0][0])
        tail = (sh != 0) if tail_min <= 0 else (np.abs(sh) >= tail_min)
        sizes_of[m] = [s for s in C.EEG[m]["sizes"]
                       if any(k.startswith(f"{s}__") for k in z.files)]
        for row in ARCHS + LLMS:
            members, _, _ = row_spec(row)
            for mem in members:
                tag = f"{row}-{mem}" if row in C.VISION else mem
                for sz in sizes_of[m]:
                    k = f"{sz}__{tag}__curve"
                    if k in z.files:
                        # per-subject curves, averaged before the null is taken, as
                        # the mKNN branch does
                        add(z[k].mean(axis=0), (m, sz, row, mem), i0, tail)
    return stats, sizes_of, r"$\mathbf{CKA}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="mknn", choices=["mknn", "cka"])
    ap.add_argument("--tail-min", type=int, default=0,
                    help="0 (default) = every non-identity rotation, the full group C_n.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--shared-y", action="store_true",
                    help="one y axis for the whole grid instead of one per row. Makes "
                         "magnitudes comparable between partners at the cost of "
                         "flattening every row but the largest.")
    ap.add_argument("--root", default="outputs_vs_vision")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = _HERE / args.root

    ROWS = ARCHS + LLMS
    stats, sizes_of, ylab = load_stats(args.metric, root, args.tail_min, args.alpha)
    models = [m for m in MODELS if m in sizes_of]
    hi = max(v[0] for v in stats.values())

    plt.rcParams.update(THEME)
    fig, axes = plt.subplots(len(ROWS), len(models), figsize=(17.5, 2.45 * len(ROWS)),
                             squeeze=False, sharey=args.shared_y)
    top, bot = 0.955, 0.075
    fig.subplots_adjust(left=0.105, right=0.875, top=top, bottom=bot,
                        wspace=0.16, hspace=0.30)

    n_sig = n_tot = 0
    for r, row in enumerate(ROWS):
        members, disp, palette = row_spec(row)
        row_vals = [v[0] for k, v in stats.items() if k[2] == row]
        row_hi = max(row_vals) if row_vals else hi
        for c, m in enumerate(models):
            ax = axes[r][c]
            ax.grid(True, color=GRID_C, lw=1.0, zorder=0)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color("black")
                ax.spines[side].set_linewidth(0.7)
            sizes = sizes_of[m]
            x = np.arange(len(sizes))
            for i, mem in enumerate(members):
                cells = [stats.get((m, sz, row, mem)) for sz in sizes]
                if not any(cells):
                    continue
                y = np.array([cc[0] if cc else np.nan for cc in cells])
                cc_ = palette[i % len(palette)]
                ax.plot(x, y, color=cc_, lw=2.4, zorder=6, label=member_label(row, mem))
                sig = np.array([bool(cc and cc[1] <= args.alpha) for cc in cells])
                n_sig += int(sig.sum()); n_tot += int(sum(1 for cc in cells if cc))
                for mask, fc in ((sig, cc_), (~sig, "white")):
                    if mask.any():
                        ax.plot(x[mask], y[mask], ls="none", marker="o", ms=8,
                                mfc=fc, mec=cc_, mew=1.8, zorder=7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{C.PARAMS[m][s]}M" for s in sizes], fontsize=15)
            ax.set_xlim(-0.28, len(sizes) - 0.72)
            top_r = hi if args.shared_y else row_hi
            ax.set_ylim(-0.04 * top_r, top_r * 1.10)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="y", labelsize=19)
            # sharey would hide these for us; without it, do it by hand so only the
            # left column carries the numbers.
            if c > 0:
                ax.tick_params(axis="y", labelleft=False)
            if r == 0:
                ax.set_title(C.DISPLAY[m], fontsize=19, color=INK, pad=12)
            if c == 0:
                ax.set_ylabel(disp, fontsize=20, color=INK, labelpad=10)
            if r == len(ROWS) - 1:
                ax.set_xlabel("EEG size", fontsize=22, color=INK)
        # Key for this row, in the right margin. Drawn as plain text rather than a
        # matplotlib legend: a legend indents its labels past the handle by an amount
        # that depends on the title's width, so six legends with titles of different
        # lengths came out ragged. Here every line starts at the same x, and the size
        # name carries its curve's colour instead of sitting beside a line sample.
        yc = top - (top - bot) / len(ROWS) * (r + 0.5)
        step = 0.0225
        y0 = yc + step * len(members) / 2
        fig.text(KEY_X, y0, disp, ha="left", va="center", fontsize=16, color=INK)
        for i, mem in enumerate(members):
            y = y0 - step * (i + 1)
            cc_ = palette[i % len(palette)]
            # line sample, then the name in the same colour. Both start at a fixed x,
            # so the six keys line up regardless of how long the row title is.
            fig.add_artist(plt.Line2D([KEY_X, KEY_X + KEY_LINE], [y, y],
                                      transform=fig.transFigure, color=cc_, lw=2.4,
                                      solid_capstyle="butt"))
            fig.text(KEY_X + KEY_LINE + 0.006, y, member_label(row, mem),
                     ha="left", va="center", fontsize=16, color=cc_)

    fig.text(0.028, 0.53, ylab, fontsize=36, color=INK, rotation=90,
             va="center", ha="center")
    handles = [plt.Line2D([], [], color="#555555", lw=0, marker="o", ms=15,
                          mfc="#555555", label=rf"$p \leq$ {args.alpha:g}"),
               plt.Line2D([], [], color="#555555", lw=0, marker="o", ms=15,
                          mfc="white", mew=2.2, label=rf"$p >$ {args.alpha:g}")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=19,
               frameon=False, bbox_to_anchor=(0.5, -0.012), handletextpad=0.4,
               columnspacing=2.4)

    out = Path(args.out) if args.out else _HERE.parent / f"grid_net_{args.metric}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print(f"{n_tot} cells, {n_sig} clear alpha={args.alpha:g}; "
          f"Delta spans {min(v[0] for v in stats.values()):.5f} .. {hi:.5f}")


if __name__ == "__main__":
    main()
