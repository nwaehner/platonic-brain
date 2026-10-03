#!/usr/bin/env python
"""
plot_native_grid.py — the no-resampling variant of the EEG x vision panels.

Reads cache/native_{mknn,cka}_<arch>.npz (compute_native_grid_vision.py), where every
EEG model was paired with the vision model tiled at exactly its own window length and
nothing was averaged onto a shared grid. Writes two figures, neither of which overwrites
the shared-grid originals:

  cyclic_eeg_x_<arch>__nativegrid_perfamily.png
      2 x 5, metric by EEG family. RAW y, not centred. This is the honest reading of a
      native-grid run: W differs between families (2160 / 1800 / 1350 / 1080), so mKNN
      with k=5 and CKA on a W x W Gram are not in comparable units ACROSS families —
      but within a family every model shares one grid, so the within-family contrast is
      clean and each panel has its own scale.

  cyclic_eeg_x_<arch>__nativegrid.png
      the familiar 1 x 2, all 14 models, family-centred — for direct comparison against
      the shared-grid cyclic_eeg_x_<arch>.png. Centring absorbs a per-family LEVEL
      difference (grid is a function of architecture) but not a per-family SCALE
      difference, which is exactly what differing W introduces here. Read it as a
      robustness check on the ordering, not as a measurement.

y is the same calibrated quantity as every other figure in the set: obs - mean(cyclic
null), the excess over the null at that grid's own W-1 rotations (make_all_figures.stats;
--yq cal switches to the paper's clamped s_cal).

Usage:
    python plot_native_grid.py
    python plot_native_grid.py --arch vjepa2 --yq cal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import _common as C  # noqa: E402
from plot_panel_b import CACHE, OUT, THEME, FAMILY_COLOR, FAMILY_MARKER, load_x  # noqa: E402
from plot_panel_b_cyclic import calibrate_cyclic  # noqa: E402
from make_all_figures import ALPHA, _place_labels, draw, note_free_corner  # noqa: E402
from make_partner_figures import _bold, _row_heading  # noqa: E402
import _llm_grids as G  # noqa: E402

# Where each modality's native-grid figures land, and how a group is displayed.
FOLDER = {"video": "eeg_vs_video_nativegrid", "language": "eeg_vs_language_nativegrid"}
GROUPS = {"video": ["dinov2", "videomae", "videomae_ft", "vjepa2"],
          "language": ["bloom", "openllama"]}


def display(modality, group):
    return C.VISION[group]["display"] if modality == "video" else G.DISPLAY[group]


def vals(arch, metric, yq="excess"):
    """-> y (n_eeg,), n_sig, n_cells, eeg_keys, fam, W_of. Mean over the arch's sizes."""
    f = CACHE / f"native_{metric}_{arch}.npz"
    if not f.exists():
        raise SystemExit(f"{f} missing — run compute_native_grid_vision.py --arch {arch}")
    d = np.load(f, allow_pickle=True)
    ek = [str(k) for k in d["eeg_keys"]]
    vk = [str(k) for k in d["vis_keys"]]
    fam = np.array([str(x) for x in d["eeg_family"]])
    y, n_sig, n_cell = [], 0, 0
    for k in ek:
        per = []
        for v in vk:
            c = np.asarray(d[f"{k}__{v}__curves"], float)
            o, tau, cal, p = calibrate_cyclic(c, ALPHA)
            per.append(cal if yq == "cal" else o - float(np.mean(c[1:])))
            n_sig += p <= ALPHA
            n_cell += 1
        y.append(np.mean(per))
    return (np.array(y), n_sig, n_cell, ek, fam,
            {k: int(w) for k, w in zip(ek, d["W_of"])})


def per_family(arch, yq, keys_x, x_mv, modality="video"):
    """2 x 5: metric by family, raw (uncentred) y, each panel on its own scale."""
    fams = ["femba", "luna", "neurolm", "reve", "steegformer"]
    fig, axs = plt.subplots(2, len(fams), figsize=(4.0 * len(fams), 9.2), squeeze=False)
    out, pend = [], []
    for row, metric in enumerate(["mknn", "cka"]):
        y, ns, nc, ek, fam, W_of = vals(arch, metric, yq)
        lab = {"mknn": "mKNN", "cka": "CKA"}[metric]
        for col, f in enumerate(fams):
            ax = axs[row][col]
            m = fam == f
            xs, ys = x_mv[[keys_x.index(k) for k in ek]][m], y[m]
            labs = [f"{C.PARAMS[k.rsplit('-', 1)[0]][k.rsplit('-', 1)[1]]}M"
                    for k in np.array(ek)[m]]
            ax.scatter(xs, ys, c=FAMILY_COLOR[f], marker=FAMILY_MARKER[f], s=110,
                       edgecolor="white", linewidth=1.2, zorder=3)
            if len(xs) > 1:
                b, a = np.polyfit(xs, ys, 1)
                xx = np.linspace(xs.min(), xs.max(), 20)
                ax.plot(xx, a + b * xx, color="0.25", lw=1.6, zorder=2)
            ax.axhline(0, color="0.85", lw=0.8, zorder=0)
            pend.append((ax, xs, ys, labs, np.array([f] * int(m.sum()))))
            W = W_of[np.array(ek)[m][0]]
            if row == 0:
                ax.set_title(rf"${_bold(C.DISPLAY[f])}$" "\n"
                             rf"$W={W}$, {C.EEG_WINDOWS[f]['win_sec']} s windows",
                             fontsize=13)
            if col == 0:
                ax.set_ylabel(rf"$\mathbf{{{lab}}}$", fontsize=16)
            ax.set_xlabel(r"$\mathbf{NICE}$ decodability" if row == 1 else "", fontsize=13)
            ax.margins(0.22)
            ax.grid(zorder=0)
            out.append((lab, f, len(xs)))
    fig.suptitle(rf"$\mathbf{{EEG}}\ \mathbf{{\times}}\ {_bold(display(modality, arch))}$"
                 "\nnative grid, no resampling - y raw, one scale per family",
                 fontsize=19, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    for ax, xs, ys, labs, ff in pend:          # limits are settled now
        _place_labels(ax, xs, ys, labs, ff, fontsize=10.0)
    p = OUT / FOLDER[modality] / f"cyclic_eeg_x_{arch}__nativegrid_perfamily.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def pooled(arch, yq, keys_x, x_mv, cells=False, modality="video"):
    """1 x 2, family-centred — same grammar as the shared-grid figure, for comparison."""
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.6))
    res = []
    for ax, metric in zip(axs, ["mknn", "cka"]):
        y, ns, nc, ek, fam, _ = vals(arch, metric, yq)
        idx = [keys_x.index(k) for k in ek]
        lab = {"mknn": "mKNN", "cka": "CKA"}[metric]
        labs = [f"{C.PARAMS[k.rsplit('-', 1)[0]][k.rsplit('-', 1)[1]]}M" for k in ek]
        r, p = draw(ax, x_mv[idx], y, fam, None,
                    r"$\mathbf{NICE}$ decodability", rf"$\mathbf{{{lab}}}$",
                    note=(rf"{ns}/{nc} cells clear $\alpha$={ALPHA}" if cells else None),
                    labels=labs, show_rho=False, s=70, labelsize=19, label_fs=10.5,
                    note_fs=14,
                    note_corner=note_free_corner(x_mv[idx], y, fam,
                                                 *((0.44, 0.30) if cells else (0.28, 0.16))),
                    axis_contours=True)
        res.append((lab, r, p, ns, nc))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03),
               fontsize=17, markerscale=1.9, handletextpad=0.5, columnspacing=1.8)
    title_suffix = "" if (modality == "language" and arch == "openllama") else " - native grid"
    fig.suptitle(rf"$\mathbf{{EEG}}\ \mathbf{{\times}}\ "
                 rf"{_bold(display(modality, arch))}${title_suffix}",
                 fontsize=21, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    p = OUT / FOLDER[modality] / (f"cyclic_eeg_x_{arch}__nativegrid"
                                  + ("__cells" if cells else "") + ".png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p, res


def stacked(rows, keys_x, x_mv, yq="excess", cells=False, stem=None):
    """One figure, one ROW per partner, columns mKNN | CKA — the native-grid twin of
    make_partner_figures.stacked, reading the no-resampling caches instead.

    rows: [(modality, group), ...]. Every panel plots the same 14 EEG models on the same
    x, so the family key is drawn once under the grid; the per-point parameter counts
    stay in every panel, since colour and marker only encode family.
    """
    # Tighter per-row height and row gap once the stack is deep: at 5.6in x 4 rows the
    # figure is 22in tall and mostly gutter.
    deep = len(rows) > 2
    # h_pad has to hold the row heading plus clearance on both sides: the headings sit
    # in this gap, and squeezing it is what put them on the panels.
    row_h, h_pad = (4.7, 5.2) if deep else (5.6, 4.5)
    fig, axs = plt.subplots(len(rows), 2, figsize=(12.4, row_h * len(rows)),
                            squeeze=False)
    res = []
    for i, (modality, group) in enumerate(rows):
        for ax, metric in zip(axs[i], ["mknn", "cka"]):
            y, ns, nc, ek, fam, _ = vals(group, metric, yq)
            idx = [keys_x.index(k) for k in ek]
            lab = {"mknn": "mKNN", "cka": "CKA"}[metric]
            labs = [f"{C.PARAMS[k.rsplit('-', 1)[0]][k.rsplit('-', 1)[1]]}M" for k in ek]
            # Every panel shares the same x. Printing it under all of them costs the
            # vertical room the next row's heading needs, so it goes on the last row
            # only once the stack is deep.
            xlab = (r"$\mathbf{NICE}$ decodability"
                    if (not deep or i == len(rows) - 1) else "")
            r, p = draw(ax, x_mv[idx], y, fam, None,
                        xlab, rf"$\mathbf{{{lab}}}$",
                        note=(rf"{ns}/{nc} cells clear $\alpha$={ALPHA}" if cells else None),
                        labels=labs, show_rho=False, s=70, labelsize=19, label_fs=10.5,
                        note_fs=14,
                        note_corner=note_free_corner(x_mv[idx], y, fam,
                                                     *((0.44, 0.30) if cells
                                                       else (0.28, 0.16))),
                        axis_contours=True)
            res.append((group, lab, r, p, ns, nc))

    # The family key goes at the END, under the last row. The reserved strip is a
    # FRACTION of figure height, so it has to shrink as the stack grows or it opens a
    # gap the legend does not fill.
    bot = 0.07 / len(rows)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03 / len(rows)),
               fontsize=17, markerscale=1.9, handletextpad=0.5, columnspacing=1.8)
    fig.tight_layout(rect=(0, bot, 1, 1 - 0.05 / len(rows)), h_pad=h_pad)
    for i, (modality, group) in enumerate(rows):
        _row_heading(fig, axs[i][0], axs[i][1],
                     rf"$\mathbf{{EEG}}\ \mathbf{{\times}}\ "
                     rf"{_bold(display(modality, group))}$", side="top")
    f = OUT / ((stem or "cyclic_eeg_x_" + "_".join(g for _, g in rows))
               + "__nativegrid" + ("__cells" if cells else "") + ".png")
    fig.savefig(f, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f, res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", nargs="+", default=list(FOLDER), choices=list(FOLDER))
    ap.add_argument("--group", nargs="+", default=None,
                    help="restrict to these groups (default: all of each modality's)")
    ap.add_argument("--yq", default="excess", choices=["excess", "cal"])
    ap.add_argument("--stack", nargs="+", metavar="MODALITY:GROUP",
                    default=["video:vjepa2", "language:bloom"],
                    help="partners to stack into one grid figure, one row each "
                         "(default video:vjepa2 language:bloom). Empty list skips it.")
    ap.add_argument("--stack-stem", default=None,
                    help="output stem for the stacked figure (default: the group names "
                         "joined by underscores)")
    ap.add_argument("--per-family", action="store_true",
                    help="also write the 2x5 metric-by-family figure, which is the "
                         "honest reading of a native-grid run: W differs between "
                         "families, so y is not in comparable units across them.")
    args = ap.parse_args()
    plt.rcParams.update({**THEME, "axes.ymargin": 0.10})
    keys_x, _, x_mv, _ = load_x()

    for modality in args.modality:
        groups = [g for g in GROUPS[modality]
                  if args.group is None or g in args.group]
        if not groups:
            continue
        print(f"\n=== {modality} -> outputs/{FOLDER[modality]}/ ===")
        for g in groups:
            if args.per_family:
                p = per_family(g, args.yq, keys_x, x_mv, modality)
                print(f"  wrote {p.name}")
            p, res = pooled(g, args.yq, keys_x, x_mv, cells=False, modality=modality)
            print(f"  wrote {p.name}")
            for lab, r, pv, ns, nc in res:
                print(f"    {lab:5} rho_adj={r:+.3f} p={pv:.4f}   sig {ns}/{nc}")

    if args.stack:
        rows = [tuple(a.split(":", 1)) for a in args.stack]
        print("\n=== stacked native grid -> outputs/ ===")
        f, res = stacked(rows, keys_x, x_mv, args.yq, stem=args.stack_stem)
        print(f"  wrote {f.name}")
        for g, lab, r, pv, ns, nc in res:
            print(f"    {g:12} {lab:5} rho_adj={r:+.3f} p={pv:.4f}   sig {ns}/{nc}")


if __name__ == "__main__":
    main()
