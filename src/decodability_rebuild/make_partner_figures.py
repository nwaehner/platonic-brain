#!/usr/bin/env python
"""
make_partner_figures.py — one two-panel figure per NON-EEG partner model.

    outputs/eeg_vs_video/cyclic_eeg_x_<arch>.png       dinov2, videomae, videomae_ft, vjepa2
    outputs/eeg_vs_language/cyclic_eeg_x_<family>.png  bloom, openllama, llama

Each is the layout make_all_figures.two_panel gives axes (b) and (c) — mKNN on the left,
CKA on the right, 14 EEG models, family-centred, parameter counts on the points — with
the partner swapped for one vision architecture or one LLM family. The point of the
layout is that metric and partner vary independently: RESULTS.md plots panel (b) from
mKNN and panel (c) from CKA, so any difference between them could be either.

    axis                mKNN cache               CKA cache
    (b) EEG x EEG       cyclic_mknn.npz          cyclic_cka_eeg.npz
    (c) fMRI x EEG      cyclic_fmri_mknn.npz     cyclic_cka.npz
    (d) vision x EEG    cyclic_mknn_vision.npz   cyclic_cka_vision.npz     <- here
    (e) language x EEG  cyclic_mknn_llm.npz      cyclic_cka_llm.npz        <- here

y = obs - mean(cyclic null), family-centred, averaged over that partner's published sizes
    — exactly make_all_figures.stats, for the reasons in that module's docstring: the
    paper's clamped s_cal is the right statistic for DECLARING a cell above chance and
    the wrong one for a scatter, so significance is reported as "n/N cells clear alpha"
    in the annotation box instead of becoming the axis.
x = NICE decodability, subject-mean view, family-centred (plot_panel_b.load_x — the same
    cached decodability__*.npz, no recomputation).

NO rho_adj. The box carries p and the cell count only. Averaging a partner's sizes and
then centring within EEG family leaves 14 points across 5 groups; the fitted slope is
drawn because it reads the scatter, but its magnitude is not a quantity worth printing.

Usage:
    python make_partner_figures.py
    python make_partner_figures.py --axis language
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import _common as C  # noqa: E402
import _llm_grids as G  # noqa: E402
from plot_panel_b import CACHE, OUT, THEME, load_x  # noqa: E402
from make_all_figures import ALPHA, draw, note_free_corner, stats  # noqa: E402

# axis -> (output subfolder, figure-label letter, the partner groups, and how to read the
# npz written by that axis's two compute scripts). `group_field` is the per-partner-model
# column that says which group (vision architecture / LLM family) a model belongs to.
# Both variants are written on every run and never share a filename: the plain figure
# carries p alone, the "__cells" one adds the per-cell alpha count.
SUF = {False: "", True: "__cells"}

AXES = {
    "video": dict(
        folder="eeg_vs_video", letter="d",
        srcs={"mknn": "cyclic_mknn_vision.npz", "cka": "cyclic_cka_vision.npz"},
        keys_field="vis_keys", group_field="vis_arch",
        groups=["dinov2", "videomae", "videomae_ft", "vjepa2"],
        display=lambda g: C.VISION[g]["display"]),
    "language": dict(
        folder="eeg_vs_language", letter="e",
        srcs={"mknn": "cyclic_mknn_llm.npz", "cka": "cyclic_cka_llm.npz"},
        keys_field="llm_keys", group_field="llm_family",
        # G.FAMILIES minus llama: only llama-13b was ever extracted, so its panel would
        # average one model where the others average three and five. The cache still
        # holds it — compute_*_cyclic_llm.py writes every stem it finds — so restoring
        # the panel is a one-word edit if llama-30b/65b are ever published.
        groups=[g for g in G.FAMILIES if g != "llama"],
        display=lambda g: G.DISPLAY[g]),
}


def partner_vals(d, group_field, group, eeg_keys):
    """Mean over the group's sizes. -> y (n_eeg,), n_sig, n_cells, members.

    The EEG-x-partner caches are rectangular — every EEG model meets every partner model
    — so unlike the EEG x EEG case there are no sibling cells to mask: the two sides
    share no family. The average is over the group's columns only, which is what makes
    a per-architecture figure a fair analogue of the EEG panel's average over the
    non-sibling partners.
    """
    grp = np.array([str(g) for g in d[group_field]])
    cols = {j for j, g in enumerate(grp) if g == group}
    E = np.full((len(eeg_keys), len(grp)), np.nan)
    n_sig = n_cell = 0
    for (i, j), c in zip(d["pairs"], d["curves"]):
        if j not in cols:
            continue
        e, cal, p = stats(c)
        E[i, j] = e
        n_sig += p <= ALPHA
        n_cell += 1
    return np.nanmean(E[:, sorted(cols)], axis=1), n_sig, n_cell, len(cols)


def _bold(name):
    """`name` as bold mathtext, with any hyphen left as a real hyphen.

    Inside mathtext a "-" is a MINUS: "VideoMAE-K400" comes out as "VideoMAE - K400",
    spaced like an operator. Closing and reopening math around each hyphen keeps it a
    text hyphen; it renders in cmr10 regular, which is invisible at a hyphen's width.
    """
    return "$-$".join(rf"\mathbf{{{part}}}" for part in name.split("-"))


def _load(cfg):
    """The two caches behind one axis, keyed by metric."""
    out = {}
    for metric, fname in cfg["srcs"].items():
        f = CACHE / fname
        if not f.exists():
            raise SystemExit(f"{f} missing — run the matching compute_*_cyclic_* script.")
        out[metric] = np.load(f, allow_pickle=True)
    return out


def _panel(ax, cfg, group, metric, loaded, keys_x, x_mv, xlabel=True, cells=False):
    """One scatter. Shared by the single-partner figures and the stacked grid, so a
    panel is identical in both and only its surroundings change.

    cells=True restores the "n/N cells clear alpha" line under the p. It counts the
    individual EEG x partner-size pairs behind the point, each tested against its own
    cyclic null with the paper's clamped s_cal. It is a different question from the p
    above it — see the note in make_all_figures — so the two can disagree freely, and
    routinely do: an axis can carry a graded ordering with almost no cell clearing."""
    d = loaded[metric]
    eeg_keys = [str(k) for k in d["eeg_keys"]]
    fam = np.array([str(f) for f in d["eeg_family"]])
    y, ns, nc, n_members = partner_vals(d, cfg["group_field"], group, eeg_keys)
    idx = [keys_x.index(k) for k in eeg_keys]
    lab = {"mknn": "mKNN", "cka": "CKA"}[metric]
    labs = [f"{C.PARAMS[k.rsplit('-', 1)[0]][k.rsplit('-', 1)[1]]}M" for k in eeg_keys]
    r, p = draw(ax, x_mv[idx], y, fam,
                None,
                r"$\mathbf{NICE}$ decodability" if xlabel else "",
                rf"$\mathbf{{{lab}}}$",
                note=(rf"{ns}/{nc} cells clear $\alpha$={ALPHA}" if cells else None),
                labels=labs, show_rho=False, s=70, labelsize=17, label_fs=10.5,
                note_fs=14,
                # These panels are not the (b)/(c) ones: the top left is sometimes
                # occupied, and the box would hide the point under it. The footprint
                # to clear depends on whether the cell count adds a second line.
                note_corner=note_free_corner(x_mv[idx], y, fam,
                                             *((0.44, 0.30) if cells else (0.28, 0.16))))
    return (lab, r, p, ns, nc), n_members


def _row_heading(fig, ax_left, ax_right, text, side="top", fontsize=21):
    """Bold heading for one row, placed from the axes' settled positions.

    side="top" centres it above the row. side="right" sets it rotated against the row's
    outer edge, which a deep stack needs: above the row, the heading and the previous
    row's x label compete for the same gap and end up on top of the panels.
    """
    fig.canvas.draw()
    a, b = ax_left.get_position(), ax_right.get_position()
    if side == "right":
        fig.text(b.x1 + 0.014, (a.y0 + a.y1) / 2, text,
                 ha="left", va="center", rotation=-90, fontsize=fontsize)
    else:
        fig.text((a.x0 + b.x1) / 2, a.y1 + 0.012, text,
                 ha="center", va="bottom", fontsize=fontsize)


def two_panel(cfg, group, keys_x, x_mv, cells=False):
    """One partner group, mKNN | CKA. Returns (path, [(metric, rho, p, n_sig, n_cells)])."""
    loaded = _load(cfg)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.6))
    res, n_members = [], None
    for ax, metric in zip(axs, ["mknn", "cka"]):
        row, n_members = _panel(ax, cfg, group, metric, loaded, keys_x, x_mv, cells=cells)
        res.append(row)

    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03),
               fontsize=17, markerscale=1.9, handletextpad=0.5, columnspacing=1.8)
    # The (b)/(c) figures carry no title because there is only one of each. Here there
    # are six files in the same grammar, so each says which partner it is — nothing else.
    # Set in mathtext bold rather than fontweight="bold": the theme's cmr10 ships no bold
    # face, so a weight request silently falls back to DejaVu and the title stops matching
    # the $\mathbf{NICE}$ / $\mathbf{mKNN}$ labels underneath it.
    fig.suptitle(rf"$\mathbf{{EEG}}\ \mathbf{{\times}}\ "
                 rf"{_bold(cfg['display'](group))}$",
                 fontsize=21, y=0.995)
    # rect's top and the suptitle's y are one gap: the axes stop at 0.97 and the title
    # sits just above them, rather than being pushed off the figure and pulled back in
    # by bbox_inches="tight".
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    outdir = OUT / cfg["folder"]
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"cyclic_eeg_x_{group}{SUF[cells]}.png"
    fig.savefig(f, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f, res


def stacked(rows, keys_x, x_mv, stem=None, cells=False):
    """One figure, one ROW per partner, columns mKNN | CKA.

    rows: [(axis, group), ...]. Every panel plots the same 14 EEG models on the same x,
    so the family key is drawn once under the whole grid instead of once per row; the
    per-point parameter counts stay in every panel, since they name individual points
    and colour/marker only encode family.
    """
    fig, axs = plt.subplots(len(rows), 2, figsize=(12.4, 5.6 * len(rows)), squeeze=False)
    res = []
    for i, (axis, group) in enumerate(rows):
        cfg = AXES[axis]
        loaded = _load(cfg)
        for ax, metric in zip(axs[i], ["mknn", "cka"]):
            row, _ = _panel(ax, cfg, group, metric, loaded, keys_x, x_mv, cells=cells)
            res.append((group, *row))

    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.015),
               fontsize=17, markerscale=1.9, handletextpad=0.5, columnspacing=1.8)
    # Leave a band above each row for its heading; tight_layout settles the axes first,
    # then the headings are placed from where the axes actually landed.
    fig.tight_layout(rect=(0, 0.035, 1, 0.975), h_pad=4.5)
    for i, (axis, group) in enumerate(rows):
        _row_heading(fig, axs[i][0], axs[i][1],
                     rf"$\mathbf{{EEG}}\ \mathbf{{\times}}\ "
                     rf"{_bold(AXES[axis]['display'](group))}$")
    f = OUT / ((stem or "cyclic_eeg_x_" + "_".join(g for _, g in rows)) + SUF[cells])
    f = f.with_suffix(".png")
    fig.savefig(f, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f, res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", nargs="+", default=list(AXES), choices=list(AXES))
    ap.add_argument("--stack", nargs="+", metavar="AXIS:GROUP",
                    default=["video:vjepa2", "language:bloom"],
                    help="partners to stack into one grid figure, one row each "
                         "(default video:vjepa2 language:bloom). Empty list skips it.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # THEME's default 5% y-margin leaves the topmost point's label nowhere legal to go
    # in the tighter partner panels, and _place_labels then clips it against the axes.
    # A little more headroom costs nothing and keeps every label inside the frame.
    plt.rcParams.update({**THEME, "axes.ymargin": 0.10})

    keys_x, _, x_mv, _ = load_x()
    for axis in args.axis:
        cfg = AXES[axis]
        print(f"\n=== {axis} -> outputs/{cfg['folder']}/ ===")
        for group in cfg["groups"]:
            for cells in (False, True):
                f, res = two_panel(cfg, group, keys_x, x_mv, cells=cells)
                print(f"  wrote {cfg['folder']}/{f.name}")
            for lab, r, p, ns, nc in res:
                print(f"    {lab:5} rho_adj={r:+.3f} p={p:.4f}   sig {ns}/{nc}")

    if args.stack:
        rows = [tuple(a.split(":", 1)) for a in args.stack]
        print(f"\n=== stacked grid -> outputs/ ===")
        for cells in (False, True):
            f, res = stacked(rows, keys_x, x_mv, cells=cells)
            print(f"  wrote {f.name}")
            for g, lab, r, p, ns, nc in res:
                print(f"    {g:12} {lab:5} rho_adj={r:+.3f} p={p:.4f}   sig {ns}/{nc}")


if __name__ == "__main__":
    main()
