#!/usr/bin/env python
"""
make_all_figures.py — every calibrated figure, on one y axis, in one place.

Y AXIS = obs - mean(cyclic null).  SIGNIFICANCE = the paper's clamped s_cal, reported as
a count in the annotation box rather than as the axis.

WHY THIS SPLIT. aristotelian.pdf calibrates with Eq. (12),
    s_cal = max( (obs - tau_a) / (1 - tau_a), 0 ),   tau_a = (1-a) quantile of the null,
which is the right statistic for DECLARING a cell above chance — Corollary 5.1 gives it
finite-sample Type-I control. It is the wrong statistic for a SCATTER, for two measured
reasons:

  * the clamp destroys ordering. 71% of fMRI x EEG mKNN cells sit at exactly 0, and the
    resulting rho_adj = +0.54 (p = .047) is a rank correlation over 50 tied zeros; the
    same data on an unclamped axis gives +0.13 (p = .65).
  * the denominator varies per pair, so it can manufacture correlation on its own. For
    fMRI x EEG mKNN, tau alone tracks decodability at rho_adj = +0.776 (p = .001) —
    stronger than the +0.538 that (obs - tau)/(1 - tau) reports. The z variant swaps in
    the null's SD, which the episode-recurrence peaks (z ~ +2.5 every 108 bins) inflate
    unevenly; that is why EEG x EEG CKA reads +0.81 under cal and +0.35 under z.

`obs - mean(null)` has no pair-varying denominator, takes its reference from all ~1079
rotations instead of one order statistic, and keeps the metric's own units. It is a
DEVIATION from the paper and carries no Type-I guarantee — hence significance is reported
separately, with their statistic, as "n/N cells clear alpha = 0.05".

The uncalibrated panel_b_uncal*.png are kept as the reference point: that is what
RESULTS.md plots, and neither MK nor the fMRI CKA is calibrated there at all
(`eeg_feature_alignment.py` -> `best_layer_pair(...)[2]`, `fmri_vs_all.py` -> `best_cka(...)`).

Usage:
    python make_all_figures.py
"""

from __future__ import annotations

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
from plot_panel_b import (CACHE, OUT, THEME, FAMILY_COLOR, FAMILY_MARKER,  # noqa: E402
                          load_x, centre_within_family)
from plot_panel_b_cyclic import calibrate_cyclic  # noqa: E402

ALPHA = 0.05
AXES = {"eeg": dict(mknn="cyclic_mknn.npz", cka="cyclic_cka_eeg.npz",
                    mknn_ps="cyclic_mknn_persubj.npz", cka_ps="cyclic_cka_eeg_persubj.npz",
                    other="other EEG families",
                    title="(b) Cross-model, within EEG - better models agree more with "
                          "other architectures"),
        "fmri": dict(mknn="cyclic_fmri_mknn.npz", cka="cyclic_cka.npz",
                     mknn_ps="cyclic_fmri_mknn.npz", cka_ps="cyclic_cka.npz",
                     other="fMRI",
                     title="(c) Cross-modality - better models agree more with fMRI")}


def stats(curve):
    """-> (excess = obs - mean(null), s_cal, p). Excess is the axis; s_cal gates."""
    o, tau, cal, p = calibrate_cyclic(curve, ALPHA)
    return o - float(np.mean(curve[1:])), cal, p


def eeg_vals(metric_file, per_subject, yq="excess"):
    """Cross-family mean per model. -> y (n[,S]), n_sig, n_cells, keys, fam.

    yq selects the magnitude: "excess" = obs - mean(null) (the default, see the module
    docstring), "cal" = the paper's clamped T_cal of Eq. (17)."""
    d = np.load(CACHE / metric_file, allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    fam = np.array([str(f) for f in d["family"]])
    curves = d["curves"]
    S = curves.shape[1] if per_subject else 1
    same = fam[:, None] == fam[None, :]
    E = np.full((len(keys), len(keys), S), np.nan)
    n_sig = n_cell = 0
    for (i, j), c in zip(d["pairs"], curves):
        if same[i, j]:
            continue
        for si in range(S):
            e, cal, p = stats(c[si] if per_subject else c)
            E[i, j, si] = E[j, i, si] = (cal if yq == "cal" else e)
            n_sig += p <= ALPHA
            n_cell += 1
    y = np.nanmean(E, axis=1)                    # (n, S)
    return (y if per_subject else y[:, 0]), n_sig, n_cell, keys, fam


def fmri_vals(metric_file, per_subject, yq="excess"):
    d = np.load(CACHE / metric_file, allow_pickle=True)
    keys = [str(k) for k in d["keys"] if f"{k}__curves" in d.files]
    subs = [str(s) for s in d[f"{keys[0]}__subjects"]]
    E = np.zeros((len(keys), len(subs)))
    n_sig = n_cell = 0
    for mi, k in enumerate(keys):
        for si, c in enumerate(d[f"{k}__curves"]):
            e, cal, p = stats(c)
            E[mi, si] = cal if yq == "cal" else e
            n_sig += p <= ALPHA
            n_cell += 1
    fam = np.array([k.rsplit("-", 1)[0] for k in keys])
    return (E if per_subject else E.mean(axis=1)), n_sig, n_cell, keys, fam, subs


def _place_labels(ax, xc, yc, labels, fam, fontsize=9.0, pin=None):
    """Put each point's label where it actually fits, measured rather than guessed.

    The previous version picked from four fixed offsets and compared point positions.
    That is not enough: what collides is the RENDERED TEXT, whose width depends on the
    string ("1696M" is five times "7M"), and labels were landing on the trend line and on
    each other. Here every candidate is scored against real display-space geometry —

        obstacles : the markers, the fitted trend line, the annotation box, and every
                    label already placed
        candidates: 12 directions x 3 radii around the point, all with centred anchors
        cost      : overlap area with obstacles, plus a hit count for trend-line samples
                    falling inside the box, plus a large penalty for leaving the axes,
                    plus a small preference for staying near its own point

    Deterministic: same inputs, same layout. Text extents are measured once per label and
    the candidate boxes are derived by translation, so this costs 14 measurements, not
    14 x 36.

    `pin` overrides the search for named labels, {label_text: (dx, dy)} in points, for the
    cases where the automatic choice is legible but not the one you want. Keying by label
    text is safe here because the 14 parameter counts are all distinct.
    """
    pin = pin or {}
    fig = ax.figure
    fig.canvas.draw()                       # a renderer must exist to measure text
    rend = fig.canvas.get_renderer()
    dpp = fig.dpi / 72.0                    # points -> display pixels

    P = np.column_stack([ax.transData.transform(np.column_stack([xc, yc]))])
    abox = ax.get_window_extent(rend)

    # obstacles that are not labels: markers, the trend line, the stats box
    obstacles = [(px - 9, py - 9, px + 9, py + 9) for px, py in P]
    line_pts = []
    for ln in ax.get_lines():
        xd, yd = ln.get_data()
        if len(xd) > 2:                     # the polyfit trend line, not the axhline
            line_pts.append(ax.transData.transform(np.column_stack([xd, yd])))
    for ann in ax.texts:
        bb = ann.get_window_extent(rend)
        obstacles.append((bb.x0, bb.y0, bb.x1, bb.y1))
    line_pts = np.vstack(line_pts) if line_pts else np.empty((0, 2))

    def overlap(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if (w > 0 and h > 0) else 0.0

    # measure each label once
    sizes = []
    for li in labels:
        t = ax.text(0, 0, li, fontsize=fontsize, transform=None)
        bb = t.get_window_extent(rend)
        sizes.append((bb.width, bb.height))
        t.remove()

    cands = [(r * np.cos(a), r * np.sin(a))
             for r in (16.0, 24.0, 33.0)
             for a in np.deg2rad(np.arange(0, 360, 30))]

    placed = []
    for i in np.argsort(-np.asarray(yc)):           # top-down, deterministic
        w, h = sizes[i]
        px, py = P[i]
        if labels[i] in pin:
            ox, oy = pin[labels[i]]
            cx, cy = px + ox * dpp, py + oy * dpp
            placed.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            ax.annotate(labels[i], (xc[i], yc[i]), textcoords="offset points",
                        xytext=(ox, oy), ha="center", va="center", fontsize=fontsize,
                        color=FAMILY_COLOR[fam[i]], zorder=6, clip_on=False)
            continue
        best, best_cost = cands[0], np.inf
        for ox, oy in cands:
            cx, cy = px + ox * dpp, py + oy * dpp
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            cost = 8.0 * sum(overlap(box, o) for o in obstacles + placed)
            if len(line_pts):
                inside = ((line_pts[:, 0] > box[0]) & (line_pts[:, 0] < box[2])
                          & (line_pts[:, 1] > box[1]) & (line_pts[:, 1] < box[3]))
                cost += 300.0 * int(inside.sum())
            if not (box[0] > abox.x0 and box[2] < abox.x1
                    and box[1] > abox.y0 and box[3] < abox.y1):
                cost += 5e4
            cost += 12.0 * np.hypot(ox, oy)         # prefer the nearest workable spot
            if cost < best_cost:
                best, best_cost = (ox, oy), cost
        ox, oy = best
        cx, cy = px + ox * dpp, py + oy * dpp
        placed.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        ax.annotate(labels[i], (xc[i], yc[i]), textcoords="offset points",
                    xytext=(ox, oy), ha="center", va="center", fontsize=fontsize,
                    color=FAMILY_COLOR[fam[i]], zorder=6, clip_on=False)


def note_free_corner(x, y, fam, w=0.44, h=0.30):
    """Which corner `draw` can put its annotation box in without covering a point.

    The box is filled white and annotated last, so anything under it vanishes — in the
    EEG x BLOOMZ mKNN panel that silently swallowed FEMBA-8M. Corners are tried in the
    (b)/(c) order so a panel with a clear top left still looks like those figures, and
    the first one holding no point wins; if every corner is occupied, the emptiest does.

    w, h are the box's footprint as a fraction of the axes, measured generously.
    """
    xc, yc = centre_within_family(x, fam), centre_within_family(y, fam)
    # Data -> axes fraction, matching the autoscale margins matplotlib will apply.
    mx, my = plt.rcParams["axes.xmargin"], plt.rcParams["axes.ymargin"]
    def frac(v, m):
        lo, hi = v.min(), v.max()
        pad = (hi - lo) * m
        return (v - (lo - pad)) / ((hi - lo) + 2 * pad)
    fx, fy = frac(xc, mx), frac(yc, my)
    best, best_n = "tl", None
    for corner in ("tl", "tr", "bl", "br"):
        in_x = fx < w if corner[1] == "l" else fx > 1 - w
        in_y = fy > 1 - h if corner[0] == "t" else fy < h
        n = int((in_x & in_y).sum())
        if n == 0:
            return corner
        if best_n is None or n < best_n:
            best, best_n = corner, n
    return best


def draw(ax, x, y, fam, title, xlabel, ylabel, note=None, labels=None,
         show_rho=True, s=55, labelsize=None, label_fs=9.0, pin=None,
         note_fs=9.0, note_box=True, note_corner="tl", axis_contours=False,
         fit_color="0.25"):
    """labels: per-point text (parameter counts), drawn in the point's family colour.
    show_rho: include rho_adj in the annotation box. p and `note` are unaffected.
    note_corner: which corner the box sits in — "tl" (default, what (b)/(c) use), "tr",
    "bl", "br". The box is opaque and drawn last, so a point underneath it disappears;
    `note_free_corner` below picks a corner that has none."""
    xc, yc = centre_within_family(x, fam), centre_within_family(y, fam)
    for f in sorted(set(np.asarray(fam).tolist())):
        m = np.asarray(fam) == f
        # C.DISPLAY carries the house capitalisation (FEMBA / NeuroLM / STEEGFormer);
        # FAMILY_COLOR and FAMILY_MARKER stay keyed by the lower-case family id.
        ax.scatter(xc[m], yc[m], c=FAMILY_COLOR[f], marker=FAMILY_MARKER[f], s=s,
                   edgecolor="white", linewidth=1.0, zorder=3, label=C.DISPLAY[f])
    if len(xc) > 2 and np.ptp(yc) > 0:
        b, a = np.polyfit(xc, yc, 1)
        xx = np.linspace(xc.min(), xc.max(), 50)
        ax.plot(xx, a + b * xx, color=fit_color, linewidth=1.6, zorder=2)
    rho, p = spearmanr(xc, yc)
    parts = ([rf"$\rho_{{adj}}$ = {rho:+.2f}"] if show_rho else []) \
        + [f"p = {p:.3f}" if p >= 1e-3 else r"p $<$ 0.001"]
    txt = "\n".join(parts)
    if note:
        txt += "\n" + note
    # The box is only worth its ink where the note sits over data; unboxed it still
    # registers as an annotation because it is the only left-aligned text up there.
    nx, ny = (0.04 if note_corner[1] == "l" else 0.96), \
             (0.96 if note_corner[0] == "t" else 0.04)
    ax.annotate(txt, xy=(nx, ny), xycoords="axes fraction", fontsize=note_fs,
                va="top" if note_corner[0] == "t" else "bottom",
                ha="left" if note_corner[1] == "l" else "right",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.8)
                if note_box else None)
    if axis_contours:
        # Draw the axes at the panel contours, not as reference lines through x=y=0.
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(0.7)
    else:
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=labelsize)
    if title:
        ax.set_title(title, loc="left")
    ax.grid(zorder=0)
    # Last: the solver measures against the trend line, the stats box and the settled
    # axis limits, so every one of them has to be on the axes before it runs.
    if labels is not None:
        _place_labels(ax, xc, yc, labels, np.asarray(fam), fontsize=label_fs, pin=pin)
    return float(rho), float(p)


def two_panel(axis, keys_x, fam_x, x_mv, yq="excess", cells=False):
    cfg = AXES[axis]
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.6))
    res = []
    for ax, metric in zip(axs, ["mknn", "cka"]):
        if axis == "eeg":
            y, ns, nc, keys, fam = eeg_vals(cfg[metric], False, yq)
        else:
            y, ns, nc, keys, fam, _ = fmri_vals(cfg[metric], False, yq)
        idx = [keys_x.index(k) for k in keys]
        lab = {"mknn": "mKNN", "cka": "CKA"}[metric]
        # parameter count in millions, from _common.PARAMS — as in plot_panel_b --labels
        labs = [f"{C.PARAMS[k.rsplit('-', 1)[0]][k.rsplit('-', 1)[1]]}M" for k in keys]
        r, p = draw(ax, x_mv[idx], y, fam,
                    None,
                    r"$\mathbf{NICE}$ decodability",
                    (rf"$\mathbf{{{lab}}}_{{\mathrm{{cal}}}}$" if yq == "cal"
                     else rf"$\mathbf{{{lab}}}$"),
                    # The EEG panels clear alpha almost everywhere (78/78, 75/78), so
                    # the count carries no information there and is never drawn. On the
                    # fMRI axis it does (20/70, 50/70), and both variants are written:
                    # the plain figure carries p alone, "__cells" adds the count.
                    note=(rf"{ns}/{nc} cells clear $\alpha$={ALPHA}"
                          if (cells and axis != "eeg") else None),
                    labels=labs, show_rho=False, s=70,
                    labelsize=(19 if axis == "eeg" else 17), label_fs=10.5,
                    # NeuroLM-254M reads better under its marker than beside it.
                    pin={"254M": (0, -16)} if (axis, metric) == ("eeg", "mknn") else None,
                    note_fs=14, axis_contours=(axis == "eeg"),
                    fit_color=("black" if axis == "eeg" else "0.25"))
        res.append((lab, r, p, ns, nc))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03),
               fontsize=17, markerscale=1.9, handletextpad=0.5, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    f = OUT / (f"cyclic_eeg_x_{axis}{'_cal' if yq == 'cal' else ''}"
               f"{'__cells' if (cells and axis != 'eeg') else ''}.png")
    fig.savefig(f, dpi=(300 if axis == "eeg" else None),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {f.name}")
    for lab, r, p, ns, nc in res:
        print(f"    {lab:5} rho_adj={r:+.3f} p={p:.4f}   sig {ns}/{nc}")


def per_subject(axis, metric, keys_x, x_ps):
    cfg = AXES[axis]
    if axis == "eeg":
        Y, ns, nc, keys, fam = eeg_vals(cfg[f"{metric}_ps"], True)
        subs = [f"sub-{i+1:04d}" for i in range(Y.shape[1])]
    else:
        Y, ns, nc, keys, fam, subs = fmri_vals(cfg[f"{metric}_ps"], True)
    idx = [keys_x.index(k) for k in keys]
    col = [int(s.split("-")[1]) - 1 for s in subs]
    S = len(subs)
    lab = {"mknn": "mKNN", "cka": "CKA"}[metric]
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 13.5))
    rows = []
    for si, ax in enumerate(axs.ravel()[:S]):
        r, p = draw(ax, x_ps[idx][:, col[si]], Y[:, si], fam, subs[si],
                    "NICE decodability" if si >= S - 2 else "",
                    (f"{lab} excess over cyclic null")
                    if si % 2 == 0 else None)
        rows.append((subs[si], r, p))
    for ax in axs.ravel()[S:]:
        ax.set_visible(False)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{cfg['title'].split(' - ')[0].strip()} - {lab}, within each subject\n"
                 f"x = that subject's decodability   $\cdot$   y = that subject's "
                 fr"excess over the cyclic null   $\cdot$   {ns}/{nc} cells clear "
                 fr"$\alpha$={ALPHA}",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    f = OUT / f"cyclic_eeg_x_{axis}_{metric}_per_subject.png"
    fig.savefig(f, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {f.name}")
    rr = np.array([r for _, r, _ in rows])
    print("    " + "  ".join(f"{n[-4:]}={r:+.2f}" for n, r, _ in rows)
          + f"   ({int((rr > 0).sum())}/{S} positive)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(THEME)
    keys_x, fam_x, x_mv, x_ps = load_x()
    for axis in ("eeg", "fmri"):
        print(f"\n=== {axis} ===")
        # fMRI gets both variants; the EEG count is uninformative and stays suppressed.
        for cells in ((False,) if axis == "eeg" else (False, True)):
            two_panel(axis, keys_x, fam_x, x_mv, cells=cells)
        for metric in ("mknn", "cka"):
            per_subject(axis, metric, keys_x, x_ps)


if __name__ == "__main__":
    main()
