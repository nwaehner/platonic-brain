#!/usr/bin/env python
"""
plot_panel_b.py — RESULTS.md panel (b), rebuilt on a from-scratch x axis.

Two figures:

  panel_b_rebuilt.png        the published figure, reproduced. Same visual grammar as
                             prh_axes.py (Okabe-Ito family colours, per-family markers,
                             family-centred axes, rho_adj box) — but the x values come
                             from compute_decodability.py, not from summary.npz.

  panel_b_per_subject.png    3x2. The same plot once per subject, using THAT subject's
                             own decodability as x.

WHAT VARIES AND WHAT DOES NOT, in the 3x2.
Only x. `MK` — the calibrated cross-architecture mKNN — exists solely as a subject-mean
quantity: `eeg_alignment/summary.npz` and `eeg_alignment_persubj/summary.npz` carry a
byte-identical MK, because only the decodability side was ever redone per subject. Each
panel therefore holds y fixed and moves x. Read a panel as "does this individual's
decodability track the shared alignment", NOT as subject-level variation in alignment.

WHY x HAS TWO DEFENSIBLE DEFINITIONS (--x):
  meanview   average the 6 subjects' embeddings AND features, then fit once.  <- published
  persubj    fit each subject against itself, then average the 6 R2 values.

The first inflates R2 (~6x noise reduction on both sides of the ridge) and admits no
error bar. The second is the honest one and is what the 3x2 decomposes. The default is
`meanview` so this figure is a faithful reproduction; `--x persubj` gives the version the
per-subject data supports, and `--errorbars` adds +-1 SD across subjects on x.

The y axis is reused from eeg_alignment_persubj/summary.npz. That is deliberate: mKNN is
already calibrated against a permutation null that re-maximises over layer pairs
(`_common.aristotelian_cross` calls max_mknn_over_layers inside all K=200 permutations),
so it is the axis with no outstanding methodological question. Holding it fixed makes any
change in rho_adj attributable to the x side alone.

Usage:
    python plot_panel_b.py
    python plot_panel_b.py --x persubj --errorbars
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
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import _common as C  # noqa: E402

CACHE = _HERE / "cache"
OUT = _HERE / "outputs"
REPO_FEAT = "triniborrell/platonic-embeddings"

# lifted verbatim from prh_axes.py so the reproduction is visually identical
FAMILY_COLOR = {"femba": "#0072B2", "luna": "#E69F00", "neurolm": "#009E73",
                "reve": "#D55E00", "steegformer": "#CC79A7"}
FAMILY_MARKER = {"femba": "o", "luna": "s", "neurolm": "^", "reve": "D", "steegformer": "v"}
THEME = {
    "figure.dpi": 150,
    # Computer Modern (cmr10) — the TeX book face — for prose AND maths, so a bold
    # $\mathbf{...}$ fragment sits in the same family as the words around it instead of
    # switching typeface mid-label. cmr10 ships no U+2212, so tick labels go through
    # mathtext; without that every negative tick warns and falls back to a hyphen.
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "cmr10",
    "axes.formatter.use_mathtext": True,
    "axes.titlesize": 13, "axes.labelsize": 11,
    # Barely-off-white panel on a white page, with the grid a shade darker than the
    # ground and no spines or tick marks -- the frame is the panel, not a box drawn
    # around it. White gridlines would vanish at this lightness.
    "axes.facecolor": "#F7F7F7",
    "axes.edgecolor": "#DDDDDD", "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 1.0,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.fontsize": 9, "legend.frameon": False,
}


def _tok():
    p = _HERE.parents[1] / "tokens" / "hf_token.txt"
    return p.read_text().strip() if p.exists() else None


def centre_within_family(x, fam):
    """Mean-centre within family — the ANCOVA `~ x + C(family)` partial-regression contrast."""
    x = np.asarray(x, float)
    out = x.copy()
    fam = np.asarray(fam)
    for f in set(fam.tolist()):
        m = fam == f
        out[m] = x[m] - x[m].mean()
    return out


def load_x():
    """Read every cached model. Returns keys, family, x_meanview, x_persubj (M,S), names."""
    keys, fam, xm, xs = [], [], [], []
    for m in C.EEG:
        for s in C.EEG[m]["sizes"]:
            f = CACHE / f"decodability__{m}-{s}.npz"
            if not f.exists():
                print(f"  [missing] {f.name} — skipped")
                continue
            d = np.load(f, allow_pickle=True)
            keys.append(f"{m}-{s}")
            # The ARCHITECTURE, not C.EEG[m]["family"] — that field is the WINDOW-GRID
            # family and returns "femba_luna" for both, which would merge two distinct
            # architectures into one centring group. summary.npz uses the 5 architecture
            # names, and the family-centring in RESULTS.md is per architecture.
            fam.append(m)
            # collapse: max over LAYERS per feature, then mean over the 16 FEATURES
            xm.append(float(np.nanmean(np.nanmax(d["R2_meanview"], axis=0))))
            xs.append(np.nanmean(np.nanmax(d["R2_subj"], axis=1), axis=1))   # (S,)
    return keys, np.array(fam), np.array(xm), np.stack(xs)


def load_y(keys):
    """Cross-family calibrated mKNN, same masking as prh_axes.load_eeg_table."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO_FEAT, "eeg_alignment_persubj/summary.npz",
                        repo_type="dataset", token=_tok())
    S = np.load(p, allow_pickle=True)
    pub = [str(k) for k in S["keys"]]
    fam = np.array([str(f) for f in S["family"]])
    MK = np.array(S["MK"], float)
    MK[fam[:, None] == fam[None, :]] = np.nan        # drop siblings AND the diagonal
    cross = np.nanmean(MK, axis=1)
    return np.array([cross[pub.index(k)] for k in keys])


def draw(ax, x, y, fam, xerr=None, title=None, xlabel=None, ylabel=None, legend=False,
         labels=None):
    xc, yc = centre_within_family(x, fam), centre_within_family(y, fam)
    for f in sorted(set(np.asarray(fam).tolist())):
        m = np.asarray(fam) == f
        if xerr is not None:
            ax.errorbar(xc[m], yc[m], xerr=np.asarray(xerr)[m], fmt="none",
                        ecolor=FAMILY_COLOR[f], elinewidth=1.0, capsize=2, zorder=2)
        ax.scatter(xc[m], yc[m], c=FAMILY_COLOR[f], marker=FAMILY_MARKER[f],
                   s=55, edgecolor="white", linewidth=1.0, zorder=3, label=f)
    if labels is not None:
        for xi, yi, fi, li in zip(xc, yc, fam, labels):
            ax.annotate(li, (xi, yi), textcoords="offset points", xytext=(6, 5),
                        fontsize=6.5, color=FAMILY_COLOR[fi], zorder=5,
                        fontweight="bold", clip_on=False)
    if len(xc) > 2:
        b, a = np.polyfit(xc, yc, 1)
        xx = np.linspace(xc.min(), xc.max(), 50)
        ax.plot(xx, a + b * xx, color="0.25", linewidth=1.6, zorder=2)
    rho, p = spearmanr(xc, yc)
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.annotate(rf"$\rho_{{adj}}$ = {rho:+.2f}" + "\n" +
                (f"p = {p:.3f}" if p >= 1e-3 else r"p $<$ 0.001"),
                xy=(0.04, 0.90), xycoords="axes fraction", fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.8))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    ax.grid(zorder=0)
    return float(rho), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", default="meanview", choices=["meanview", "persubj"])
    ap.add_argument("--errorbars", action="store_true")
    ap.add_argument("--labels", action="store_true",
                    help="Annotate each point with its parameter count in "
                         "millions (_common.PARAMS), coloured to match the point.")
    ap.add_argument("--err", default="paired", choices=["paired", "sd", "sem"],
                    help="paired (DEFAULT, recommended): each subject's mean is removed "
                         "first, then SEM of the residual. All 14 models are measured on "
                         "the SAME 6 subjects, so a subject who is hard to decode shifts "
                         "every point together — 26.6%% of the variance here is exactly "
                         "that, and it cannot affect a correlation across models. The "
                         "classic repeated-measures error-bar problem. | sd = raw "
                         "between-subject spread (overstates it ~4.6x) | sem = sd/sqrt(6). "
                         "None apply to y: MK has no per-subject version.")
    args = ap.parse_args()
    if args.errorbars and args.x != "persubj":
        raise SystemExit(
            "--errorbars needs --x persubj. The 'meanview' x is a SINGLE ridge fit on "
            "subject-averaged embeddings and features, so it has no per-subject values "
            "to form an error bar from. Borrowing the persubj spread and drawing it on "
            "the meanview point would be a different quantity than the point itself.")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(THEME)

    keys, fam, x_mv, x_ps = load_x()
    print(f"loaded {len(keys)} models from cache")
    y = load_y(keys)
    x = x_mv if args.x == "meanview" else x_ps.mean(axis=1)
    xerr = None
    if args.errorbars:
        S_ = x_ps.shape[1]
        if args.err == "paired":
            # remove each SUBJECT's mean (the shared difficulty offset), leaving the
            # model x subject interaction — the only part that can move model rankings.
            resid = x_ps - x_ps.mean(axis=0, keepdims=True)
            xerr = resid.std(axis=1, ddof=1) / np.sqrt(S_)
        else:
            sd = x_ps.std(axis=1, ddof=1)
            xerr = sd if args.err == "sd" else sd / np.sqrt(S_)

    # ── figure 1: the reproduction ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    xlab = "NICE decodability"
    if args.errorbars:
        _ERRLAB = {"paired": "±1 SEM of the model×subject residual\n"
                             "(each subject's mean removed first — repeated measures)",
                   "sd": "±1 SD across subjects\n"
                         "(includes the shared subject offset — overstates it)",
                   "sem": "±1 SEM across subjects"}
        xlab += f"\npoint = mean of 6 per-subject fits;  bars = {_ERRLAB[args.err]}"
    labs = None
    if args.labels:
        # parameter count in millions, from _common.PARAMS
        labs = [f"{C.PARAMS[k.rsplit('-',1)[0]][k.rsplit('-',1)[1]]}M" for k in keys]
    rho, p = draw(ax, x, y, fam, xerr,
                  title="(b) Cross-model, within EEG\nbetter models agree more with "
                        "other architectures",
                  xlabel=xlab,
                  ylabel="mKNN to other EEG families",
                  labels=labs)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    stem = "panel_b_rebuilt"
    if args.x == "persubj":
        stem += "_persubj"
    if args.errorbars:
        stem += f"_{args.err}"
    if args.labels:
        stem += "_labelled"
    f1 = OUT / f"{stem}.png"
    fig.savefig(f1, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {f1.name}   rho_adj={rho:+.3f} p={p:.4f}   (x = {args.x})")

    rho2, p2 = spearmanr(centre_within_family(x_ps.mean(1), fam),
                         centre_within_family(y, fam))
    rho1, p1 = spearmanr(centre_within_family(x_mv, fam), centre_within_family(y, fam))
    print(f"  for reference: x=meanview rho_adj={rho1:+.3f} p={p1:.4f} | "
          f"x=persubj rho_adj={rho2:+.3f} p={p2:.4f}")

    # ── figure 2: per subject ─────────────────────────────────────────────────
    S = x_ps.shape[1]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 13.5), sharey=True)
    rows = []
    for s, ax in enumerate(axes.ravel()[:S]):
        r, pv = draw(ax, x_ps[:, s], y, fam,
                     title=f"sub-{s+1:04d}",
                     xlabel="NICE decodability" if s >= S - 2 else None,
                     ylabel="mKNN to other EEG families" if s % 2 == 0 else None)
        rows.append((f"sub-{s+1:04d}", r, pv))
    for ax in axes.ravel()[S:]:
        ax.set_visible(False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Panel (b) computed separately for each subject's own decodability\n"
                 "(y — calibrated cross-family mKNN — is a subject-mean quantity and is "
                 "identical in every panel)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    f2 = OUT / "panel_b_per_subject.png"
    fig.savefig(f2, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {f2.name}")
    for n, r, pv in rows:
        print(f"    {n}  rho_adj={r:+.3f}  p={pv:.4f}")
    rr = np.array([r for _, r, _ in rows])
    print(f"    -> {int((rr>0).sum())}/{S} subjects positive, "
          f"range [{rr.min():+.2f}, {rr.max():+.2f}]")


if __name__ == "__main__":
    main()
