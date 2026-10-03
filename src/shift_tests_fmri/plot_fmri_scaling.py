"""
plot_fmri_scaling.py — F1 and F2, for either modality (vision or language).

F1  fmri_<mod>_scaling_view.png   THE PRIMARY TEST (H1 and H2 read together)
    One panel per scaling family (vision architecture, or LLM family).
    x = size rank within family, y = subject-mean mKNN.
      * thick black line     = d=0, the true pairing
      * faint coloured lines = individual far-tail rotations, coloured by d
      * grey band            = 5-95 pct of the far-tail null at each size
    d=0 rising WHILE the bundle stays flat is the result. d=0 rising inside a bundle that
    rises with it means the size trend is a shared temporal geometry — which is exactly
    what happened here before temporal exclusion, so read the two runs side by side.

F2  fmri_<mod>_shift_curves.png
    rows = family, cols = size. x = circular shift d, y = mKNN, star at d=0, dashed
    verticals at multiples of the 1080 s episode length (270 windows at 4 s). A +-10
    window zoom inset shows whether the peak sits exactly at d=0 or is displaced — the
    hemodynamic-lag check, since BOLD trails the stimulus.

Usage:
    python plot_fmri_scaling.py --npz outputs_vs_vision/circular_shift_fmri_vision.npz
    python plot_fmri_scaling.py --npz outputs_vs_llm/circular_shift_fmri_llm.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import _common as C  # noqa: E402

EPISODE_S = 1080.0
WIN_SEC = 4


def load(npz_path):
    """Curves plus the explicit cell-metadata table written by _fmri_shift.sweep.

    Members of each group are ordered by the stored `ranks`, so x is ascending model size
    for BOTH modalities. Older npz files without the table fall back to hyphen-splitting
    the key, which is correct for vision cells only ("open_llama_3b" has no such prefix).
    """
    d = np.load(npz_path, allow_pickle=True)
    shifts = d["shifts"]
    tail_min = int(d["tail_min"]) if "tail_min" in d.files else 100
    excl = int(d["excl"]) if "excl" in d.files else 0
    modality = str(d["modality"]) if "modality" in d.files else "vision"

    groups = {}
    if "cells" in d.files:
        for key, grp, rank, lab in zip(d["cells"], d["groups"], d["ranks"], d["labels"]):
            groups.setdefault(str(grp), []).append(
                (int(rank), str(key), str(lab), d[f"{key}__curve"]))
    else:                                              # legacy vision-only layout
        for key in d.files:
            if key.endswith("__curve"):
                name = key[: -len("__curve")]
                arch, size = name.rsplit("-", 1)
                pm = C.VISION_PARAMS.get((arch, size))
                groups.setdefault(arch, []).append(
                    (C.VISION[arch]["sizes"].index(size), name,
                     f"{size.upper()}\n{pm}M" if pm else size.upper(), d[key]))
    for g in groups:
        groups[g].sort(key=lambda t: t[0])
    return shifts, tail_min, excl, modality, groups


def group_title(name):
    return C.VISION[name]["display"] if name in C.VISION else name.upper()


def _suffix(excl):
    return f"temporal exclusion +-{excl} win (+-{excl*WIN_SEC}s)" if excl \
        else "no temporal exclusion"


# ── F1 ────────────────────────────────────────────────────────────────────────
def fig_scaling_view(shifts, tail_min, excl, modality, groups, out_path, max_lines=300):
    names = [g for g in groups if len(groups[g]) >= 2]
    if not names:
        raise SystemExit("no family has >=2 sizes to plot")
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.8))
    axes = np.atleast_1d(axes)

    tail = np.abs(shifts) >= tail_min
    tail_idx = np.where(tail)[0]
    norm = colors.Normalize(vmin=-np.abs(shifts).max(), vmax=np.abs(shifts).max())
    cmap = cm.coolwarm
    rng = np.random.default_rng(0)
    sub = rng.choice(tail_idx, min(max_lines, len(tail_idx)), replace=False)

    for ax, name in zip(axes, names):
        cells = groups[name]
        x = np.arange(len(cells), dtype=float)
        M = np.stack([c[3].mean(axis=0) for c in cells])       # (n_size, W)

        for di in sub:
            ax.plot(x, M[:, di], color=cmap(norm(shifts[di])), lw=0.5, alpha=0.22, zorder=1)
        ax.fill_between(x, np.percentile(M[:, tail], 5, axis=1),
                        np.percentile(M[:, tail], 95, axis=1),
                        color="0.55", alpha=0.35, zorder=2, label="far-tail null (5-95%)")
        ax.plot(x, M[:, 0], "-o", color="black", lw=2.4, ms=7, zorder=4, label="d = 0")

        rho = spearmanr(x, M[:, 0]).statistic if len(x) > 2 else np.nan
        s0 = np.polyfit(x, M[:, 0], 1)[0]
        st = np.array([np.polyfit(x, M[:, di], 1)[0] for di in tail_idx])
        p_slope = (1 + int((st >= s0).sum())) / (len(st) + 1)

        ax.set_title(f"{group_title(name)}\nrho={rho:+.2f}   slope p={p_slope:.4f}",
                     fontsize=10, color="crimson" if p_slope <= 0.05 else "black")
        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in cells], fontsize=8)
        ax.set_xlabel(f"{modality} model size")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("mKNN (subject mean, uncalibrated)")
    axes[0].legend(fontsize=7, loc="best")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cb.set_label("circular shift d (windows)   [d=0 in black]", fontsize=8)
    fig.suptitle(f"fMRI x {modality}: does alignment rise with scale, and only at d=0?"
                 f"   [{_suffix(excl)}]", fontsize=12, y=1.03)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ── F2 ────────────────────────────────────────────────────────────────────────
def fig_shift_curves(shifts, excl, modality, groups, out_path, zoom=10):
    names = list(groups)
    ncol = max(len(groups[g]) for g in names)
    fig, axes = plt.subplots(len(names), ncol, figsize=(3.5 * ncol, 2.9 * len(names)),
                             squeeze=False)
    order = np.argsort(shifts)
    ds = shifts[order]
    ep_win = EPISODE_S / WIN_SEC

    for r, name in enumerate(names):
        cells = groups[name]
        for c in range(ncol):
            ax = axes[r][c]
            if c >= len(cells):
                ax.axis("off"); continue
            _, key, lab, curve = cells[c]
            m = curve.mean(axis=0)
            ax.plot(ds, m[order], color="#3b6ea5", lw=0.4)
            ax.plot([0], [m[0]], "*", color="crimson", ms=13, zorder=5)
            for e in range(1, int(np.abs(ds).max() // ep_win) + 1):
                for sgn in (-1, 1):
                    ax.axvline(sgn * e * ep_win, color="crimson", ls="--", lw=0.5, alpha=0.4)
            ax.set_title(f"{group_title(name)}-{lab.splitlines()[0]}", fontsize=9)
            ax.grid(alpha=0.2, lw=0.4)
            if r == len(names) - 1:
                ax.set_xlabel("circular shift d (windows)")
            if c == 0:
                ax.set_ylabel("mKNN")
            ins = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
            sel = np.abs(ds) <= zoom
            ins.plot(ds[sel], m[order][sel], "-o", color="#3b6ea5", lw=0.8, ms=2.5)
            ins.axvline(0, color="crimson", ls=":", lw=0.8)
            ins.tick_params(labelsize=5)
            ins.set_title(f"+-{zoom} win", fontsize=5)

    fig.suptitle(f"fMRI x {modality}: mKNN at every circular rotation "
                 f"(dashed = 1080 s episodes)   [{_suffix(excl)}]", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--zoom", type=int, default=10)
    args = ap.parse_args()

    p = Path(args.npz)
    if not p.exists():
        raise SystemExit(f"{p} not found — run the sweep first.")
    shifts, tail_min, excl, modality, groups = load(p)
    print(f"  {sum(len(v) for v in groups.values())} cells in {len(groups)} families, "
          f"W={len(shifts)}, tail_min={tail_min}, excl={excl}, modality={modality}")
    od = Path(args.out_dir) if args.out_dir else p.parent
    od.mkdir(parents=True, exist_ok=True)
    fig_scaling_view(shifts, tail_min, excl, modality, groups,
                     od / f"fmri_{modality}_scaling_view.png")
    fig_shift_curves(shifts, excl, modality, groups,
                     od / f"fmri_{modality}_shift_curves.png", args.zoom)


if __name__ == "__main__":
    main()
