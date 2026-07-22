"""
fmri_scaling.py — the canonical Platonic scaling view for the fMRI analysis.

For each window family, plot fMRI–model alignment (Y = mKNN or CKA) against model
SCALE (X = parameters, log axis) as ONE CONNECTED LINE PER MODEL FAMILY (bloom,
openllama, dinov2, videomae, videomae_ft, vjepa2, and the family's EEG model over
its sizes). A rising line = larger/better models of that family converge toward the
fMRI representation (the Platonic Representation Hypothesis signature).

This complements fmri_vs_all.py (per-model bars) and fmri_consolidate.py (pooled
scatter). Reads the outputs/fmri_vs_all__<family>.npz written by fmri_vs_all.py.

Two panels per family figure: left = mKNN(calibrated), right = linear CKA.
Lines are drawn within a model family only (points sorted by size), because scaling
is a within-family statement; cross-architecture baselines differ.

Usage:
    python fmri_scaling.py                 # every family npz present
    python fmri_scaling.py --family neurolm
    python fmri_scaling.py --x perf        # X = performance where defined
                                           #   (LLM 1-BPB, vision K400 top-1); else params
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import _common as C

OUT = Path(__file__).resolve().parent / "outputs"
FAM_ORDER = ["femba_luna", "steegformer", "neurolm", "reve", "clip4s"]

# EEG model → color for its scaling line
EEG_LINE_COLOR = {"femba": "#1a9850", "luna": "#66bd63", "neurolm": "#1a9850",
                  "reve": "#1a9850", "steegformer": "#1a9850"}


def _label_meta(label, kind):
    """Return (model_family, params_M, perf_or_None) for a bar label, or None."""
    if kind == "llm":
        for stem in [s for fam in C.LLM.values() for s in fam]:
            if C.LLM_LABEL.get(stem, stem) == label:
                fam = C.LLM_FAMILY_OF[stem]
                return fam, C.LLM_PARAMS.get(stem), C.LLM_PERF.get(stem)
        return None
    if kind == "vision":
        arch, size = label.rsplit("-", 1)
        return arch, C.VISION_PARAMS.get((arch, size)), C.VIDEO_PERF.get((arch, size))
    if kind == "eeg":
        model, size = label.rsplit("-", 1)
        return model, C.PARAMS.get(model, {}).get(size), None
    return None


def _fam_color(mf):
    if mf in C.LLM_FAMILY_COLORS:
        return C.LLM_FAMILY_COLORS[mf]
    if mf in C.VISION_ARCH_COLORS:
        return C.VISION_ARCH_COLORS[mf]
    return EEG_LINE_COLOR.get(mf, "#1a9850")


def _marker(kind):
    return {"eeg": "s", "vision": "o", "llm": "^"}.get(kind, "o")


def plot_family(fam, use_perf):
    d = np.load(OUT / f"fmri_vs_all__{fam}.npz", allow_pickle=True)
    labels = [str(x) for x in d["labels"]]
    kinds = [str(x) for x in d["kinds"]]
    mknn = d["mknn_cal"]; cka = d["cka"]

    # group points by model family: mf -> list of (x, y_mknn, y_cka, kind)
    groups = {}
    for lbl, kind, mk, ck in zip(labels, kinds, mknn, cka):
        meta = _label_meta(lbl, kind)
        if meta is None:
            continue
        mf, params, perf = meta
        x = perf if (use_perf and perf is not None) else params
        if x is None:
            continue
        groups.setdefault((mf, kind), []).append((x, mk, ck))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for j, (metric_idx, mname) in enumerate([(1, "mKNN (calibrated)"), (2, "linear CKA")]):
        ax = axes[j]
        for (mf, kind), pts in sorted(groups.items()):
            pts = sorted(pts, key=lambda t: t[0])
            xs = [p[0] for p in pts]
            ys = [p[metric_idx] for p in pts]
            ax.plot(xs, ys, marker=_marker(kind), color=_fam_color(mf), lw=2, ms=8,
                    label=f"{mf} ({kind})")
        if not (use_perf):
            ax.set_xscale("log")
        ax.set_xlabel("model performance" if use_perf else "model parameters (M, log)")
        ax.set_ylabel(mname)
        ax.set_title(f"fMRI–model {mname} vs scale — {fam}")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    suffix = "perf" if use_perf else "params"
    p = OUT / f"fmri_scaling__{fam}__{suffix}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all", choices=FAM_ORDER + ["all"])
    ap.add_argument("--x", default="params", choices=["params", "perf"])
    args = ap.parse_args()
    fams = FAM_ORDER if args.family == "all" else [args.family]
    for fam in fams:
        if (OUT / f"fmri_vs_all__{fam}.npz").exists():
            plot_family(fam, args.x == "perf")
        else:
            print(f"[skip] no npz for {fam}")


if __name__ == "__main__":
    main()
