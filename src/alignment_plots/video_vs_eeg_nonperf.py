"""
video_vs_eeg_nonperf.py — cross-modal EEG × vision alignment, *size* axis (no perf).

This is the classic view (notebook analyze_mknn_hf.ipynb §2–5): for each vision
architecture, one figure with one subplot per EEG model, x = EEG size, one colored
curve per vision size. Significance brackets:
  • colored text between consecutive EEG sizes  = EEG-scaling paired t-test,
  • yellow boxes above each EEG size            = vision-scaling paired t-tests.

"nonperf" = the x-axis is model size, NOT a benchmark score (that's video_vs_eeg.py).

Outputs (one per vision arch):
  src/alignment_plots/outputs/video_vs_eeg_nonperf__<arch>.png
  src/alignment_plots/outputs/video_vs_eeg_nonperf__<arch>.npz

Usage:
  python src/alignment_plots/video_vs_eeg_nonperf.py
  python src/alignment_plots/video_vs_eeg_nonperf.py --vision-arch dinov2 --k-mknn 5
  python src/alignment_plots/video_vs_eeg_nonperf.py --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

import _common as C


def _bracket_label(a, b, S_n):
    _, pval = ttest_rel(a, b)
    n_up = int((b > a).sum())
    return f"{C.sig_stars(pval)} ({n_up}/{S_n}↑)", pval


def _plot_arch(axes2d, arch, results_by_eeg, eeg_models, vision_sizes, k, k_perm):
    """axes2d shape (2, n_eeg): row 0 = calibrated (Aristotelian) with scaling brackets,
    row 1 = observed (uncalibrated) mKNN vs the permutation-null baseline band."""
    vis_colors = C.VISION_SIZE_COLORS[arch]
    arch_name = C.VISION[arch]["display"]

    for j, model in enumerate(eeg_models):
        ax_cal, ax_obs = axes2d[0][j], axes2d[1][j]
        results = results_by_eeg[model]
        eeg_sizes = C.EEG[model]["sizes"]
        x = np.arange(len(eeg_sizes))
        S_n = len(results[(eeg_sizes[0], vision_sizes[0])]["s_cal"])
        xticklabels = [f"{sz}\n({C.PARAMS[model][sz]}M)" for sz in eeg_sizes]

        obs = {vs: np.array([[results[(esz, vs)]["T_obs"][s] for esz in eeg_sizes]
                             for s in range(S_n)]) for vs in vision_sizes}
        cal = {vs: np.array([[results[(esz, vs)]["s_cal"][s] for esz in eeg_sizes]
                             for s in range(S_n)]) for vs in vision_sizes}
        cmean = {vs: cal[vs].mean(0) for vs in vision_sizes}
        cstd  = {vs: cal[vs].std(0)  for vs in vision_sizes}
        omean = {vs: obs[vs].mean(0) for vs in vision_sizes}
        ostd  = {vs: obs[vs].std(0)  for vs in vision_sizes}

        # ── row 0: calibrated (Aristotelian) + scaling-significance brackets ──
        for vs in vision_sizes:
            ax_cal.errorbar(x, cmean[vs], yerr=cstd[vs], color=vis_colors[vs], lw=2.5,
                            marker="o", ms=7, capsize=4, label=f"{arch_name}-{vs}")
        ax_cal.axhline(0.0, color="black", lw=1.0, ls=":", label="calibrated chance (0)")
        c_top = max((cmean[vs] + cstd[vs]).max() for vs in vision_sizes)
        c_bot = min(0.0, min((cmean[vs] - cstd[vs]).min() for vs in vision_sizes))
        crng = max(c_top - c_bot, 1e-6)
        for xi in range(len(eeg_sizes) - 1):
            for vs in vision_sizes:
                label, _ = _bracket_label(cal[vs][:, xi], cal[vs][:, xi + 1], S_n)
                my = (cmean[vs][xi] + cmean[vs][xi + 1]) / 2
                ax_cal.text(xi + 0.5, my, label, ha="center", va="center", fontsize=6.5,
                            color=vis_colors[vs], fontweight="bold",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
        v_pairs = list(zip(vision_sizes[:-1], vision_sizes[1:]))
        box_base = c_top + crng * 0.10
        last_xi = len(eeg_sizes) - 1
        for xi in range(len(eeg_sizes)):
            lines = [f"{v1}→{v2}: {_bracket_label(cal[v1][:, xi], cal[v2][:, xi], S_n)[0]}"
                     for v1, v2 in v_pairs]
            ha = "left" if xi == 0 else ("right" if xi == last_xi else "center")
            if lines:
                ax_cal.text(xi, box_base, "\n".join(lines), ha=ha, va="bottom",
                            fontsize=5.5, linespacing=1.4,
                            bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                                      edgecolor="dimgray", alpha=0.9))
        ax_cal.set_ylim(c_bot - crng * 0.04, box_base + crng * 0.22 * max(len(v_pairs), 1))
        ax_cal.set_xticks(x); ax_cal.set_xticklabels(xticklabels, fontsize=9)
        ax_cal.set_title(f"{C.DISPLAY[model]} — calibrated", fontweight="bold")
        ax_cal.grid(axis="y", alpha=0.3); ax_cal.legend(fontsize=7)

        # ── row 1: observed (uncalibrated) vs permutation-null baseline band ──
        null_mean = np.array([np.mean([results[(esz, vs)]["null_mean"].mean()
                                       for vs in vision_sizes]) for esz in eeg_sizes])
        null_tau  = np.array([np.mean([results[(esz, vs)]["tau"].mean()
                                       for vs in vision_sizes]) for esz in eeg_sizes])
        for vs in vision_sizes:
            ax_obs.errorbar(x, omean[vs], yerr=ostd[vs], color=vis_colors[vs], lw=2.5,
                            marker="o", ms=7, capsize=4, label=f"{arch_name}-{vs}")
        ax_obs.plot(x, null_mean, color="black", lw=1.6, ls="--", marker="s", ms=4,
                    label="permutation baseline (null mean)", zorder=2)
        ax_obs.fill_between(x, null_mean, null_tau, color="grey", alpha=0.20,
                            label="null spread (mean→95th pct)", zorder=1)
        o_lo = min(null_mean.min(), min((omean[vs] - ostd[vs]).min() for vs in vision_sizes))
        o_hi = max((omean[vs] + ostd[vs]).max() for vs in vision_sizes)
        C.tight_ylim(ax_obs, [o_lo], [o_hi])
        ax_obs.set_xticks(x); ax_obs.set_xticklabels(xticklabels, fontsize=9)
        ax_obs.set_xlabel(f"{C.DISPLAY[model]} size")
        ax_obs.set_title(f"{C.DISPLAY[model]} — observed vs null", fontweight="bold")
        ax_obs.grid(axis="y", alpha=0.3); ax_obs.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser(description="Cross-modal EEG × vision (size axis).")
    ap.add_argument("--vision-arch", choices=[*C.VISION.keys(), "all"], default="all")
    ap.add_argument("--eeg-model", choices=[*C.EEG.keys(), "all"], default="all")
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--k-perm", type=int, default=C.K_PERM_DEFAULT)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    C.HF_TOKEN_CACHE = C.resolve_hf_token(args.hf_token)
    k_perm = 20 if args.smoke else args.k_perm
    archs = list(C.VISION.keys()) if args.vision_arch == "all" else [args.vision_arch]
    eeg_models = list(C.EEG.keys()) if args.eeg_model == "all" else [args.eeg_model]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for arch in archs:
        vsizes = C.VISION[arch]["sizes"]
        if args.smoke:
            vsizes = vsizes[:2]
        print(f"\n=== EEG × {arch} (sizes={vsizes}) ===")
        results_by_eeg = {}
        payload = {}
        for model in eeg_models:
            print(f"  -- {model} --")
            res = C.run_cross_modal(model, C.VISION[arch]["display"], vsizes,
                                    C.VISION[arch]["path"], args.k_mknn, k_perm=k_perm)
            results_by_eeg[model] = res
            for (esz, vs), d in res.items():
                payload[f"{model}__{esz}__{vs}__s_cal"] = d["s_cal"]
                payload[f"{model}__{esz}__{vs}__T_obs"] = d["T_obs"]

        n = len(eeg_models)
        fig, axes = plt.subplots(2, n, figsize=(max(n, 1) * 5.8, 10.6), squeeze=False)
        _plot_arch(axes, arch, results_by_eeg, eeg_models, vsizes, args.k_mknn, k_perm)
        axes[0][0].set_ylabel("calibrated mKNN (Aristotelian)", fontsize=10)
        axes[1][0].set_ylabel("mKNN (observed, uncalibrated)", fontsize=10)
        fig.suptitle(
            f"Cross-modal mKNN — EEG × {C.VISION[arch]['display']} (k={args.k_mknn}, "
            f"K_perm={k_perm})\ntop row = calibrated (Aristotelian) + scaling brackets  |  "
            f"bottom row = observed vs permutation-null baseline (grey band)  |  "
            f"* p<.05 ** p<.01 *** p<.001", fontsize=9)
        fig.tight_layout()
        p = out_dir / f"video_vs_eeg_nonperf__{arch}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        np.savez(out_dir / f"video_vs_eeg_nonperf__{arch}.npz", **payload)
        print(f"  Saved {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
