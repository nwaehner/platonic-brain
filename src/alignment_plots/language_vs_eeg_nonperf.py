"""
language_vs_eeg_nonperf.py — cross-modal EEG × LLM-caption alignment, *size* axis.

The language analogue of video_vs_eeg_nonperf.py: for each LLM family, one figure with
one subplot per EEG model, x = EEG size, one colored curve per LLM size (stem).
Significance brackets:
  • colored text between consecutive EEG sizes  = EEG-scaling paired t-test,
  • yellow boxes above each EEG size            = LLM-scaling paired t-tests.

"nonperf" = the x-axis is EEG model size, NOT an LLM benchmark score (that's
language_vs_eeg.py, which puts 1-BPB performance on x).

LLM caption embeddings live at llms/<eeg_model>/<stem>_layerwise.npz, aligned to that
EEG model's window scheme. Only stems present for *every* plotted EEG model are kept,
so the shared per-stem colours/brackets line up across subplots.

Outputs (one per LLM family):
  src/alignment_plots/outputs/language_vs_eeg_nonperf__<family>.png
  src/alignment_plots/outputs/language_vs_eeg_nonperf__<family>.npz

Usage:
  python src/alignment_plots/language_vs_eeg_nonperf.py
  python src/alignment_plots/language_vs_eeg_nonperf.py --llm-family bloom --k-mknn 5
  python src/alignment_plots/language_vs_eeg_nonperf.py --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

import _common as C


def available_stems(eeg_model, stems):
    """Stems whose llms/<eeg_model>/<stem> NPZ is fetchable (graceful skip of missing)."""
    ok = []
    for s in stems:
        try:
            C.fetch(C.llm_path(eeg_model, s))
            ok.append(s)
        except Exception:
            pass
    return ok


def _stem_colors(stems):
    """A perceptually-ordered colour per stem so size-scaling reads as a gradient."""
    cmap = plt.get_cmap("viridis")
    n = max(len(stems) - 1, 1)
    return {s: cmap(i / n) for i, s in enumerate(stems)}


def _bracket_label(a, b, S_n):
    _, pval = ttest_rel(a, b)
    n_up = int((b > a).sum())
    return f"{C.sig_stars(pval)} ({n_up}/{S_n}↑)", pval


def _plot_family(axes2d, fam, results_by_eeg, eeg_models, stems, k, k_perm):
    """axes2d shape (2, n_eeg): row 0 = calibrated (Aristotelian) with scaling brackets,
    row 1 = observed (uncalibrated) mKNN vs the permutation-null baseline band.
    `stems` are ordered by performance (llm_x); curves coloured by stem."""
    colors = _stem_colors(stems)

    for j, model in enumerate(eeg_models):
        ax_cal, ax_obs = axes2d[0][j], axes2d[1][j]
        results = results_by_eeg[model]
        eeg_sizes = C.EEG[model]["sizes"]
        x = np.arange(len(eeg_sizes))
        S_n = len(results[(eeg_sizes[0], stems[0])]["s_cal"])
        xticklabels = [f"{sz}\n({C.PARAMS[model][sz]}M)" for sz in eeg_sizes]

        obs = {st: np.array([[results[(esz, st)]["T_obs"][s] for esz in eeg_sizes]
                             for s in range(S_n)]) for st in stems}
        cal = {st: np.array([[results[(esz, st)]["s_cal"][s] for esz in eeg_sizes]
                             for s in range(S_n)]) for st in stems}
        cmean = {st: cal[st].mean(0) for st in stems}
        cstd  = {st: cal[st].std(0)  for st in stems}
        omean = {st: obs[st].mean(0) for st in stems}
        ostd  = {st: obs[st].std(0)  for st in stems}

        # ── row 0: calibrated (Aristotelian) + scaling-significance brackets ──
        for st in stems:
            ax_cal.errorbar(x, cmean[st], yerr=cstd[st], color=colors[st], lw=2.5,
                            marker="o", ms=7, capsize=4, label=C.LLM_LABEL.get(st, st))
        ax_cal.axhline(0.0, color="black", lw=1.0, ls=":", label="calibrated chance (0)")
        c_top = max((cmean[st] + cstd[st]).max() for st in stems)
        c_bot = min(0.0, min((cmean[st] - cstd[st]).min() for st in stems))
        crng = max(c_top - c_bot, 1e-6)
        for xi in range(len(eeg_sizes) - 1):
            for st in stems:
                label, _ = _bracket_label(cal[st][:, xi], cal[st][:, xi + 1], S_n)
                my = (cmean[st][xi] + cmean[st][xi + 1]) / 2
                ax_cal.text(xi + 0.5, my, label, ha="center", va="center", fontsize=6.5,
                            color=colors[st], fontweight="bold",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
        s_pairs = list(zip(stems[:-1], stems[1:]))
        box_base = c_top + crng * 0.10
        last_xi = len(eeg_sizes) - 1
        for xi in range(len(eeg_sizes)):
            lines = [f"{C.LLM_LABEL.get(s1, s1)}→{C.LLM_LABEL.get(s2, s2)}: "
                     f"{_bracket_label(cal[s1][:, xi], cal[s2][:, xi], S_n)[0]}"
                     for s1, s2 in s_pairs]
            ha = "left" if xi == 0 else ("right" if xi == last_xi else "center")
            if lines:
                ax_cal.text(xi, box_base, "\n".join(lines), ha=ha, va="bottom",
                            fontsize=5.5, linespacing=1.4,
                            bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                                      edgecolor="dimgray", alpha=0.9))
        ax_cal.set_ylim(c_bot - crng * 0.04, box_base + crng * 0.22 * max(len(s_pairs), 1))
        ax_cal.set_xticks(x); ax_cal.set_xticklabels(xticklabels, fontsize=9)
        ax_cal.set_title(f"{C.DISPLAY[model]} — calibrated", fontweight="bold")
        ax_cal.grid(axis="y", alpha=0.3); ax_cal.legend(fontsize=7)

        # ── row 1: observed (uncalibrated) vs permutation-null baseline band ──
        null_mean = np.array([np.mean([results[(esz, st)]["null_mean"].mean()
                                       for st in stems]) for esz in eeg_sizes])
        null_tau  = np.array([np.mean([results[(esz, st)]["tau"].mean()
                                       for st in stems]) for esz in eeg_sizes])
        for st in stems:
            ax_obs.errorbar(x, omean[st], yerr=ostd[st], color=colors[st], lw=2.5,
                            marker="o", ms=7, capsize=4, label=C.LLM_LABEL.get(st, st))
        ax_obs.plot(x, null_mean, color="black", lw=1.6, ls="--", marker="s", ms=4,
                    label="permutation baseline (null mean)", zorder=2)
        ax_obs.fill_between(x, null_mean, null_tau, color="grey", alpha=0.20,
                            label="null spread (mean→95th pct)", zorder=1)
        o_lo = min(null_mean.min(), min((omean[st] - ostd[st]).min() for st in stems))
        o_hi = max((omean[st] + ostd[st]).max() for st in stems)
        C.tight_ylim(ax_obs, [o_lo], [o_hi])
        ax_obs.set_xticks(x); ax_obs.set_xticklabels(xticklabels, fontsize=9)
        ax_obs.set_xlabel(f"{C.DISPLAY[model]} size")
        ax_obs.set_title(f"{C.DISPLAY[model]} — observed vs null", fontweight="bold")
        ax_obs.grid(axis="y", alpha=0.3); ax_obs.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser(description="Cross-modal EEG × LLM (size axis).")
    ap.add_argument("--llm-family", choices=[*C.LLM.keys(), "all"], default="all")
    ap.add_argument("--eeg-model", choices=[*C.EEG.keys(), "all"], default="all")
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--k-perm", type=int, default=C.K_PERM_DEFAULT)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    C.HF_TOKEN_CACHE = C.resolve_hf_token(args.hf_token)
    k_perm = 20 if args.smoke else args.k_perm
    families = list(C.LLM.keys()) if args.llm_family == "all" else [args.llm_family]
    eeg_models = list(C.EEG.keys()) if args.eeg_model == "all" else [args.eeg_model]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for fam in families:
        fam_stems = C.LLM[fam]
        print(f"\n=== language × EEG (size axis) · {fam} ===")
        avail = {m: available_stems(m, fam_stems) for m in eeg_models}
        models_ok = [m for m in eeg_models if avail[m]]
        if not models_ok:
            print(f"  no LLM embeddings for {fam}; skipping."); continue
        # stems present for EVERY plotted EEG model, ordered by performance
        common = [s for s in fam_stems if all(s in avail[m] for m in models_ok)]
        if not common:
            print(f"  no stems common to {models_ok}; skipping {fam}."); continue
        stems = sorted(common, key=C.llm_x)
        print(f"  EEG models: {models_ok}\n  stems: {stems}")

        results_by_eeg, payload = {}, {}
        for m in models_ok:
            vfname = (lambda m: lambda stem, _fam: C.llm_path(m, stem))(m)
            res = C.run_cross_modal(m, "LLM", stems, vfname, args.k_mknn, k_perm=k_perm)
            results_by_eeg[m] = res
            for (esz, st), d in res.items():
                payload[f"{m}__{esz}__{st}__s_cal"] = d["s_cal"]
                payload[f"{m}__{esz}__{st}__T_obs"] = d["T_obs"]

        n = len(models_ok)
        fig, axes = plt.subplots(2, n, figsize=(max(n, 1) * 5.8, 10.6), squeeze=False)
        _plot_family(axes, fam, results_by_eeg, models_ok, stems, args.k_mknn, k_perm)
        axes[0][0].set_ylabel("calibrated mKNN (Aristotelian)", fontsize=10)
        axes[1][0].set_ylabel("mKNN (observed, uncalibrated)", fontsize=10)
        fig.suptitle(
            f"Cross-modal mKNN — EEG × {fam} captions (k={args.k_mknn}, K_perm={k_perm})\n"
            f"top row = calibrated (Aristotelian) + scaling brackets  |  "
            f"bottom row = observed vs permutation-null baseline (grey band)  |  "
            f"* p<.05 ** p<.01 *** p<.001", fontsize=9)
        fig.tight_layout()
        p = out_dir / f"language_vs_eeg_nonperf__{fam}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        np.savez(out_dir / f"language_vs_eeg_nonperf__{fam}.npz", **payload)
        print(f"  Saved {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
