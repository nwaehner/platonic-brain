"""
language_vs_eeg.py — EEG × LLM-caption alignment, performance axis + null tests.

LLM caption embeddings live at llms/<eeg_model>/<stem>_layerwise.npz, aligned to that
EEG model's windows. The x-axis is LLM *performance* (1-bits-per-byte over 4M
OpenWebText tokens, the PRH protocol of Huh et al. 2024). Until LLM_PERF is filled it
falls back to a log-param proxy (monotone with perf within a family); a banner warns.

Three figure types:
  (a) per EEG model — calibrated mKNN vs LLM performance, one curve per EEG size.
  (b) per EEG model — significance: one subplot per EEG size showing the shift-null
      curve (alignment vs temporal offset d, d=0 highlighted) against the
      block-permutation null band, with both p-values. This is the temporal-confound
      check: real (content) alignment should put d=0 above BOTH nulls.
  (c) combined — one row, one subplot per EEG model.

Missing (eeg, stem) NPZs are skipped with a warning, so NeuroLM×{BLOOM,OpenLLaMA}
runs today and the rest fills in as extraction lands.

Outputs:
  src/alignment_plots/outputs/language_vs_eeg__<model>.png            (a)
  src/alignment_plots/outputs/language_vs_eeg__<model>__significance.png  (b)
  src/alignment_plots/outputs/language_vs_eeg__combined.png           (c)
  src/alignment_plots/outputs/language_vs_eeg.npz

Usage:
  python src/alignment_plots/language_vs_eeg.py --eeg-model neurolm
  python src/alignment_plots/language_vs_eeg.py --eeg-model neurolm --llm-family bloom --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _common as C


# ── data loading (graceful skip of missing NPZs) ───────────────────────────────
def available_stems(eeg_model, stems):
    ok = []
    for s in stems:
        try:
            C.fetch(C.llm_path(eeg_model, s))
            ok.append(s)
        except Exception:
            print(f"  [skip] no embeddings for {eeg_model} × {s}")
    return ok


def compute_alignment(eeg_model, stems, k_mknn, k_perm):
    """results[(esz, stem)] = dict(s_cal, T_obs, ...) over subjects (S,)."""
    def vfname(stem, fam):
        return C.llm_path(eeg_model, stem)
    return C.run_cross_modal(eeg_model, "LLM", stems, vfname, k_mknn, k_perm=k_perm)


# ── (a)/(c) alignment-vs-performance plot ──────────────────────────────────────
def _plot_alignment(ax, eeg_model, results, stems):
    stems_sorted = sorted(stems, key=C.llm_x)
    xs = np.array([C.llm_x(s) for s in stems_sorted])
    sizes = C.EEG[eeg_model]["sizes"]
    colors = C.EEG_SIZE_COLORS[eeg_model]

    lows, highs = [], []
    for sz in sizes:
        m = np.array([results[(sz, s)]["s_cal"].mean() for s in stems_sorted])
        sd = np.array([results[(sz, s)]["s_cal"].std() for s in stems_sorted])
        ax.errorbar(xs, m, yerr=sd, color=colors[sz], lw=2.4, marker="o", ms=7,
                    capsize=3, label=f"{C.DISPLAY[eeg_model]}-{sz.upper()}", zorder=3)
        lows.append((m - sd).min()); highs.append((m + sd).max())
    C.tight_ylim(ax, lows, highs)
    ax.set_xticks(xs)
    ax.set_xticklabels([C.LLM_LABEL.get(s, s) for s in stems_sorted],
                       rotation=55, ha="right", fontsize=9)
    xlabel = "LLM performance (1 − bits-per-byte, 4M OpenWebText tokens)" \
        if not C.llm_x_is_proxy() else "LLM size proxy  [log10 params — run measure_llm_bpb.py]"
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("mKNN (calibrated)", fontsize=10)
    ax.set_title(C.DISPLAY[eeg_model], fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def _significance_for_size(eeg_model, esz, stem, k_mknn, shifts, block_len, k_perm):
    eeg = C.load_npz(C.EEG[eeg_model]["fname"](esz))["embeddings"].astype(np.float32)
    llm = C.load_npz(C.llm_path(eeg_model, stem))["embeddings"].astype(np.float32)
    return C.significance_for_size(eeg, llm, k_mknn, shifts, block_len, k_perm)


def main():
    ap = argparse.ArgumentParser(description="EEG × LLM alignment + null tests.")
    ap.add_argument("--eeg-model", choices=[*C.EEG.keys(), "all"], default="all")
    ap.add_argument("--llm-family", choices=[*C.LLM.keys(), "all"], default="all")
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--k-perm", type=int, default=C.K_PERM_DEFAULT)
    ap.add_argument("--block-len", type=int, default=30, help="block size for block-perm null")
    ap.add_argument("--shifts", default="-100:101:5",
                    help="start:stop:step for shift-null (fine steps near 0 are added)")
    ap.add_argument("--no-significance", action="store_true",
                    help="skip the (b) significance figures (they are the slow part)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    C.HF_TOKEN_CACHE = C.resolve_hf_token(args.hf_token)
    k_perm = 20 if args.smoke else args.k_perm
    a, b, st = (int(v) for v in args.shifts.split(":"))
    shifts = sorted(set(list(range(a, b, st)) + list(range(-3, 4)))) if not args.smoke \
        else sorted(set([-50, -20, -5, 0, 5, 20, 50]))

    fams = list(C.LLM.keys()) if args.llm_family == "all" else [args.llm_family]
    stems = [s for f in fams for s in C.LLM[f]]
    eeg_models = list(C.EEG.keys()) if args.eeg_model == "all" else [args.eeg_model]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if C.llm_x_is_proxy():
        print("WARN: LLM_PERF is empty — x-axis uses a log-param proxy. Fill _common.LLM_PERF "
              "with 1-BPB OpenWebText scores for the true PRH performance axis.")

    all_results = {}
    payload = {}
    for m in eeg_models:
        print(f"\n=== language × {m} ===")
        stems_ok = available_stems(m, stems)
        if not stems_ok:
            print(f"  no LLM embeddings for {m}; skipping.")
            continue
        res = compute_alignment(m, stems_ok, args.k_mknn, k_perm)
        all_results[m] = (res, stems_ok)
        for (esz, s), d in res.items():
            payload[f"{m}__{esz}__{s}__s_cal"] = d["s_cal"]

        # (a)
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_alignment(ax, m, res, stems_ok)
        fig.suptitle(f"EEG × LLM caption alignment — {C.DISPLAY[m]} "
                     f"(k={args.k_mknn}, K_perm={k_perm})  {C.llm_axis_tag()}", fontsize=12)
        fig.tight_layout()
        p = out_dir / f"language_vs_eeg__{m}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")

        # (b)
        if not args.no_significance:
            sig_stem = sorted(stems_ok, key=C.llm_x)[-1]   # highest-perf available
            print(f"  significance vs {sig_stem} (shift-null + block-perm)...")
            sig = {esz: _significance_for_size(m, esz, sig_stem, args.k_mknn,
                                               shifts, args.block_len, k_perm)
                   for esz in C.EEG[m]["sizes"]}
            sizes = C.EEG[m]["sizes"]
            fig, axes = C.make_significance_fig(
                [f"{C.DISPLAY[m]}-{sz.upper()}" for sz in sizes], [sig[sz] for sz in sizes])
            smk = "  [smoke: sparse shifts]" if args.smoke else ""
            fig.suptitle(f"Significance — {C.DISPLAY[m]} × {sig_stem} "
                         f"(block_len={args.block_len}){smk}\n"
                         f"top row = shift-null,  bottom row = block-permutation null",
                         fontsize=12)
            fig.tight_layout()
            p = out_dir / f"language_vs_eeg__{m}__significance.png"
            fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
            for esz, d in sig.items():
                payload[f"{m}__{esz}__{sig_stem}__shift_mean"] = d["shift_mean"]
                payload[f"{m}__{esz}__{sig_stem}__shift_p"] = np.array(d["shift_p"])
                payload[f"{m}__{esz}__{sig_stem}__block_p"] = np.array(d["block_p"])
            print(f"  Saved {p}")

    # (c) combined
    if len(all_results) > 1:
        models = list(all_results.keys())
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), squeeze=False)
        for ax, m in zip(axes[0], models):
            res, stems_ok = all_results[m]
            _plot_alignment(ax, m, res, stems_ok)
        fig.suptitle(f"EEG × LLM caption alignment across EEG models "
                     f"(k={args.k_mknn}, K_perm={k_perm})  {C.llm_axis_tag()}", fontsize=13)
        fig.tight_layout()
        p = out_dir / "language_vs_eeg__combined.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")

    np.savez(out_dir / "language_vs_eeg.npz", **payload)
    print("\nDone.")


if __name__ == "__main__":
    main()
