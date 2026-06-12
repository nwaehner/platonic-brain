"""
video_vs_eeg.py — EEG × vision alignment on a *performance* axis (the missing
"video performance vs alignment" plot, analogous to language_vs_eeg.py).

x-axis = Kinetics-400 top-1 accuracy (see _common.VIDEO_PERF). Only (arch, size)
points with a published K400 score appear — that excludes the smaller DINOv2 sizes
(DINOv2 → giant only). Curves = EEG sizes; points colored by vision architecture.

Three figure types (mirrors language_vs_eeg.py):
  (a) per EEG model — calibrated mKNN vs K400 accuracy.
  (b) per EEG model — significance: shift-null + block-permutation per EEG size,
      vs the top-K400 vision model.
  (c) combined — one row, one subplot per EEG model.

Outputs:
  src/alignment_plots/outputs/video_vs_eeg__<model>.png
  src/alignment_plots/outputs/video_vs_eeg__<model>__significance.png
  src/alignment_plots/outputs/video_vs_eeg__combined.png
  src/alignment_plots/outputs/video_vs_eeg.npz

Usage:
  python src/alignment_plots/video_vs_eeg.py --eeg-model neurolm
  python src/alignment_plots/video_vs_eeg.py --eeg-model neurolm --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _common as C


def perf_points():
    """(arch, size, k400) for every vision point with a published K400 score, by perf."""
    pts = [(a, s, C.VIDEO_PERF[(a, s)])
           for a in C.VISION for s in C.VISION[a]["sizes"]
           if C.VIDEO_PERF.get((a, s)) is not None]
    return sorted(pts, key=lambda t: t[2])


def compute_alignment(eeg_model, points, k_mknn, k_perm):
    """results[(esz, arch, size)] = dict(s_cal, ...) over subjects."""
    results = {}
    for arch, size, _ in points:
        try:
            res = C.run_cross_modal(eeg_model, C.VISION[arch]["display"], [size],
                                    C.VISION[arch]["path"], k_mknn, k_perm=k_perm,
                                    verbose=False)
        except Exception as e:
            print(f"  [skip] {eeg_model} × {arch}-{size}: {e}")
            continue
        for (esz, _sz), d in res.items():
            results[(esz, arch, size)] = d
    return results


def _plot_alignment(ax, eeg_model, results, points):
    avail = [(a, s, p) for (a, s, p) in points if any((esz, a, s) in results
             for esz in C.EEG[eeg_model]["sizes"])]
    xs = np.array([p for _, _, p in avail])
    sizes = C.EEG[eeg_model]["sizes"]
    colors = C.EEG_SIZE_COLORS[eeg_model]

    lows, highs = [], []
    for sz in sizes:
        m = np.array([results[(sz, a, s)]["s_cal"].mean() for a, s, _ in avail])
        sd = np.array([results[(sz, a, s)]["s_cal"].std() for a, s, _ in avail])
        ax.errorbar(xs, m, yerr=sd, color=colors[sz], lw=2.4, marker="o", ms=7,
                    capsize=3, label=f"{C.DISPLAY[eeg_model]}-{sz.upper()}", zorder=3)
        lows.append((m - sd).min()); highs.append((m + sd).max())
    C.tight_ylim(ax, lows, highs)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{C.vision_label(a, s)}\nK400={p:.1f}" for a, s, p in avail],
                       rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Kinetics-400 top-1 (%)", fontsize=10)
    ax.set_ylabel("mKNN (calibrated)", fontsize=10)
    ax.set_title(C.DISPLAY[eeg_model], fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def _significance_for_size(eeg_model, esz, arch, size, k_mknn, shifts, block_len, k_perm):
    eeg = C.load_npz(C.EEG[eeg_model]["fname"](esz))["embeddings"].astype(np.float32)
    fam = C.EEG[eeg_model]["family"]
    vis = C.load_npz(C.VISION[arch]["path"](size, fam))["embeddings"].astype(np.float32)
    if vis.ndim == 2:
        vis = vis[None]
    return C.significance_for_size(eeg, vis, k_mknn, shifts, block_len, k_perm)


def main():
    ap = argparse.ArgumentParser(description="EEG × vision alignment (K400 perf axis).")
    ap.add_argument("--eeg-model", choices=[*C.EEG.keys(), "all"], default="all")
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--k-perm", type=int, default=C.K_PERM_DEFAULT)
    ap.add_argument("--block-len", type=int, default=30)
    ap.add_argument("--shifts", default="-100:101:5")
    ap.add_argument("--no-significance", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    C.HF_TOKEN_CACHE = C.resolve_hf_token(args.hf_token)
    k_perm = 20 if args.smoke else args.k_perm
    a, b, st = (int(v) for v in args.shifts.split(":"))
    shifts = sorted(set([-50, -20, -5, 0, 5, 20, 50])) if args.smoke \
        else sorted(set(list(range(a, b, st)) + list(range(-3, 4))))
    eeg_models = list(C.EEG.keys()) if args.eeg_model == "all" else [args.eeg_model]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    points = perf_points()
    print("Vision perf points (K400):", [(a, s, p) for a, s, p in points])

    all_results, payload = {}, {}
    for m in eeg_models:
        print(f"\n=== video × {m} ===")
        res = compute_alignment(m, points, args.k_mknn, k_perm)
        if not res:
            print(f"  no vision alignment for {m}; skipping."); continue
        all_results[m] = res
        for (esz, arch, size), d in res.items():
            payload[f"{m}__{esz}__{arch}__{size}__s_cal"] = d["s_cal"]

        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_alignment(ax, m, res, points)
        fig.suptitle(f"EEG × vision alignment vs K400 — {C.DISPLAY[m]} "
                     f"(k={args.k_mknn}, K_perm={k_perm})", fontsize=13)
        fig.tight_layout()
        p = out_dir / f"video_vs_eeg__{m}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")

        if not args.no_significance:
            ref_arch, ref_size, _ = points[-1]   # top K400
            print(f"  significance vs {ref_arch}-{ref_size} (shift + block)...")
            sizes = C.EEG[m]["sizes"]
            sig = {esz: _significance_for_size(m, esz, ref_arch, ref_size, args.k_mknn,
                                               shifts, args.block_len, k_perm)
                   for esz in sizes}
            fig, axes = C.make_significance_fig(
                [f"{C.DISPLAY[m]}-{sz.upper()}" for sz in sizes], [sig[sz] for sz in sizes])
            smk = "  [smoke: sparse shifts]" if args.smoke else ""
            fig.suptitle(f"Significance — {C.DISPLAY[m]} × {C.VISION[ref_arch]['display']}-"
                         f"{ref_size} (block_len={args.block_len}){smk}\n"
                         f"top row = shift-null,  bottom row = block-permutation null",
                         fontsize=12)
            fig.tight_layout()
            p = out_dir / f"video_vs_eeg__{m}__significance.png"
            fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
            print(f"  Saved {p}")

    if len(all_results) > 1:
        models = list(all_results.keys())
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), squeeze=False)
        for ax, m in zip(axes[0], models):
            _plot_alignment(ax, m, all_results[m], points)
        fig.suptitle(f"EEG × vision alignment vs K400 across EEG models "
                     f"(k={args.k_mknn}, K_perm={k_perm})", fontsize=14)
        fig.tight_layout()
        p = out_dir / "video_vs_eeg__combined.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")

    np.savez(out_dir / "video_vs_eeg.npz", **payload)
    print("\nDone.")


if __name__ == "__main__":
    main()
