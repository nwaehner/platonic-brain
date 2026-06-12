"""
video_vs_language.py — direct vision × LLM-caption alignment on the movie stimulus.

Both vision and LLM caption embeddings are aligned to a shared EEG-family window
scheme (`--window-scheme`, default `neurolm`), so we can measure language↔vision
convergence directly, with no EEG on either axis.

x-axis = LLM performance (1-BPB OpenWebText; log-param proxy until LLM_PERF is filled).
y-axis = calibrated mKNN. One curve per video model (colored by architecture).

There is no subject axis here (vision/LLM embeddings are subject-independent), so each
point is a single calibrated score; significance uses the temporal-confound nulls.

Outputs:
  src/alignment_plots/outputs/video_vs_language__<scheme>.png
  src/alignment_plots/outputs/video_vs_language__<scheme>__significance.png
  src/alignment_plots/outputs/video_vs_language__<scheme>.npz

Usage:
  python src/alignment_plots/video_vs_language.py --window-scheme neurolm
  python src/alignment_plots/video_vs_language.py --window-scheme neurolm --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _common as C


def load_layers_or_none(path):
    try:
        emb = C.load_npz(path)["embeddings"].astype(np.float32)
        return emb[None] if emb.ndim == 2 else emb
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Vision × LLM alignment on shared windows.")
    ap.add_argument("--window-scheme", choices=[*C.EEG.keys(), "clip4s"], default="clip4s",
                    help="Shared windowing for both modalities. 'clip4s' = native 4 s clips "
                         "(1 caption ↔ 1 clip, 2700 windows; needs clip4s embeddings); an EEG "
                         "model name reuses that family's window tiling.")
    ap.add_argument("--llm-family", choices=[*C.LLM.keys(), "all"], default="all")
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
    scheme = args.window_scheme
    if scheme == "clip4s":
        fam, scheme_label = "clip4s", "4 s clip"
    else:
        fam, scheme_label = C.EEG[scheme]["family"], C.DISPLAY[scheme]
    k_perm = 20 if args.smoke else args.k_perm
    a, b, st = (int(v) for v in args.shifts.split(":"))
    shifts = sorted(set([-50, -20, -5, 0, 5, 20, 50])) if args.smoke \
        else sorted(set(list(range(a, b, st)) + list(range(-3, 4))))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if C.llm_x_is_proxy():
        print("WARN: LLM_PERF empty — x uses a log-param proxy.")

    # LLM stems available for this window scheme
    fams = list(C.LLM.keys()) if args.llm_family == "all" else [args.llm_family]
    stems = [s for f in fams for s in C.LLM[f]]
    llm_emb, llm_nn = {}, {}
    for s in stems:
        e = load_layers_or_none(C.llm_path(scheme, s))
        if e is None:
            print(f"  [skip] no LLM {s} for scheme {scheme}"); continue
        llm_emb[s] = e
        llm_nn[s] = C.precompute_knn_layers(e, args.k_mknn)
    stems_ok = sorted(llm_emb.keys(), key=C.llm_x)
    if not stems_ok:
        print("No LLM embeddings for this scheme; nothing to do."); return

    # video models available for this family
    vid_emb, vid_nn, vid_points = {}, {}, []
    for arch in C.VISION:
        for size in C.VISION[arch]["sizes"]:
            e = load_layers_or_none(C.VISION[arch]["path"](size, fam))
            if e is None:
                continue
            vid_emb[(arch, size)] = e
            vid_nn[(arch, size)] = C.precompute_knn_layers(e, args.k_mknn)
            vid_points.append((arch, size))
    if not vid_points:
        print("No vision embeddings for this family; nothing to do."); return
    print(f"LLMs: {stems_ok}\nVideo models: {vid_points}")

    # ── alignment: s_cal[(arch,size)][stem] ────────────────────────────────────
    s_cal = {pt: {} for pt in vid_points}
    payload = {}
    for pt in vid_points:
        for s in stems_ok:
            _, sc, _, _, _ = C.aristotelian_cross(vid_nn[pt], llm_nn[s],
                                                  K=k_perm, alpha=C.ALPHA_DEFAULT, seed=0)
            s_cal[pt][s] = sc
            payload[f"{pt[0]}__{pt[1]}__{s}__s_cal"] = np.array(sc)

    xs = np.array([C.llm_x(s) for s in stems_ok])
    fig, ax = plt.subplots(figsize=(9, 6))
    lows, highs = [], []
    for arch, size in vid_points:
        y = np.array([s_cal[(arch, size)][s] for s in stems_ok])
        ax.plot(xs, y, color=C.VISION_ARCH_COLORS[arch], lw=2.0, marker="o", ms=6,
                label=f"{C.VISION[arch]['display']}-{size}")
        lows.append(y.min()); highs.append(y.max())
    C.tight_ylim(ax, lows, highs)
    ax.set_xticks(xs)
    ax.set_xticklabels([C.LLM_LABEL.get(s, s) for s in stems_ok],
                       rotation=55, ha="right", fontsize=9)
    ax.set_xlabel("LLM performance (1 − bits-per-byte, 4M OpenWebText tokens)"
                  if not C.llm_x_is_proxy()
                  else "LLM size proxy [log10 params — run measure_llm_bpb.py]", fontsize=10)
    ax.set_ylabel("mKNN (calibrated)", fontsize=10)
    ax.set_title(f"Vision × LLM caption alignment on {scheme_label} windows "
                 f"(k={args.k_mknn}, K_perm={k_perm})  {C.llm_axis_tag()}", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = out_dir / f"video_vs_language__{scheme}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # ── significance: one subplot per arch (largest size) vs top-perf LLM ───────
    if not args.no_significance:
        ref_llm = stems_ok[-1]
        archs = list(dict.fromkeys(a for a, _ in vid_points))
        sigs, titles = [], []
        for arch in archs:
            size = [s for (a, s) in vid_points if a == arch][-1]   # largest available
            vid4 = vid_emb[(arch, size)][:, :, None, :]            # add subject axis
            sig = C.significance_for_size(vid4, llm_emb[ref_llm], args.k_mknn,
                                          shifts, args.block_len, k_perm)
            sigs.append(sig)
            titles.append(f"{C.VISION[arch]['display']}-{size}")
        fig, axes = C.make_significance_fig(titles, sigs)
        smk = "  [smoke: sparse shifts]" if args.smoke else ""
        fig.suptitle(f"Significance — vision × {C.LLM_LABEL.get(ref_llm, ref_llm)} on "
                     f"{scheme_label} windows (block_len={args.block_len}){smk}\n"
                     f"top row = shift-null,  bottom row = block-permutation null", fontsize=12)
        fig.tight_layout()
        p = out_dir / f"video_vs_language__{scheme}__significance.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")

    np.savez(out_dir / f"video_vs_language__{scheme}.npz", **payload)
    print("\nDone.")


if __name__ == "__main__":
    main()
