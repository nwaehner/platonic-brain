"""
caption_similarity.py — Component 4: caption variance ratio (parallels Component 2).

Same SHARED-neighbour sets S_w per window (every window, k=10) as neighbour_enrichment. The
per-window quantity is the caption-embedding variance ratio

    cap_ratio_w = trace(cov(txt_emb[S_w])) / trace(cov(txt_emb[all]))

(txt_emb = the per-window caption embedding from Component 1 — CLIP-text, else TF-IDF). For
unit vectors trace(cov)=1−‖centroid‖², so a SMALL ratio ⇔ the neighbour captions cluster
tightly ⇔ high mutual cosine. We test (one-sided, matched-n permutation) whether the median
ratio is below the null, and BH-correct across the pairs of a regime.

Outputs (outputs/caption_sim/<grid>/): capvar_<regime>_<label>.png (histogram of cap_ratio_w,
line at 1) + capvar_<regime>.npz (median ratio, p, q per pair) + a saved table of the most
similar anchor↔neighbour caption SENTENCE pairs (interpretable).

Usage:
  python src/interp/caption_similarity.py --smoke
  python src/interp/caption_similarity.py --regimes intersection_llm_vision
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import _interp_common as IC
import interp_features as F
import neighbour_enrichment as NE


def _sentence_index(feats):
    by_w = defaultdict(list)
    for i, w in enumerate(feats["sent_window"]):
        by_w[int(w)].append(i)
    return by_w, feats["sent_emb"], feats["sent_text"]


def top_sentence_pairs(sets, by_w, emb, sent_text, max_windows=200, topk=8):
    """Most-similar (anchor-sentence, neighbour-sentence) cosine pairs across windows."""
    pairs = []
    for w, S in enumerate(sets):
        if len(S) < 2 or w >= max_windows:
            continue
        a_idx = by_w.get(int(w), [])
        n_idx = [i for nb in S for i in by_w.get(int(nb), []) if nb != w]
        if not a_idx or not n_idx:
            continue
        sim = emb[a_idx] @ emb[n_idx].T
        am, nm = np.unravel_index(np.argmax(sim), sim.shape)
        pairs.append((float(sim[am, nm]), sent_text[a_idx[am]], sent_text[n_idx[nm]]))
    return sorted(pairs, reverse=True)[:topk]


def plot_hist(res, title, path):
    vals = res["ratios"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.clip(vals, 0, 2), bins=np.linspace(0, 2, 31), density=True,
            color="#7570b3", alpha=0.8)
    ax.axvline(1, color="k", lw=1.2)
    ax.axvline(res["median"], color="#d62728", lw=1.6, label=f"median={res['median']:.2f}")
    ax.set_xlabel("trace-cov(neighbour captions) / trace-cov(all)")
    ax.set_ylabel("density")
    ax.set_title(f"{title}\nmedian={res['median']:.2f}  p={res['p']:.3g}  "
                 f"q={res.get('q', np.nan):.3g}  n={res['n_used']}", fontsize=9)
    ax.legend(fontsize=8)
    IC.savefig(fig, path)


def run_regime(regime, smoke, k, R):
    results = {}
    for grid, label, specA, specB in NE.iter_pairs(regime, smoke):
        feats = F.load_features(grid, smoke)
        if feats is None:
            continue
        W = int(feats["W"]); txt = np.asarray(feats["txt_emb"])[:W]
        if txt.shape[1] < 2:
            print(f"  [skip] {label}: no caption embedding in cache")
            continue
        sets, score = NE.shared_neighbours(specA, specB, k, W)
        if sets is None:
            continue
        res = IC.trace_cov_ratio_test(txt, sets, R=R)
        if res is None:
            print(f"  [skip] {label}: <3 windows with ≥2 shared neighbours (k={k})")
            continue
        res["mKNN"] = score
        by_w, semb, stext = _sentence_index(feats)
        res["top_pairs"] = (top_sentence_pairs(sets, by_w, semb, stext)
                            if semb.shape[0] > 1 else [])
        results[(grid, label)] = res
        print(f"  {regime} {label}: median cap-ratio={res['median']:.2f} p={res['p']:.3g} "
              f"n={res['n_used']}")
        for c, sa, sn in res["top_pairs"][:2]:
            print(f"      [{c:.2f}] “{sa[:55]}” ↔ “{sn[:55]}”")
    # BH across the pairs of this regime
    if results:
        ps = np.array([r["p"] for r in results.values()])
        qs = IC.benjamini_hochberg(ps)
        for r, q in zip(results.values(), qs):
            r["q"] = float(q)
    return results


def save_and_plot(regime, results):
    flat = {}
    for (grid, label), r in results.items():
        plot_hist(r, f"{regime}: {label}", f"caption_sim/{grid}/capvar_{regime}_{label}.png")
        flat[f"{label}__median"] = np.array([r["median"]])
        flat[f"{label}__p"] = np.array([r["p"]])
        flat[f"{label}__q"] = np.array([r.get("q", np.nan)])
        flat[f"{label}__mKNN"] = np.array([r["mKNN"]])
        if r["top_pairs"]:
            flat[f"{label}__top_pairs"] = np.array(
                [f"{c:.3f}\t{sa}\t{sn}" for c, sa, sn in r["top_pairs"]], dtype=object)
    if flat:
        IC.save_npz(f"caption_sim/capvar_{regime}.npz", **flat)


def main():
    ap = argparse.ArgumentParser(description="Component 4: caption variance ratio.")
    ap.add_argument("--regimes", nargs="*", default=NE.REGIMES, choices=NE.REGIMES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    R = 50 if args.smoke else args.R
    for regime in args.regimes:
        print(f"\n=== {regime} ===")
        results = run_regime(regime, args.smoke, args.k, R)
        save_and_plot(regime, results)
    print("\nDone.")


if __name__ == "__main__":
    main()
