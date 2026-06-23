"""
vis_lang_probing.py — Component 5: Component 3, but for vision↔language on the 4 s grid.

Mirrors feature_linear_probing, with the roles set by the user:
  • modality pair = vision ↔ language, both on the 4 s 'clip4s' grid (triniborrell repo)
  • PROBE VISION: for each (vision variant × LLM) pair take the pair's best-aligned VISION
    layer, probe its embedding to predict each feature (Ridge block-CV), average R² per tier.
  • plot grid: rows = LLMs, points = vision variants (colour/marker per vision family),
    x = mean R², y = mKNN(vision, language). 3a linear fit + 3b family ANCOVA, plus an
    intramodal-mKNN variant (vision within-family across sizes).

Component 4 (caption similarity) for the LLM∩Vision regime is covered by
`caption_similarity.py --regimes intersection_llm_vision`.

Usage:
  python src/interp/vis_lang_probing.py --smoke
  python src/interp/vis_lang_probing.py --vision-arch dinov2 vjepa2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _interp_common as IC
import _common as C
import interp_features as F
from attribute_probing import block_cv_r2, _zscore
from neighbour_enrichment import _clean
import feature_linear_probing as P

GRID = "clip4s"


def _vision_variants(archs, smoke):
    if smoke:
        return [("dinov2", "small"), ("dinov2", "base"), ("vjepa2", "large")]
    return [(a, s) for a in archs for s in C.VISION[a]["sizes"]]


def _llms(smoke):
    if smoke:
        return ["bloomz-560m", "bloomz-1b1"]
    return [st for fam in C.LLM for st in C.LLM[fam]]


def _vision_cache(smoke, W, k):
    cache = {}

    def get(arch, size):
        key = (arch, size)
        if key not in cache:
            emb = IC.load_vision_grid(arch, size, "clip4s")
            cache[key] = None if emb is None else \
                dict(emb=emb[:, :W], nn=C.precompute_knn_layers(emb[:, :W], k))
        return cache[key]
    return get


def intramodal_mknn(get, arch, size):
    sizes = C.VISION[arch]["sizes"]
    i = sizes.index(size)
    j = IC.intramodal_partner(sizes, i)
    if j is None:
        return np.nan
    a, b = get(arch, size), get(arch, sizes[j])
    if a is None or b is None:
        return np.nan
    return IC.best_layer_pair(a["nn"], b["nn"])[2]


def build_table(vision_variants, llms, smoke, k, folds, max_per_tier):
    feats = F.load_features(GRID, smoke)
    if feats is None:
        print(f"  [skip] no feature cache for grid '{GRID}' — build Comp 1 first")
        return pd.DataFrame()
    W = int(feats["W"]); feat, sel = P._select_features(feats, max_per_tier)
    get = _vision_cache(smoke, W, k)
    # cache LLM kNN
    llm_nn = {}
    for st in llms:
        emb = IC.load_llm_grid(GRID, st)
        llm_nn[st] = None if emb is None else C.precompute_knn_layers(emb[:, :W], k)

    rows = []
    for (va, vsz) in vision_variants:
        v = get(va, vsz)
        if v is None:
            continue
        intra = intramodal_mknn(get, va, vsz)
        for st in llms:
            if llm_nn[st] is None:
                continue
            lv, ll, cross = IC.best_layer_pair(v["nn"], llm_nn[st])
            X = _zscore(C.l2(v["emb"][lv]))[:W]
            rec = dict(vision=f"{va}-{vsz}", family=va, llm=C.LLM_LABEL.get(st, st),
                       cross_mknn=float(cross), intra_mknn=float(intra))
            for t, cols in sel.items():
                r2s = [block_cv_r2(X, feat[:, c], folds) for c in cols]
                rec[f"R2_{t}"] = float(np.nanmean(r2s)) if r2s else np.nan
            rows.append(rec)
            print(f"  probe {rec['vision']} × {rec['llm']}: cross={cross:.3f} "
                  + " ".join(f"{t}={rec.get(f'R2_{t}', np.nan):.3f}" for t in sel))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Component 5: vision↔language probing (4 s).")
    ap.add_argument("--vision-arch", nargs="*", default=list(C.VISION), choices=list(C.VISION))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-per-tier", type=int, default=12)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)

    vis = _vision_variants(args.vision_arch, args.smoke)
    llms = _llms(args.smoke)
    mpt = 4 if args.smoke else args.max_per_tier
    folds = 3 if args.smoke else args.folds

    df = build_table(vis, llms, args.smoke, args.k, folds, mpt)
    if df.empty:
        print("No probe results."); return
    IC.save_npz("probing/vis_lang_probing.npz",
                **{c: df[c].to_numpy() for c in df.columns})

    vfam = [a for a in P.FAM_COLORS if a in C.VISION]
    P.plot_fit(df, "cross_mknn", "Vision→feature R² vs vision↔language mKNN  (5·3a)",
               "probing/vislang_cross__a_fit.png", row_col="llm", fam_pool=vfam)
    P.plot_ancova(df, "cross_mknn", "Vision↔language — family ANCOVA  (5·3b)",
                  "probing/vislang_cross__b_ancova.png", row_col="llm")
    if df["intra_mknn"].notna().any():
        P.plot_fit(df, "intra_mknn", "Vision→feature R² vs intramodal mKNN  (5·3a-intra)",
                   "probing/vislang_intra__a_fit.png", row_col="llm", fam_pool=vfam)
        P.plot_ancova(df, "intra_mknn", "Vision↔language intramodal — family ANCOVA  (5·3b-intra)",
                      "probing/vislang_intra__b_ancova.png", row_col="llm")
    print("Done.")


if __name__ == "__main__":
    main()
