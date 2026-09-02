"""
compute_decodability.py — per-EEG-variant × per-feature decodability R², nested subject × time CV.

Shared prerequisite for blocks (b) and (c). For each EEG variant we predict every interpretable
feature from the model's embedding. Three tiers:

  eeg      — NICE EEG-derived features (16), from eeg_features/<grid>__eegfeat.npz.
  low      — low-level visual features (12), from data/interp/features/<grid>__features.npz.
  semantic — BERT [CLS] encoding of the window's captions (768), from <grid>__bert.npz
             (compute_bert_semantic.py). Headline = mean over the 768 dims. The top-24 PCA
             components are also probed, stored under tier `semantic_pca`, and are what the
             disaggregated per-feature panels use.

The old `mid` (142 CLIP object-presence) and `high` (39 Empath) tiers are gone, replaced by
`semantic`.

WHAT CHANGED IN THE PROBE (and why the old numbers were not trustworthy):
  • The old path took the best α by `acc.max(0)` over the TEST folds and then `np.maximum` over
    every layer, also on test — two selection-on-test biases, both optimistic.
  • It split on time only, over a subject-MEAN embedding, so no subject-level generalisation
    was ever established.
  • Its α grid stopped at 1e4, too low for near-zero-signal tiers to back off to the mean.
  Now: `nested_ridge_r2` chooses α AND layer on a validation split and reports only test R²,
  crossed over subjects × contiguous time blocks (6 Latin-square folds), α up to 1e9.
  See src/interp/feature_search/feature_gridsearch.py and _cache/NEGATIVE_R2_DIAGNOSIS.md.

READING THE OUTPUT: `perf` is negative for weakly-encoded tiers. Compare it with `perf_null`,
the intercept-only floor, which is itself negative under block CV — not with zero.

Output: data/paper-plots/_cache/decodability_allfeatures.npz
  keys (V,) · family (V,) · feat_names (F,) · feat_tier (F,) · perf (V,F) · perf_folds (V,F,6)
  · perf_sd (V,F) · perf_null (V,F) · best_layer (V,F) · best_alpha (V,F)

Usage:
  python src/paper_plots/compute_decodability.py               # all 14 variants
  python src/paper_plots/compute_decodability.py --smoke       # femba+luna only (cached)
  python src/paper_plots/compute_decodability.py --models femba,luna
"""

from __future__ import annotations

import argparse

import numpy as np

import _paper_common as P
from _paper_common import C, IC
import compute_bert_semantic as BERT
from feature_gridsearch import nested_ridge_r2, ALPHAS_EXT
from eeg_feature_alignment import load_eeg_feat        # published NICE targets (subject-mean)

FEATURES_DIR = P.PROJECT_ROOT / "data" / "interp" / "features"


def _clean(feat):
    """NaN/inf → column mean (same as eeg_feature_alignment._clean)."""
    F = np.asarray(feat, np.float64).copy()
    cm = np.nanmean(F, axis=0)
    cm = np.where(np.isfinite(cm), cm, 0.0)
    bad = np.where(~np.isfinite(F))
    F[bad] = np.take(cm, bad[1])
    return F


def load_targets(grid):
    """All three tiers for one window grid, concatenated into a single (W, F) target block.

    Stacking them means `nested_ridge_r2` shares one SVD per (fold, layer) across every
    feature of every tier, instead of re-decomposing once per tier.
    Returns (Y, names, tiers) or (None, None, None) if a tier is missing."""
    blocks, names, tiers = [], [], []

    feat, fnames = load_eeg_feat(grid)                       # eeg tier (16)
    if feat is None:
        print(f"  [skip] {grid}: no NICE eeg features")
        return None, None, None
    blocks.append(feat); names += [str(n) for n in fnames]; tiers += ["eeg"] * feat.shape[1]

    fpath = FEATURES_DIR / f"{grid}__features.npz"           # low tier (12)
    if not fpath.exists():
        print(f"  [skip] {grid}: no features file {fpath.name}")
        return None, None, None
    z = np.load(fpath, allow_pickle=True)
    blocks.append(z["low"].astype(np.float64))
    names += [str(n) for n in z["low_names"]]
    tiers += ["low"] * z["low"].shape[1]

    cls, pca = BERT.load(grid)                               # semantic tier (768 + 24 PCA)
    if cls is None:
        print(f"  [skip] {grid}: no BERT targets — run compute_bert_semantic.py first")
        return None, None, None
    blocks.append(cls); names += [f"sem:d{i:03d}" for i in range(cls.shape[1])]
    tiers += ["semantic"] * cls.shape[1]
    blocks.append(pca); names += [f"sem:pc{i + 1:02d}" for i in range(pca.shape[1])]
    tiers += ["semantic_pca"] * pca.shape[1]

    W = min(b.shape[0] for b in blocks)
    Y = _clean(np.concatenate([b[:W] for b in blocks], axis=1))
    return Y, names, tiers


def variant_decodability(model, size, Y, n_blocks, alphas, hold_out_subject):
    """Nested-CV R² of every column of Y from one EEG variant, or None if unavailable.

    Default (`hold_out_subject=False`) uses the SUBJECT-MEAN embedding against the
    subject-mean targets — a matched task — with held-out contiguous time blocks. Holding out
    a subject as well costs 1.91 R² on the eeg tier and drives every tier to its chance floor,
    because it asks one person's embedding to predict the GROUP AVERAGE. That comparison is
    only well-posed once per-subject targets exist. See _cache/NEGATIVE_R2_DIAGNOSIS.md."""
    if hold_out_subject:
        emb = IC.load_eeg_emb_subjects(model, size)          # (L, W, S, D) — subjects KEPT
    else:
        emb = IC.load_eeg_emb(model, size)                   # (L, W, D) subject-mean
        emb = None if emb is None else emb[:, :, None, :]    # → (L, W, 1, D)
    if emb is None:
        print(f"  [skip] {model}-{size}: embedding unavailable")
        return None
    print(f"  {model}-{size}: emb{emb.shape} × Y{Y.shape}", flush=True)
    return nested_ridge_r2(emb, Y, n_blocks=n_blocks, alphas=alphas, verbose=True,
                           hold_out_subject=hold_out_subject)


def main():
    ap = argparse.ArgumentParser(description="Per-variant nested-CV decodability R², 3 tiers.")
    ap.add_argument("--blocks", type=int, default=6, help="contiguous time blocks (6)")
    ap.add_argument("--hold-out-subject", action="store_true",
                    help="also hold out a subject (per-subject X). Drives every tier to its "
                         "chance floor with the current subject-MEAN targets — only meaningful "
                         "once per-subject NICE targets are rebuilt.")
    ap.add_argument("--smoke", action="store_true", help="femba+luna only (cached locally)")
    ap.add_argument("--models", default=None, help="comma list of EEG models to include")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    P.ensure_dirs()

    if args.models:
        models = args.models.split(",")
    elif args.smoke:
        models = ["femba", "luna"]
    else:
        models = list(C.EEG.keys())

    keys, fams, rows = [], [], []
    feat_names = feat_tier = None
    for m in models:
        Y, names, tiers = load_targets(m)
        if Y is None:
            continue
        feat_names, feat_tier = names, tiers
        for s in C.EEG[m]["sizes"]:
            out = variant_decodability(m, s, Y, args.blocks, ALPHAS_EXT,
                                       args.hold_out_subject)
            if out is None:
                continue
            rows.append(out)
            keys.append(f"{m}-{s}")
            fams.append(m)                    # model name (FAM_COLORS/MARKERS key)

    if not rows:
        raise SystemExit("no variants produced — nothing to save")

    tiers_arr = np.array(feat_tier)
    perf = np.vstack([r["r2"] for r in rows])
    null = np.vstack([r["r2_null"] for r in rows])
    # Skill score: 0 = no better than the intercept-only predictor, 1 = perfect. Features have
    # wildly different floors (−0.12 to −5.86 within the eeg tier alone), so a raw-R² tier mean
    # is dominated by whichever features drift most and is not a meaningful summary. The skill
    # score puts every feature on the same scale before averaging.
    skill = (perf - null) / (1.0 - null)
    P.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(P.DECODABILITY_NPZ,
             keys=np.array(keys), family=np.array(fams),
             feat_names=np.array(feat_names), feat_tier=tiers_arr,
             perf=perf, perf_skill=skill,
             perf_folds=np.stack([r["r2_folds"].T for r in rows]),      # (V, F, folds)
             perf_sd=np.vstack([r["r2_sd"] for r in rows]),
             perf_null=np.vstack([r["r2_null"] for r in rows]),
             best_layer=np.vstack([r["best_layer"][0] for r in rows]),
             best_alpha=np.vstack([r["best_alpha"][0] for r in rows]))

    print(f"\nSaved {P.DECODABILITY_NPZ}  ({perf.shape[0]} variants × {perf.shape[1]} feats)")
    for t in ["eeg", "low", "semantic", "semantic_pca"]:
        c = tiers_arr == t
        if c.any():
            print(f"  {t:<13} n={c.sum():<4} meanR²={np.nanmean(perf[:, c]):+.3f}  "
                  f"(floor {np.nanmean(null[:, c]):+.3f})  "
                  f"mean skill={np.nanmean(skill[:, c]):+.4f}  "
                  f"frac feats with skill>0.05: {np.nanmean(skill[:, c] > 0.05):.1%}")


if __name__ == "__main__":
    main()
