"""
feature_gridsearch.py — which interpretable features are (a) decodable from a model's
embedding and (b) MORE decodable in models that are BETTER aligned (mKNN)?

Replaces the broken Component-3/5 probe (fixed Ridge α=10, contiguous block-CV on spiky
CLIP/Empath targets → R² of −10²…−10³, tracking alignment backwards). Fixes:

  • TARGETS  = the clean scalar stack from `gridsearch_features.py` (low-level visual +
    caption-semantic), probed one scalar at a time (no tier averaging).
  • PROBE    = ridge with a CV-tuned α (best of logspace(-1,4)), standardised X per fold,
    evaluated under BOTH contiguous-block CV (temporal-leakage-safe, conservative) and
    interleaved k-fold (optimistic). A genuinely encoded feature survives both; a merely
    temporally-smooth one only survives interleaved. Implemented as a single vectorised
    SVD ridge-path so all features × all α share one decomposition per fold.
  • AXES     = three cross-modal alignment regimes, each pair → its best-aligned layer of
    the probed model → probe every feature:
        eeg_vision : probe EEG,    align mKNN(EEG, vision)        [EEG grid]
        eeg_llm    : probe EEG,    align mKNN(EEG, LLM-captions)  [EEG grid]
        vision_llm : probe vision, align mKNN(vision, LLM)        [clip4s grid]
    Each pair also carries the probed model's INTRAMODAL (adjacent-size) mKNN, so the
    intramodal axis is recovered at aggregation without a separate sweep.

Aggregation (`--aggregate`): per (axis, feature) report median block/interleaved R²
(decodability) and the family-ANCOVA-adjusted slope of R² vs cross-mKNN and vs intra-mKNN
(does better alignment ⇒ better decoding, net of architecture family?). "Winners" = features
with positive median block R² AND a positive, significant alignment slope on ≥1 axis.

Usage (staged on SLURM — one sweep job per axis, then one aggregate job):
  python src/interp/feature_gridsearch.py --axes eeg_vision   # → rows__eeg_vision.npz
  python src/interp/feature_gridsearch.py --axes eeg_llm
  python src/interp/feature_gridsearch.py --axes vision_llm
  python src/interp/feature_gridsearch.py --aggregate         # → summary + plots
  python src/interp/feature_gridsearch.py --smoke --axes eeg_vision   # tiny
"""

from __future__ import annotations

# bootstrap: add interp root + all study subdirs to sys.path
import sys as _sys; from pathlib import Path as _Path
_INTERP = _Path(__file__).resolve().parent.parent
for _d in ([_INTERP] + [p for p in _INTERP.iterdir() if p.is_dir() and p.name[0] not in "._o"]):
    if str(_d) not in _sys.path: _sys.path.insert(0, str(_d))

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _interp_common as IC
import _common as C
import gridsearch_features as GF

OUT = IC.OUT_DIR / "gridsearch"
ALPHAS = np.logspace(-1, 4, 10)
# Extended grid for nested_ridge_r2. The 1e4 ceiling of ALPHAS is too low for near-zero-signal
# targets (low-level visual): ridge cannot back off all the way to the mean predictor, which
# costs ~0.013 R² of spurious negativity. See _cache/NEGATIVE_R2_DIAGNOSIS.md.
ALPHAS_EXT = np.logspace(-1, 9, 21)
AXES = ["eeg_vision", "eeg_llm", "vision_llm"]

FAM_COLORS = {"femba": "#1b9e77", "luna": "#d95f02", "neurolm": "#7570b3",
              "reve": "#e7298a", "steegformer": "#66a61e",
              "dinov2": "#1b9e77", "videomae": "#d95f02", "videomae_ft": "#7570b3",
              "vjepa2": "#e7298a"}
FAM_MARKERS = {"femba": "o", "luna": "s", "neurolm": "^", "reve": "D", "steegformer": "P",
               "dinov2": "o", "videomae": "s", "videomae_ft": "^", "vjepa2": "D"}


# ── vectorised ridge-path CV R² (all features at once) ────────────────────────────
def _fold_indices(W, n_folds, scheme):
    idx = np.arange(W)
    if scheme == "block":
        b = np.linspace(0, W, n_folds + 1).astype(int)
        return [np.arange(b[f], b[f + 1]) for f in range(n_folds)]
    return [idx[idx % n_folds == f] for f in range(n_folds)]      # interleaved


def ridge_path_r2(X, Y, scheme, n_folds=5, alphas=ALPHAS):
    """Best-α mean CV R² per feature column of Y, predicted from X. One SVD per fold,
    shared across every α and every feature. Returns (F,) R² (NaN if unfittable)."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    W, F = Y.shape
    folds = _fold_indices(W, n_folds, scheme)
    acc = np.zeros((len(alphas), F))            # Σ R² over valid folds, per (α, feature)
    n_ok = np.zeros(F)                          # # folds with real test variance, per feature
    for te in folds:
        tr = np.setdiff1d(np.arange(W), te)
        if len(te) < 3 or len(tr) < 5:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = Y[tr].mean(0)
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)        # U(n,r) s(r) Vt(r,D)
        UtY = U.T @ (Y[tr] - ym)                                  # (r, F)
        yvar = Y[te].var(0)                                       # (F,) test variance
        # a feature ~constant within this test block has no variance to explain → its R²
        # would explode; mask it out of this fold's average.
        valid = yvar > 1e-9
        sstot = yvar * len(te) + 1e-12
        n_ok += valid
        for ai, al in enumerate(alphas):
            B = Vt.T @ ((s / (s ** 2 + al))[:, None] * UtY)       # (D, F)
            pred = Xte @ B + ym                                   # (te, F)
            r2 = np.clip(1.0 - ((Y[te] - pred) ** 2).sum(0) / sstot, -10.0, 1.0)
            acc[ai] += np.where(valid, r2, 0.0)
    gvar = Y.var(0)
    n = np.where(n_ok > 0, n_ok, np.nan)
    best = (acc / n).max(0)                                       # best-α mean-fold R²
    return np.where((n_ok > 0) & (gvar > 1e-9), best, np.nan)


# ── nested subject × time CV (the leakage-safe probe used by the paper plots) ──────
#
# `ridge_path_r2` above chooses α by `acc.max(0)` over the *test* folds, and its callers
# then take `np.maximum` over every layer — both are selection-on-test, i.e. optimistic.
# It also splits on time only, so with a subject-mean embedding there is no subject-level
# generalisation claim at all. `nested_ridge_r2` fixes both: α AND layer are chosen on a
# validation split, R² is reported only on a test split, and the two splits are crossed
# over subjects × contiguous time blocks so neither the subject nor the stimulus window
# is ever shared with training.
#
# A subject-only split would NOT be safe here: the low/semantic targets are stimulus-locked
# (window w has the identical y for every subject), so training on other subjects exposes
# the exact y of every test window. Hence the crossing with time blocks.

def nested_cv_folds(W, n_subjects=6, n_blocks=5, n_inner=4, hold_out_subject=True):
    """Nested folds over (subject, contiguous time block), or over time blocks alone.

    `hold_out_subject=False` → TIME-ONLY mode: every subject appears in every split and only
    the contiguous time blocks rotate (`n_blocks` outer folds). This is the right mode when X
    is the SUBJECT-MEAN embedding and Y is the subject-mean target, because there is then no
    subject axis to generalise across. Measured on neurolm-b's NICE tier: subject-mean X scores
    +0.475 this way, while a single held-out subject's X scores −1.430 against a −1.405 floor —
    the subject dimension costs 1.91 R², versus 0.12 for the selection fix. Predicting the
    GROUP AVERAGE from ONE person's embedding is a mismatched task; it becomes well-posed only
    once per-subject targets exist.

    OUTER fold f (one per subject, so every subject is tested exactly once):
        test      = subject f × time block f % n_blocks
        remaining = the other S-1 subjects × the other n_blocks-1 blocks
    INNER folds (i = 0 … n_inner-1), used ONLY to choose α and the layer:
        val   = remaining-subject i × remaining-block i
        train = the rest of `remaining`
    The final model for the fold is refitted on ALL of `remaining` and scored once on test.

    Averaging the val score over several inner folds is not cosmetic: with a single
    (subject, block) val cell the per-feature argmax over ~40 (layer, α) options overfits the
    val split badly — the chosen layer then differs in 5 of 6 outer folds, i.e. selection is
    close to random, and the resulting test R² lands BELOW the intercept-only floor.

    Returns dicts with `test`, `remaining`, and `inner` = [(train, val), …], each split being
    a (subject-index array, window-index array) pair."""
    blocks = _fold_indices(W, n_blocks, "block")

    def cells(subs, bs):
        return (np.asarray(subs), np.sort(np.concatenate([blocks[b] for b in bs])))

    folds = []
    if not hold_out_subject:
        all_s = list(range(n_subjects))                                # every split sees all
        for f in range(n_blocks):
            rem_b = [b for b in range(n_blocks) if b != f]
            inner = []
            for i in range(min(n_inner, len(rem_b))):
                va_b = rem_b[i]
                inner.append((cells(all_s, [b for b in rem_b if b != va_b]),
                              cells(all_s, [va_b])))
            folds.append(dict(test=cells(all_s, [f]),
                              remaining=cells(all_s, rem_b),
                              inner=inner))
        return folds

    for f in range(n_subjects):
        te_s, te_b = f, f % n_blocks
        rem_s = [s for s in range(n_subjects) if s != te_s]
        rem_b = [b for b in range(n_blocks) if b != te_b]
        inner = []
        for i in range(min(n_inner, len(rem_s), len(rem_b))):
            va_s, va_b = rem_s[i % len(rem_s)], rem_b[i % len(rem_b)]
            inner.append((cells([s for s in rem_s if s != va_s],
                                [b for b in rem_b if b != va_b]),      # inner train
                          cells([va_s], [va_b])))                      # inner val
        folds.append(dict(test=cells([te_s], [te_b]),
                          remaining=cells(rem_s, rem_b),
                          inner=inner))
    return folds


def _rows(E, Y, subs, wins):
    """Stack (window × subject) rows for one split, window-major so X and Y stay aligned.

    E : (W, S, D) one layer, already L2-normalised.  Y : (W, F) stimulus-locked target,
    or (S, W, F) per-subject target. Returns (X (n,D), Yr (n,F))."""
    X = E[np.ix_(wins, subs)].reshape(-1, E.shape[-1])          # window-major
    if Y.ndim == 2:
        Yr = np.repeat(Y[wins], len(subs), axis=0)              # same y for every subject
    else:
        Yr = Y[np.ix_(subs, wins)].transpose(1, 0, 2).reshape(-1, Y.shape[-1])
    return X, Yr


def _split_r2(Ysplit, pred, ymean):
    """R² per feature against the split's own variance (the standard, conservative
    convention already used by ridge_path_r2). Features with no variance in the split
    have nothing to explain → NaN."""
    var = Ysplit.var(0)
    ok = var > 1e-9
    sstot = var * len(Ysplit) + 1e-12
    r2 = 1.0 - ((Ysplit - pred) ** 2).sum(0) / sstot
    return np.where(ok, np.clip(r2, -10.0, 1.0), np.nan), ok


def _fit_eval(E, Y, train, evals, alphas):
    """Fit the whole ridge path on `train`, then score every α on each split in `evals`.

    Uses the eigendecomposition of the d×d Gram matrix XᵀX rather than the SVD of the n×d X:
    (XᵀX + αI)⁻¹XᵀY = V diag(1/(λ+α)) Vᵀ XᵀY, so one `eigh(d)` replaces an `svd(n, d)`. With
    n ≫ d (5 subjects × ~1000 windows vs ≤1600 dims) that is several times cheaper and gives
    the same estimator. Predictions are formed as (Xe·V) @ (G/(λ+α)) so no d×d coefficient
    matrix is ever built per α.

    Returns (list of (A,F) R² arrays, one per eval split; (F,) intercept-only R² on evals[0])."""
    Xtr, Ytr = _rows(E, Y, *train)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8              # train-only standardisation
    ym = Ytr.mean(0)                                     # train-only centring
    Xtr = (Xtr - mu) / sd
    lam, V = np.linalg.eigh(Xtr.T @ Xtr)
    lam = np.maximum(lam, 0.0)
    G = V.T @ (Xtr.T @ (Ytr - ym))                       # (d, F)

    out, null = [], None
    for j, ev in enumerate(evals):
        Xe, Ye = _rows(E, Y, *ev)
        Z = ((Xe - mu) / sd) @ V                         # (n_e, d), once for all α
        r2 = np.empty((len(alphas), Y.shape[-1]))
        for ai, al in enumerate(alphas):
            r2[ai], _ = _split_r2(Ye, Z @ (G / (lam + al)[:, None]) + ym, ym)
        out.append(r2)
        if j == 0:
            null, _ = _split_r2(Ye, np.broadcast_to(ym, Ye.shape), ym)
    return out, null


def _one_se_pick(mean, se, alpha_of):
    """1-SE rule per feature: among the (layer, α) candidates whose mean val R² is within one
    standard error of the best, take the most REGULARISED one (largest α).

    Plain argmax is not safe here. The val cell and the test cell are different time blocks, so
    a small α that happened to suit the val block's drift does not transfer, and the selected
    model can end up scoring BELOW the intercept-only floor — which is what happened before
    this rule was added. The 1-SE rule keeps the flexible model only when the evidence for it
    survives the spread across inner folds, and otherwise falls back towards the mean
    predictor, which is the honest answer for a feature the embedding does not encode.

    mean, se : (n_candidates, F).  alpha_of : (n_candidates,) the α of each candidate.
    Returns (F,) indices into the candidate axis."""
    mean = np.where(np.isfinite(mean), mean, -np.inf)
    best = mean.max(axis=0)                                       # (F,)
    best_idx = mean.argmax(axis=0)
    thresh = best - se[best_idx, np.arange(mean.shape[1])]
    eligible = mean >= thresh[None, :]                            # (n_candidates, F)
    # among eligible, maximise α; ties broken by the higher val score
    score = np.where(eligible, alpha_of[:, None], -np.inf)
    pick = score.argmax(axis=0)
    return np.where(np.isfinite(best), pick, best_idx)


def nested_ridge_r2(emb, Y, n_blocks=5, n_inner=4, alphas=ALPHAS_EXT, verbose=False,
                    hold_out_subject=True):
    """Leakage-safe decodability R² of every column of `Y` from a layerwise EEG embedding.

    emb : (L, W, S, D) per-subject embedding (IC.load_eeg_emb_subjects — NOT the
          subject-mean one).
    Y   : (W, F) stimulus-locked target, or (S, W, F) per-subject target.

    Per outer fold and layer: `n_inner` inner fits score every α on a held-out (subject, block)
    val cell; their MEAN picks the (layer, α) per feature; the model is then refitted on all
    remaining cells and scored once on test. Nothing about the test split ever informs a
    modelling choice.

    Returns dict:
      r2        (F,)      mean test R² over folds        ← the headline number
      r2_folds  (nf, F)   per-fold test R²
      r2_sd     (F,)      SD over folds
      r2_null   (F,)      test R² of the intercept-only (train-mean) predictor. This is the
                          honest chance floor and is NEGATIVE under block CV, because the
                          intercept comes from train while R²'s denominator is the test
                          block's variance about its own mean.
      best_layer, best_alpha  (nf, F)  the val-chosen hyperparameters
    """
    emb = np.asarray(emb)
    Y = np.asarray(Y, np.float64)
    L, W, S, D = emb.shape
    Wy = Y.shape[-2] if Y.ndim == 3 else Y.shape[0]
    W = min(W, Wy)
    F = Y.shape[-1]
    folds = nested_cv_folds(W, n_subjects=S, n_blocks=n_blocks, n_inner=n_inner,
                            hold_out_subject=hold_out_subject)
    nf = len(folds)

    r2_folds = np.full((nf, F), np.nan)
    r2_null = np.full((nf, F), np.nan)
    best_layer = np.full((nf, F), -1, dtype=int)
    best_alpha = np.full((nf, F), np.nan)

    n_in = max(len(folds[0]["inner"]), 1)
    val_sum = np.zeros((nf, L, len(alphas), F))              # Σ and Σ² over inner folds →
    val_sq = np.zeros((nf, L, len(alphas), F))               # mean and SE for the 1-SE rule
    test_r2 = np.full((nf, L, len(alphas), F), np.nan)

    # layer OUTER, fold inner: the float64 cast + L2 of a layer costs tens of MB and is
    # identical for every fold, so doing it once per layer instead of once per (fold, layer)
    # removes a redundant factor of `nf` from the pre-processing.
    for l in range(L):
        E = C.l2(emb[l, :W].astype(np.float64))              # (W, S, D)
        for fi, fold in enumerate(folds):
            for inner_train, inner_val in fold["inner"]:     # α + layer chosen HERE only
                (rv,), _ = _fit_eval(E, Y, inner_train, [inner_val], alphas)
                rv = np.nan_to_num(rv, nan=-10.0)
                val_sum[fi, l] += rv
                val_sq[fi, l] += rv ** 2
            # refit on ALL remaining (subject, block) cells; score once on test
            (rt,), null = _fit_eval(E, Y, fold["remaining"], [fold["test"]], alphas)
            test_r2[fi, l] = rt
            if l == 0:
                r2_null[fi] = null                           # intercept-only floor
        if verbose and (l % 8 == 0 or l == L - 1):
            print(f"      layer {l + 1}/{L}", flush=True)

    alpha_of = np.tile(np.asarray(alphas, float), L)         # α of each flat (layer, α) index
    for fi, fold in enumerate(folds):
        mean = (val_sum[fi] / n_in).reshape(L * len(alphas), F)
        se = (np.sqrt(np.maximum(val_sq[fi] / n_in - (val_sum[fi] / n_in) ** 2, 0.0))
              / np.sqrt(n_in)).reshape(L * len(alphas), F)
        pick = _one_se_pick(mean, se, alpha_of)
        r2_folds[fi] = test_r2[fi].reshape(-1, F)[pick, np.arange(F)]
        best_layer[fi] = pick // len(alphas)
        best_alpha[fi] = np.asarray(alphas)[pick % len(alphas)]
        if verbose:
            te_s, te_w = fold["test"]
            who = f"subj {te_s[0]} × " if hold_out_subject else ""
            print(f"    fold {fi}: test {who}block[{te_w[0]}:{te_w[-1]}]  "
                  f"meanR²={np.nanmean(r2_folds[fi]):+.3f} "
                  f"(null {np.nanmean(r2_null[fi]):+.3f})", flush=True)

    return dict(r2=np.nanmean(r2_folds, axis=0), r2_folds=r2_folds,
                r2_sd=np.nanstd(r2_folds, axis=0), r2_null=np.nanmean(r2_null, axis=0),
                r2_null_folds=r2_null, best_layer=best_layer, best_alpha=best_alpha)


def _clean(feat):
    """Column-mean impute NaNs (constant columns → 0 after centring)."""
    F = np.asarray(feat, np.float64).copy()
    cm = np.nanmean(F, axis=0)
    cm = np.where(np.isfinite(cm), cm, 0.0)
    inds = np.where(~np.isfinite(F))
    F[inds] = np.take(cm, inds[1])
    return F


# ── model caches (embedding + layerwise kNN) ──────────────────────────────────────
def _knn(emb, W, k):
    return C.precompute_knn_layers(emb[:, :W], k)


def _eeg_cache(k):
    cache = {}
    def get(model, size, W):
        key = (model, size)
        if key not in cache:
            emb = IC.load_eeg_emb(model, size)
            cache[key] = None if emb is None else dict(emb=emb, nn=_knn(emb, W, k))
        return cache[key]
    return get


def _vision_cache(k):
    cache = {}
    def get(arch, size, family, W):
        key = (arch, size, family)
        if key not in cache:
            emb = IC.load_vision_grid(arch, size, family)
            cache[key] = None if emb is None else dict(emb=emb, nn=_knn(emb, W, k))
        return cache[key]
    return get


def _llm_cache(k):
    cache = {}
    def get(grid, stem, W):
        key = (grid, stem)
        if key not in cache:
            emb = IC.load_llm_grid(grid, stem)
            cache[key] = None if emb is None else dict(emb=emb, nn=_knn(emb, W, k))
        return cache[key]
    return get


def _intra_mknn(get, arch, size, sizes, W, k, **kw):
    """Adjacent-size mKNN of (arch,size) within its own modality, best layer pair."""
    i = sizes.index(size)
    j = IC.intramodal_partner(sizes, i)
    if j is None:
        return np.nan
    a = get(arch, size, W=W, **kw) if kw else get(arch, size, W)
    b = get(arch, sizes[j], W=W, **kw) if kw else get(arch, sizes[j], W)
    if a is None or b is None:
        return np.nan
    return float(IC.best_layer_pair(a["nn"], b["nn"])[2])


# ── enumeration ───────────────────────────────────────────────────────────────────
def _eeg_variants(smoke):
    if smoke:
        return [("reve", "base"), ("reve", "large"), ("neurolm", "b"), ("neurolm", "l")]
    return [(m, s) for m in C.EEG for s in C.EEG[m]["sizes"]]


def _vision_variants(smoke):
    if smoke:
        return [("dinov2", "small"), ("dinov2", "large"), ("vjepa2", "large")]
    return [(a, s) for a in C.VISION for s in C.VISION[a]["sizes"]]


def _llm_stems(smoke):
    if smoke:
        return ["bloomz-560m", "bloomz-1b7"]
    return [st for fam in C.LLM for st in C.LLM[fam]]


# ── one sweep axis → rows ─────────────────────────────────────────────────────────
def _probe_pair(P, Q, feat, folds, k):
    """best layer of P vs Q, then ridge-path R² of every feature from P's best layer."""
    lp, lq, cross = IC.best_layer_pair(P["nn"], Q["nn"])
    X = C.l2(P["emb"][lp])[: feat.shape[0]]
    r2b = ridge_path_r2(X, feat, "block", folds)
    r2i = ridge_path_r2(X, feat, "interleaved", folds)
    return cross, r2b, r2i


def sweep_axis(axis, smoke, k, folds):
    rows, names = [], None
    eeg_get, vis_get, llm_get = _eeg_cache(k), _vision_cache(k), _llm_cache(k)

    if axis in ("eeg_vision", "eeg_llm"):
        eegs = _eeg_variants(smoke)
        grids = {m: GF.load_features(m, smoke) for m, _ in eegs}
        partners = _vision_variants(smoke) if axis == "eeg_vision" else \
            [(None, st) for st in _llm_stems(smoke)]
        for (em, esz) in eegs:
            fz = grids.get(em)
            if fz is None:
                print(f"  [skip] no features for grid {em}"); continue
            W = int(fz["W"]); feat = _clean(fz["feat"]); names = list(fz["feat_names"])
            P = eeg_get(em, esz, W)
            if P is None:
                continue
            fam = C.EEG[em]["family"]
            intra = _intra_mknn(eeg_get, em, esz, C.EEG[em]["sizes"], W, k)
            for (pa, psz) in partners:
                if axis == "eeg_vision":
                    Q = vis_get(pa, psz, fam, W); plabel = f"{pa}-{psz}"
                else:
                    Q = llm_get(em, psz, W); plabel = psz
                if Q is None:
                    continue
                cross, r2b, r2i = _probe_pair(P, Q, feat, folds, k)
                rows.append(_row(axis, "eeg", f"{em}-{esz}", em, em, plabel,
                                 cross, intra, names, r2b, r2i))
                print(f"  {axis} {em}-{esz} × {plabel}: cross={cross:.3f} "
                      f"medR2blk={np.nanmedian(r2b):+.3f}")

    elif axis == "vision_llm":
        visions = _vision_variants(smoke)
        fz = GF.load_features("clip4s", smoke)
        if fz is None:
            raise SystemExit("build clip4s features first (gridsearch_features.py --grid clip4s)")
        W = int(fz["W"]); feat = _clean(fz["feat"]); names = list(fz["feat_names"])
        llms = _llm_stems(smoke)
        for (va, vsz) in visions:
            P = vis_get(va, vsz, "clip4s", W)
            if P is None:
                continue
            fam = va
            intra = _intra_mknn(vis_get, va, vsz, C.VISION[va]["sizes"], W, k,
                                family="clip4s")
            for st in llms:
                Q = llm_get("clip4s", st, W)
                if Q is None:
                    continue
                cross, r2b, r2i = _probe_pair(P, Q, feat, folds, k)
                rows.append(_row(axis, "vision", f"{va}-{vsz}", va, fam, st,
                                 cross, intra, names, r2b, r2i))
                print(f"  {axis} {va}-{vsz} × {st}: cross={cross:.3f} "
                      f"medR2blk={np.nanmedian(r2b):+.3f}")
    else:
        raise SystemExit(f"unknown axis {axis}")
    return rows, names


def _row(axis, probe, model, arch, family, partner, cross, intra, names, r2b, r2i):
    r = dict(axis=axis, probe=probe, model=model, arch=arch, family=family,
             partner=partner, cross_mknn=float(cross), intra_mknn=float(intra))
    for n, b, i in zip(names, r2b, r2i):
        r[f"blk__{n}"] = float(b)
        r[f"int__{n}"] = float(i)
    return r


def _save_rows(axis, rows):
    if not rows:
        print(f"  no rows for {axis}"); return
    df = pd.DataFrame(rows)
    IC.save_npz(f"gridsearch/rows__{axis}.npz", **{c: df[c].to_numpy() for c in df.columns})


# ── aggregation ───────────────────────────────────────────────────────────────────
def _load_rows():
    frames = []
    for ax in AXES:
        p = OUT / f"rows__{ax}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            frames.append(pd.DataFrame({c: z[c] for c in z.files}))
    if not frames:
        raise SystemExit("no rows__*.npz — run the sweep first.")
    return pd.concat(frames, ignore_index=True)


def _feature_names(df):
    return [c[5:] for c in df.columns if c.startswith("blk__")]


def aggregate(alpha_sig=0.05):
    df = _load_rows()
    feats = _feature_names(df)
    recs = []
    for ax in sorted(df["axis"].unique()):
        sub = df[df["axis"] == ax]
        for f in feats:
            blk, inte = sub[f"blk__{f}"].to_numpy(float), sub[f"int__{f}"].to_numpy(float)
            rec = dict(axis=ax, feature=f, tier=f.split(":")[0],
                       n=int(np.isfinite(blk).sum()),
                       med_r2_block=float(np.nanmedian(blk)),
                       med_r2_inter=float(np.nanmedian(inte)),
                       frac_pos_block=float(np.nanmean(blk > 0)))
            for mk, tag in [("cross_mknn", "cross"), ("intra_mknn", "intra")]:
                d = pd.DataFrame(dict(R2=blk, MKNN=sub[mk].to_numpy(float),
                                      family=sub["family"].to_numpy()))
                a = IC.ancova_family(d)
                rec[f"slope_{tag}"] = a["slope_adj"]
                rec[f"p_{tag}"] = a["p_mknn"]
                rec[f"eta2_{tag}"] = a["partial_eta2"]
            rec["winner"] = bool(
                rec["med_r2_block"] > 0 and (
                    (rec["slope_cross"] > 0 and rec["p_cross"] < alpha_sig) or
                    (rec["slope_intra"] > 0 and rec["p_intra"] < alpha_sig)))
            recs.append(rec)
    summ = pd.DataFrame(recs)
    IC.save_npz("gridsearch/summary.npz", **{c: summ[c].to_numpy() for c in summ.columns})
    summ.to_csv(OUT / "summary.csv", index=False)
    print(f"  saved → outputs/gridsearch/summary.csv  ({len(summ)} axis×feature rows)")

    win = summ[summ["winner"]].sort_values("med_r2_block", ascending=False)
    print("\n=== WINNERS (median block R²>0 AND positive significant alignment slope) ===")
    if win.empty:
        print("  (none)")
    else:
        for _, r in win.iterrows():
            print(f"  [{r['axis']:>10}] {r['feature']:<22} "
                  f"R²blk={r['med_r2_block']:+.3f} R²int={r['med_r2_inter']:+.3f} | "
                  f"cross β={r['slope_cross']:+.2f}(p={r['p_cross']:.2g}) "
                  f"intra β={r['slope_intra']:+.2f}(p={r['p_intra']:.2g})")
    _plot_decodability(summ)
    _plot_alignment(summ)
    _plot_winners(df, win)
    return summ


# ── plots ─────────────────────────────────────────────────────────────────────────
def _plot_decodability(summ):
    axes_ = sorted(summ["axis"].unique())
    fig, axs = plt.subplots(1, len(axes_), figsize=(5.4 * len(axes_), 7), squeeze=False)
    for ci, ax in enumerate(axes_):
        s = summ[summ["axis"] == ax].sort_values("med_r2_block")
        a = axs[0][ci]
        y = np.arange(len(s))
        a.barh(y, s["med_r2_block"], color=["#2c7fb8" if v > 0 else "#d7301f"
                                            for v in s["med_r2_block"]])
        a.scatter(s["med_r2_inter"], y, c="k", s=12, label="interleaved", zorder=3)
        a.set_yticks(y); a.set_yticklabels(s["feature"], fontsize=6)
        a.axvline(0, color="k", lw=0.6)
        a.set_title(f"{ax}\nmedian R² (bar=block · dot=interleaved)", fontsize=9)
        a.set_xlabel("median R²", fontsize=8); a.tick_params(labelsize=6)
        a.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    IC.savefig(fig, "gridsearch/decodability.png")


def _plot_alignment(summ):
    axes_ = sorted(summ["axis"].unique())
    fig, axs = plt.subplots(1, len(axes_), figsize=(5.4 * len(axes_), 7), squeeze=False)
    for ci, ax in enumerate(axes_):
        s = summ[summ["axis"] == ax].sort_values("slope_cross")
        a = axs[0][ci]
        y = np.arange(len(s))
        colors = ["#1a9850" if (v > 0 and p < 0.05) else
                  "#bd0026" if (v < 0 and p < 0.05) else "#bdbdbd"
                  for v, p in zip(s["slope_cross"], s["p_cross"])]
        a.barh(y, s["slope_cross"], color=colors)
        a.set_yticks(y); a.set_yticklabels(s["feature"], fontsize=6)
        a.axvline(0, color="k", lw=0.6)
        a.set_title(f"{ax}\nfamily-ANCOVA slope: R² vs cross-mKNN\n"
                    f"(green=+sig, red=−sig)", fontsize=8)
        a.set_xlabel("adjusted slope", fontsize=8); a.tick_params(labelsize=6)
    fig.tight_layout()
    IC.savefig(fig, "gridsearch/alignment_slope.png")


def _plot_winners(df, win, max_panels=9):
    if win.empty:
        print("  (no winners to scatter)"); return
    sel = win.head(max_panels)
    nc = min(3, len(sel)); nr = int(np.ceil(len(sel) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr), squeeze=False)
    for i, (_, r) in enumerate(sel.iterrows()):
        a = axs[i // nc][i % nc]
        sub = df[df["axis"] == r["axis"]]
        x = sub[f"blk__{r['feature']}"].to_numpy(float)
        ymk = sub["cross_mknn"].to_numpy(float)
        for _, row in sub.iterrows():
            fam = row["family"]
            a.scatter(row[f"blk__{r['feature']}"], row["cross_mknn"],
                      color=FAM_COLORS.get(fam, "#333"), marker=FAM_MARKERS.get(fam, "o"),
                      s=45, edgecolor="k", lw=0.3)
        m = np.isfinite(x) & np.isfinite(ymk)
        if m.sum() >= 2 and np.ptp(x[m]) > 0:
            b, c = np.polyfit(x[m], ymk[m], 1)
            xs = np.linspace(x[m].min(), x[m].max(), 20)
            a.plot(xs, c + b * xs, "k", lw=1)
        a.set_title(f"{r['axis']} · {r['feature']}\nβ={r['slope_cross']:+.2f} "
                    f"p={r['p_cross']:.2g}", fontsize=7)
        a.set_xlabel("R² (block)", fontsize=8); a.set_ylabel("cross-mKNN", fontsize=8)
        a.tick_params(labelsize=6)
    for j in range(len(sel), nr * nc):
        axs[j // nc][j % nc].axis("off")
    fig.tight_layout()
    IC.savefig(fig, "gridsearch/winners_scatter.png")


def main():
    ap = argparse.ArgumentParser(description="Feature-decodability × alignment grid search.")
    ap.add_argument("--axes", nargs="*", default=AXES, choices=AXES)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)

    if args.aggregate:
        aggregate(); return

    folds = 3 if args.smoke else args.folds
    for ax in args.axes:
        print(f"===== sweep axis: {ax} =====")
        rows, _ = sweep_axis(ax, args.smoke, args.k, folds)
        _save_rows(ax, rows)
    print("Done sweep. Run with --aggregate to summarise + plot.")


if __name__ == "__main__":
    main()
