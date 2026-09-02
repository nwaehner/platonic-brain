"""
_paper_common.py — shared toolkit for the three paper-plot blocks.

Everything is reused from the existing analysis code; this module only adds the glue
the paper figures need (a generic ANCOVA that also returns R², a Spearman helper, the
family palette, the output-dir layout, and loaders for the three cached artifacts).

Blocks:
  (a) block_a.py — cross-arch / cross-modal  (LLM → EEG), perf-axis curves + ANCOVA(size).
  (b) block_b.py — intra-arch / same-modal   (EEG), R² vs intramodal alignment + Spearman.
  (c) block_c.py — cross-modal per-variant    (EEG ↔ LLM), Δ-vs-Δ ANCOVA R2 ~ align + C(family).

Cached compute artifacts (produced by compute_*.py, read by block_*.py) live in
`data/paper-plots/_cache/`. Once present, all plotting is offline and fast.
"""

from __future__ import annotations

# ── path bootstrap: reuse alignment_plots/_common + interp helpers ─────────────────
import sys as _sys
from pathlib import Path as _Path

_THIS = _Path(__file__).resolve()
_SRC = _THIS.parent.parent                         # …/platonic-brain/src
_INTERP = _SRC / "interp"
# add interp root + its study subdirs (this pulls in eeg_features/, feature_search/, …)
_dirs = [_INTERP] + [p for p in _INTERP.iterdir() if p.is_dir() and p.name[0] not in "._"]
_dirs += [_SRC / "alignment_plots"]
for _d in _dirs:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import numpy as np
from scipy.stats import spearmanr

import _interp_common as IC          # noqa: E402  (also puts alignment_plots on the path)
import _common as C                  # noqa: E402  (alignment_plots/_common.py)


# ── family palette / markers (single source, matches plot_eeg_perfeature.py) ───────
FAM_COLORS = {"femba": "#1b9e77", "luna": "#d95f02", "neurolm": "#7570b3",
              "reve": "#e7298a", "steegformer": "#66a61e"}
FAM_MARKERS = {"femba": "o", "luna": "s", "neurolm": "^", "reve": "D", "steegformer": "P"}

# Feature tiers used across (b) and (c). The old `mid` (142 CLIP object-presence) and `high`
# (39 Empath) visual tiers are replaced by `semantic`: the ModernBERT [CLS] encoding of the
# captions covering each window (compute_bert_semantic.py).
TIERS = ["eeg", "low", "semantic"]
TIER_LABEL = {"eeg": "EEG-derived", "low": "Low-level Visual",
              "semantic": "Semantic (BERT)"}

# Tier whose columns the per-feature ("disaggregated") panels iterate. Raw BERT dimensions
# carry no standalone meaning, so `semantic` disaggregates over its top-24 PCA components,
# which compute_decodability.py stores under the parallel tier `semantic_pca`.
DISAGG_TIER = {"semantic": "semantic_pca"}


def disagg_tier(tier):
    return DISAGG_TIER.get(tier, tier)

# LLM stems that actually have a measured 1−BPB (block a x-axis); order low→high perf.
LLM_STEMS = [s for fam in C.LLM for s in C.LLM[fam] if C.LLM_PERF.get(s) is not None]

# pretty x-tick names for block (a): 'Bloom 560M', 'OpenLLaMA 3B', …
LLM_DISPLAY = {
    "bloomz-560m": "Bloom 560M", "bloomz-1b1": "Bloom 1B1", "bloomz-1b7": "Bloom 1B7",
    "bloomz-3b": "Bloom 3B", "bloomz-7b1": "Bloom 7B1",
    "open_llama_3b": "OpenLLaMA 3B", "open_llama_7b": "OpenLLaMA 7B",
    "open_llama_13b": "OpenLLaMA 13B",
    "llama-13b": "LLaMA 13B", "llama-30b": "LLaMA 33B", "llama-65b": "LLaMA 65B"}

# ── unified EEG-size legend (small/base/large by ordinal rank; end-aligned) ──────────
# 3-size families fill all three slots; 2-size REVE fills the last two (base, large).
# The colours match C.EEG_SIZE_COLORS exactly under this end-alignment.
ORDINAL = ["small", "base", "large"]
ORDINAL_COLOR = {"small": "#c8b400", "base": "#1a9850", "large": "#5b2182"}


def size_ordinal(model, size):
    """Ordinal label ('small'|'base'|'large') for a model's size, end-aligned so a k-size
    family occupies the last k slots (REVE base→'base', large→'large')."""
    sizes = C.EEG[model]["sizes"]
    return ORDINAL[sizes.index(size) + (len(ORDINAL) - len(sizes))]


def size_color(model, size):
    return ORDINAL_COLOR[size_ordinal(model, size)]


def size_legend_handles():
    import matplotlib.pyplot as plt
    return [plt.Line2D([], [], marker="o", ls="", color=ORDINAL_COLOR[o], label=o,
                       markeredgecolor="none") for o in ORDINAL]


# ── output layout ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = _SRC.parent
PAPER_DIR = PROJECT_ROOT / "data" / "paper-plots"
CACHE_DIR = PAPER_DIR / "_cache"
DIR_A = PAPER_DIR / "cross-arch_cross-alignm_LLM-to-EEG"
DIR_A_PERM = PAPER_DIR / "cross-arch_cross-alignm_LLM-to-EEG_perm-test"
DIR_A_NOCONF = PAPER_DIR / "cross-arch_cross-alignm_LLM-to-EEG_no-confound"
DIR_A_WITHCONF = PAPER_DIR / "cross-arch_cross-alignm_LLM-to-EEG_with-confounds"
DIR_A_COMBINED = PAPER_DIR / "cross-arch_cross-alignm_LLM-to-EEG_combined"
DIR_B = PAPER_DIR / "intra-arch_same-modality_EEG"
DIR_C = PAPER_DIR / "cross-modal_diff-arch_EEG-to-LLM"
DIR_C_CONFOUND = PAPER_DIR / "cross-modal_diff-arch_EEG-to-LLM_with-confound"
DIR_C_VIDEO = PAPER_DIR / "cross-modal_diff-arch_EEG-to-Video"
DIR_SIZE = PAPER_DIR / "size-vs-alignment"

# ── the semantic-tier / nested-CV study writes to its OWN folders ──────────────────
# The tiers changed (mid+high → semantic) AND the probe changed (nested subject × time CV),
# so these results are not comparable with the originals above and must not overwrite them —
# `data/paper-plots` is not under version control.
SUFFIX_NESTED = "__semantic-nestedcv"
DIR_B_NESTED = PAPER_DIR / (DIR_B.name + SUFFIX_NESTED)
DIR_C_NESTED = PAPER_DIR / (DIR_C.name + SUFFIX_NESTED)
DIR_C_CONFOUND_NESTED = PAPER_DIR / (DIR_C_CONFOUND.name + SUFFIX_NESTED)
DIR_C_VIDEO_NESTED = PAPER_DIR / (DIR_C_VIDEO.name + SUFFIX_NESTED)

# Original artifact (14 variants × 209 feats, tiers eeg/low/mid/high, old biased probe).
# Kept read-only; a restored copy lives beside it as *__ORIGINAL-pre-nestedcv.npz.
DECODABILITY_ORIG_NPZ = CACHE_DIR / "decodability_allfeatures.npz"
# What compute_decodability.py now writes: 3 tiers, nested subject × time CV.
DECODABILITY_NPZ = CACHE_DIR / "decodability_nestedcv.npz"
CROSSMODAL_NPZ = CACHE_DIR / "crossmodal_llm_calibrated.npz"
CROSSMODAL_VIDEO_NPZ = CACHE_DIR / "crossmodal_video_calibrated.npz"
INTRAMODAL_NPZ = CACHE_DIR / "intramodal_full.npz"


def ensure_dirs(*extra):
    for d in (CACHE_DIR, DIR_A, DIR_B_NESTED, DIR_C_NESTED, *extra):
        d.mkdir(parents=True, exist_ok=True)


# ── EEG variant / pair enumeration ─────────────────────────────────────────────────
def eeg_variants():
    """All (model, size) EEG variants in registry order (14 total)."""
    return [(m, s) for m in C.EEG for s in C.EEG[m]["sizes"]]


def size_pairs():
    """Consecutive size pairs per family, incl reve (base↔large). 9 pairs total.
    Returns list of dict(model, a, b) with a=smaller, b=larger."""
    out = []
    for m in C.EEG:
        sizes = C.EEG[m]["sizes"]
        for a, b in zip(sizes[:-1], sizes[1:]):
            out.append({"model": m, "a": a, "b": b})
    return out


# ── statistics ─────────────────────────────────────────────────────────────────────
def spearman(x, y):
    """Spearman ρ, p over finite pairs; (nan, nan) if <3 usable points."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan, np.nan
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p)


def _resid_slope_maker(y, control_design):
    """Return (M, ry, slope_fn) for fast partial-regression slopes: residualise y and any x
    against `control_design` D (n×p, incl. constant) via M = I − D(DᵀD)⁻¹Dᵀ; slope of x =
    (Mx·My)/(Mx·Mx). Used by perm_p so each permutation is a couple of mat-vecs, not an OLS refit."""
    D = np.asarray(control_design, float)
    M = np.eye(len(D)) - D @ np.linalg.pinv(D.T @ D) @ D.T
    ry = M @ np.asarray(y, float)

    def slope(xx):
        rx = M @ np.asarray(xx, float)
        denom = float(rx @ rx)
        return float(rx @ ry) / denom if denom > 1e-12 else np.nan
    return M, ry, slope


def _design(n, cats):
    """const + first-differenced dummies for each categorical array in `cats`."""
    import pandas as pd
    parts = [np.ones(n)]
    for c in cats:
        d = pd.get_dummies(np.asarray(c), drop_first=True).astype(float).to_numpy()
        if d.size:
            parts.append(d)
    return np.column_stack(parts)


def ancova_multi(y, x, *cats):
    """OLS  y ~ x + C(cat1) + C(cat2) + …  → dict(beta, p, r2_full, n). beta/p are for x."""
    import statsmodels.api as sm
    import pandas as pd
    df = pd.DataFrame({"y": np.asarray(y, float), "x": np.asarray(x, float)})
    for i, c in enumerate(cats):
        df[f"c{i}"] = list(c)
    df = df.dropna()
    out = dict(n=int(len(df)), beta=np.nan, p=np.nan, r2_full=np.nan)
    if len(df) < 3 or df["x"].nunique() < 2:
        return out
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy(float),
                         *[pd.get_dummies(df[f"c{i}"], drop_first=True).astype(float).to_numpy()
                           for i in range(len(cats)) if df[f"c{i}"].nunique() > 1]])
    fit = sm.OLS(df["y"].to_numpy(float), X).fit()
    out.update(beta=float(fit.params[1]), p=float(fit.pvalues[1]), r2_full=float(fit.rsquared))
    return out


def perm_p(y, x, stem_ids, control_cats=(), stem_strata=None, K=10000, seed=0):
    """Assumption-free two-sided p for the performance→alignment slope. `x` (performance) is
    constant within each partner stem, so we permute the mapping stem→perf (all sizes of a stem
    move together) and recompute the slope adjusted for `control_cats` (per-point categoricals,
    e.g. size or size+family). Returns (p, beta_obs); p = (1+#{|β*|≥|β_obs|})/(K+1).

    `stem_strata` (per-point stratum, e.g. LLM family) → **restricted** permutation: perf is
    shuffled only *within* a stratum. This is REQUIRED when a controlled covariate is stem-level
    and collinear with perf (family ≈ perf tier): global shuffling would break the perf↔family
    structure, inflate the null spread, and make the test badly anti-conservative. None → global."""
    y = np.asarray(y, float); x = np.asarray(x, float); stem_ids = np.asarray(stem_ids)
    cats = [np.asarray(c) for c in control_cats]
    m = np.isfinite(y) & np.isfinite(x)
    y, x, stem_ids = y[m], x[m], stem_ids[m]
    cats = [c[m] for c in cats]
    strata = np.asarray(stem_strata)[m] if stem_strata is not None else None
    n = len(y)
    if n < 3 or np.ptp(x) == 0:
        return np.nan, np.nan
    _, _, slope = _resid_slope_maker(y, _design(n, cats))
    b_obs = slope(x)
    if not np.isfinite(b_obs):
        return np.nan, b_obs
    stems = list(dict.fromkeys(stem_ids.tolist()))
    perf_vals = np.array([x[stem_ids == s][0] for s in stems])
    idx_of = {s: i for i, s in enumerate(stems)}
    point_idx = np.array([idx_of[s] for s in stem_ids])
    # groups of stem-indices to permute within (all stems together if unstratified)
    if strata is not None:
        stem_stratum = [strata[stem_ids == s][0] for s in stems]
        groups = [np.array([i for i, g in enumerate(stem_stratum) if g == u])
                  for u in dict.fromkeys(stem_stratum)]
    else:
        groups = [np.arange(len(stems))]
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(K):
        pv = perf_vals.copy()
        for g in groups:
            pv[g] = rng.permutation(pv[g])
        b = slope(pv[point_idx])
        if np.isfinite(b) and abs(b) >= abs(b_obs) - 1e-12:
            cnt += 1
    return (1 + cnt) / (K + 1), b_obs


def per_size_fits(y, x, size):
    """One simple OLS y~x within each size level (9 LLM points each) → list of
    dict(size, beta, p, r2, n). These are the '3 p-values' (one per EEG size)."""
    import statsmodels.api as sm
    y = np.asarray(y, float); x = np.asarray(x, float); size = np.asarray(size)
    out = []
    for s in dict.fromkeys(size.tolist()):
        m = (size == s) & np.isfinite(y) & np.isfinite(x)
        if m.sum() < 3 or np.ptp(x[m]) == 0:
            out.append(dict(size=str(s), beta=np.nan, p=np.nan, r2=np.nan, n=int(m.sum())))
            continue
        fit = sm.OLS(y[m], sm.add_constant(x[m])).fit()
        out.append(dict(size=str(s), beta=float(fit.params[1]), p=float(fit.pvalues[1]),
                        r2=float(fit.rsquared), n=int(m.sum())))
    return out


def ancova(y, x, group):
    """ANCOVA  y ~ x + C(group).  Returns the group-adjusted slope of x and its p,
    the FULL-model R², partial η² for x (type-II SS added last), the within-group
    added-variable residuals (y|group, x|group) for the Δ-vs-Δ scatter, and n.

    Mirrors IC.ancova_family but is generic in the column roles and also returns the
    model R² (which the paper annotations need). Uses explicit dummies, not patsy."""
    import pandas as pd
    import statsmodels.api as sm

    d = pd.DataFrame({"y": np.asarray(y, float), "x": np.asarray(x, float),
                      "g": list(group)}).dropna(subset=["y", "x", "g"])
    out = dict(n=int(len(d)), beta=np.nan, p=np.nan, r2_full=np.nan,
               partial_eta2=np.nan, slope_unadj=np.nan,
               resid_y=np.array([]), resid_x=np.array([]))
    if len(d) < 3 or d["x"].nunique() < 2:
        return out
    yv = d["y"].to_numpy(float)
    xv = d["x"].to_numpy(float)
    out["slope_unadj"] = float(np.polyfit(xv, yv, 1)[0])

    if d["g"].nunique() < 2:
        # no group structure → plain OLS y ~ x
        Xf = sm.add_constant(xv)
        full = sm.OLS(yv, Xf).fit()
        out.update(beta=float(full.params[1]), p=float(full.pvalues[1]),
                   r2_full=float(full.rsquared),
                   resid_y=yv - yv.mean(), resid_x=xv - xv.mean())
        return out

    dum = pd.get_dummies(d["g"], drop_first=True).astype(float).to_numpy()
    Xf = sm.add_constant(dum)                              # group-only design
    Xfull = sm.add_constant(np.column_stack([xv, dum]))    # + x
    full = sm.OLS(yv, Xfull).fit()
    reduced = sm.OLS(yv, Xf).fit()
    ss_x = float(reduced.ssr - full.ssr)                   # type-II SS for x
    out.update(
        beta=float(full.params[1]), p=float(full.pvalues[1]),
        r2_full=float(full.rsquared),
        partial_eta2=float(ss_x / (ss_x + full.ssr + 1e-12)),
        resid_y=yv - reduced.predict(Xf),
        resid_x=xv - sm.OLS(xv, Xf).fit().predict(Xf),
    )
    return out


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


# ── cached-artifact loaders ────────────────────────────────────────────────────────
def load_decodability():
    """decodability_allfeatures.npz → dict(keys, family, feat_names, feat_tier, perf, …).

    perf is (n_variants, F), the mean over the 6 nested-CV folds; rows keyed by 'model-size'
    in `keys`. The optional arrays (present since the nested-CV rewrite) are:
      perf_folds (V,F,6) · perf_sd (V,F) · perf_null (V,F) · best_layer · best_alpha.
    `perf_null` is the intercept-only floor and is NEGATIVE under block CV — R² should be read
    against it, not against zero (see _cache/NEGATIVE_R2_DIAGNOSIS.md)."""
    if not DECODABILITY_NPZ.exists():
        raise SystemExit(f"missing {DECODABILITY_NPZ} — run compute_decodability.py first")
    z = np.load(DECODABILITY_NPZ, allow_pickle=True)
    out = {"keys": [str(k) for k in z["keys"]],
           "family": [str(f) for f in z["family"]],
           "feat_names": [str(n) for n in z["feat_names"]],
           "feat_tier": np.array([str(t) for t in z["feat_tier"]]),
           "perf": z["perf"].astype(float)}
    for opt in ("perf_folds", "perf_sd", "perf_null", "perf_skill",
                "best_layer", "best_alpha"):
        if opt in z.files:
            out[opt] = z[opt].astype(float)
    return out


def tier_mean_r2(dec, tier, key="perf"):
    """Per-variant mean decodability over the features of one tier → dict variant→value.

    `key` selects the array:
      'perf'       raw nested-CV R² (default) — comparable with the pre-nested-CV figures.
      'perf_skill' skill score (R²−floor)/(1−floor): 0 = no better than the intercept-only
                   predictor, 1 = perfect. Prefer this when averaging ACROSS features: their
                   chance floors differ by more than an order of magnitude (−0.12 to −5.86
                   within the eeg tier alone), so a raw-R² mean is dominated by whichever
                   features drift most rather than by what the embedding encodes.
      'perf_sd', 'perf_null'  the fold spread and the chance floor."""
    cols = dec["feat_tier"] == tier
    vals = np.nanmean(dec[key][:, cols], axis=1)
    return {k: float(v) for k, v in zip(dec["keys"], vals)}


X_LABEL = {"perf": "Mean R² (nested CV, 6 folds)",
           "perf_skill": "Mean skill score  (R²−floor)/(1−floor)"}

AT_CHANCE_SKILL = 0.05          # mean skill below this ⇒ the tier carries no signal


def tier_at_chance(dec, tier):
    """(is_at_chance, mean_skill) for a tier, across all variants.

    A tier whose skill is ~0 sits on its own chance floor for EVERY variant, so the x-axis of
    any panel built from it has no real spread and the ρ/β fitted there is fitting noise.
    Panels say so on their face rather than inviting a reader to over-read the line."""
    if "perf_skill" not in dec:
        return False, np.nan
    cols = dec["feat_tier"] == tier
    s = float(np.nanmean(dec["perf_skill"][:, cols]))
    return bool(s < AT_CHANCE_SKILL), s


def x_key(name):
    """'r2' | 'skill' → the cache array name."""
    return {"r2": "perf", "skill": "perf_skill"}[name]


def tier_sem_r2(dec, tier):
    """Per-variant standard ERROR of the tier-mean R² across the nested-CV folds → key→SE.
    Empty dict if the cache predates the nested-CV rewrite (no per-fold array)."""
    if "perf_folds" not in dec:
        return {}
    cols = dec["feat_tier"] == tier
    fold_means = np.nanmean(dec["perf_folds"][:, cols, :], axis=1)      # (V, folds)
    n = np.sum(np.isfinite(fold_means), axis=1)
    se = np.nanstd(fold_means, axis=1) / np.sqrt(np.maximum(n, 1))
    return {k: float(v) for k, v in zip(dec["keys"], se)}


def load_crossmodal(path):
    """Any crossmodal_*_calibrated.npz → list of per-(EEG variant, partner) row dicts."""
    path = _Path(path)
    if not path.exists():
        raise SystemExit(f"missing {path} — run the corresponding compute_crossmodal_* script first")
    z = np.load(path, allow_pickle=True)
    n = len(z["eeg"])
    return [{f: (str(z[f][i]) if z[f].dtype.kind in "US" else float(z[f][i]))
             for f in z.files} for i in range(n)]


def load_crossmodal_llm():
    return load_crossmodal(CROSSMODAL_NPZ)


def load_intramodal():
    """intramodal_full.npz → dict pair_key→dict(model,a,b, mk_cal(S,), ck_cal(S,))."""
    if not INTRAMODAL_NPZ.exists():
        raise SystemExit(f"missing {INTRAMODAL_NPZ} — run compute_intramodal.py first")
    z = np.load(INTRAMODAL_NPZ, allow_pickle=True)
    out = {}
    for key in [str(k) for k in z["pair_keys"]]:
        out[key] = {"model": key.split("__")[0],
                    "a": key.split("__")[1], "b": key.split("__")[2],
                    "mk_cal": z[f"{key}__mk_cal"].astype(float),
                    "ck_cal": z[f"{key}__ck_cal"].astype(float)}
    return out
