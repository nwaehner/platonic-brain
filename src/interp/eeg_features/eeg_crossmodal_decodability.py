"""
eeg_crossmodal_decodability.py — does cross-MODAL alignment (EEG ↔ vision, EEG ↔ language)
predict EEG-NICE-feature decodability?

Same performance axis as eeg_feature_alignment.py: best-decoding-layer ridge block-CV R²
on NICE features from the EEG model's own embedding (native grid, mean over 17 features).
Performance is loaded from summary.npz — run eeg_feature_alignment.py first.

Alignment axis: aristotelian-calibrated mKNN and CKA between each EEG model and every
vision (or LLM) partner on the EEG model's NATIVE window grid. No to_common() needed:
partner embeddings are already extracted on the EEG grid, so both spaces share the same W.
Calibration (K=100 permutation null) corrects for the chance level that varies with W,
making scores comparable across families (femba W=2160 vs reve W=1080). See NOTES_CROSS_ARCH.md.

Plots (→ outputs/eeg_alignment/):
  Aggregated (one point per EEG variant, y = mean calibrated score over all partners):
    {vision,llm}_crossmodal_{mknn,cka}_aggregated.png
  Disaggregated (subplot per partner, one point per EEG variant):
    {vision,llm}_crossmodal_{mknn,cka}_disaggregated.png
  Data:
    summary_crossmodal.npz

Usage:
  python src/interp/eeg_features/eeg_crossmodal_decodability.py
  python src/interp/eeg_features/eeg_crossmodal_decodability.py --modality vision
  python src/interp/eeg_features/eeg_crossmodal_decodability.py --modality llm --k 5 --K 100
"""

from __future__ import annotations

# bootstrap: add interp root + all study subdirs to sys.path
import sys as _sys; from pathlib import Path as _Path
_INTERP = _Path(__file__).resolve().parent.parent
for _d in ([_INTERP] + [p for p in _INTERP.iterdir() if p.is_dir() and p.name[0] not in "._o"]):
    if str(_d) not in _sys.path: _sys.path.insert(0, str(_d))

import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import _interp_common as IC
import _common as C

OUT = IC.OUT_DIR / "eeg_alignment"
SUMMARY_IN = OUT / "summary.npz"

FAM_COLORS  = {"femba": "#1b9e77", "luna": "#d95f02", "neurolm": "#7570b3",
               "reve": "#e7298a", "steegformer": "#66a61e"}
FAM_MARKERS = {"femba": "o", "luna": "s", "neurolm": "^", "reve": "D", "steegformer": "P"}


# ── aristotelian calibration ───────────────────────────────────────────────────────
def _calibrate(T_obs, T_null, alpha=0.05):
    tau   = float(np.quantile(T_null, 1 - alpha))
    s_cal = max(T_obs - tau, 0.0) / (1 - tau + 1e-12)
    p_val = float((1 + (np.asarray(T_null) >= T_obs).sum()) / (len(T_null) + 1))
    return tau, float(s_cal), p_val


def _aristotelian_mknn(nn_a, nn_b, K, alpha, seed):
    """Best-layer-pair mKNN + K-permutation null calibration.
    nn_a (LA, W, k), nn_b (LB, W, k) → (T_obs, s_cal, tau, p_val)."""
    T_obs = C.max_mknn_over_layers(nn_a, nn_b)
    rng   = np.random.default_rng(seed)
    W     = nn_a.shape[1]
    T_null = np.zeros(K)
    for ki in range(K):
        perm        = rng.permutation(W)
        inv         = np.argsort(perm)
        b_perm      = inv[nn_b[:, perm, :]]
        T_null[ki]  = C.max_mknn_over_layers(nn_a, b_perm)
    tau, s_cal, p_val = _calibrate(T_obs, T_null, alpha)
    return float(T_obs), s_cal, tau, p_val


def _aristotelian_cka(grams_a, grams_b, K, alpha, seed):
    """Best-layer-pair linear CKA + K-permutation null calibration.
    grams_a/b: lists of normalised (W, W) Gram matrices → (T_obs, s_cal, tau, p_val)."""
    def _best(ga, gb):
        return max(float(np.sum(Ga * Gb)) for Ga in ga for Gb in gb)

    T_obs  = _best(grams_a, grams_b)
    rng    = np.random.default_rng(seed)
    W      = grams_a[0].shape[0]
    T_null = np.zeros(K)
    for ki in range(K):
        perm       = rng.permutation(W)
        gb_perm    = [G[np.ix_(perm, perm)] for G in grams_b]
        T_null[ki] = _best(grams_a, gb_perm)
    tau, s_cal, p_val = _calibrate(T_obs, T_null, alpha)
    return float(T_obs), s_cal, tau, p_val


def _norm_gram(X):
    Xc = X - X.mean(0, keepdims=True)
    G  = Xc @ Xc.T
    return G / (np.linalg.norm(G) + 1e-20)


def _encode(emb, k):
    """emb (L, W, D) → layerwise kNN (L, W, k) + list of L normalised Gram matrices."""
    emb  = np.asarray(emb, np.float64)
    nn   = C.precompute_knn_layers(emb, k)
    grams = [_norm_gram(C.l2(emb[l])) for l in range(emb.shape[0])]
    return nn, grams


# ── model enumeration ─────────────────────────────────────────────────────────────
_SMOKE_EEG     = [("reve", "base"), ("neurolm", "b"), ("femba", "base")]
_SMOKE_VISION  = [("dinov2", "large"), ("vjepa2", "large")]
_SMOKE_LLM     = ["bloomz-560m", "bloomz-1b7"]


def _eeg_variants(smoke=False):
    if smoke:
        return _SMOKE_EEG
    return [(m, s) for m in C.EEG for s in C.EEG[m]["sizes"]]

def _vision_partners(smoke=False):
    if smoke:
        return _SMOKE_VISION
    return [(a, s) for a in C.VISION for s in C.VISION[a]["sizes"]]

def _llm_stems(smoke=False):
    if smoke:
        return _SMOKE_LLM
    return [st for fam in C.LLM for st in C.LLM[fam]]


# ── pairwise computation ──────────────────────────────────────────────────────────
def compute_pairs(modality, args):
    """Aristotelian mKNN + CKA for every (EEG variant × partner) pair.
    Performance is loaded from the cross-arch summary.npz."""
    if not SUMMARY_IN.exists():
        raise SystemExit(
            f"summary.npz not found at {SUMMARY_IN}.\n"
            "Run eeg_feature_alignment.py first to compute EEG-feature decodability.")
    z        = np.load(SUMMARY_IN, allow_pickle=True)
    perf_map = {str(k): float(v) for k, v in zip(z["keys"], z["perf_mean"])}

    smoke    = getattr(args, "smoke", False)
    partners = _vision_partners(smoke) if modality == "vision" else _llm_stems(smoke)
    emb_cache = {}   # eeg_key → (raw_emb float64, W)
    enc_cache = {}   # (eeg_key, k) → (nn, grams)
    rows = []

    for (em, esz) in _eeg_variants(smoke):
        eeg_key = f"{em}-{esz}"
        perf    = perf_map.get(eeg_key, np.nan)
        if not np.isfinite(perf):
            print(f"  [skip] {eeg_key}: no perf entry in summary.npz"); continue

        # load + cache EEG embedding
        if eeg_key not in emb_cache:
            raw = IC.load_eeg_emb(em, esz)
            emb_cache[eeg_key] = None if raw is None else (raw.astype(np.float64), raw.shape[1])
        cached = emb_cache[eeg_key]
        if cached is None:
            print(f"  [skip] {eeg_key}: embedding unavailable"); continue
        emb_e, W = cached

        # encode EEG (kNN + grams) once per (key, k)
        enc_key = (eeg_key, args.k)
        if enc_key not in enc_cache:
            enc_cache[enc_key] = _encode(emb_e, args.k)
        nn_e, grams_e = enc_cache[enc_key]
        fam = C.EEG[em]["family"]

        for partner in partners:
            if modality == "vision":
                va, vsz = partner
                emb_p   = IC.load_vision_grid(va, vsz, fam)
                plabel  = f"{va}-{vsz}"
                pfam    = va
            else:
                stem   = partner
                emb_p  = IC.load_llm_grid(em, stem)
                plabel = stem
                pfam   = C.LLM_FAMILY_OF.get(stem, stem.split("-")[0])

            if emb_p is None:
                continue

            # align window counts (partner should match EEG W; warn if not)
            Wp = emb_p.shape[1]
            if Wp != W:
                print(f"    [warn] {plabel} W={Wp} ≠ EEG W={W}; truncating to min")
                Wp       = min(W, Wp)
                emb_p    = emb_p[:, :Wp].astype(np.float64)
                nn_e_use = nn_e[:, :Wp]
                grams_eu = [_norm_gram(C.l2(emb_e[:, :Wp][l])) for l in range(emb_e.shape[0])]
            else:
                emb_p    = emb_p.astype(np.float64)
                nn_e_use = nn_e
                grams_eu = grams_e

            nn_p, grams_p = _encode(emb_p, args.k)

            seed_mk = abs(hash((eeg_key, plabel, "mk"))) & 0x7FFFFFFF
            seed_ck = abs(hash((eeg_key, plabel, "ck"))) & 0x7FFFFFFF

            mk_obs, mk_cal, mk_tau, mk_p = _aristotelian_mknn(
                nn_e_use, nn_p, args.K, args.alpha, seed_mk)
            ck_obs, ck_cal, ck_tau, ck_p = _aristotelian_cka(
                grams_eu, grams_p, args.K, args.alpha, seed_ck)

            rows.append(dict(
                eeg=eeg_key, eeg_model=em, eeg_size=esz, family=fam,
                partner=plabel, partner_family=pfam,
                perf=float(perf),
                mk_obs=mk_obs, mk_cal=mk_cal, mk_tau=mk_tau, mk_p=mk_p,
                ck_obs=ck_obs, ck_cal=ck_cal, ck_tau=ck_tau, ck_p=ck_p,
            ))
            print(f"  {eeg_key} ↔ {plabel}: "
                  f"mKNN={mk_obs:.3f}(cal={mk_cal:.3f}) "
                  f"CKA={ck_obs:.3f}(cal={ck_cal:.3f})", flush=True)

        # free this EEG model's large buffers (gram matrices are W×W float64, ~450 MB
        # for femba/luna at W=2160); only one model needs to be live at a time
        enc_cache.pop(enc_key, None)
        emb_cache[eeg_key] = None

    return rows


# ── aggregation ───────────────────────────────────────────────────────────────────
def aggregate(rows):
    """Per-EEG-model mean calibrated score over all partners of the same modality."""
    acc_mk, acc_ck, meta = defaultdict(list), defaultdict(list), {}
    for r in rows:
        k = r["eeg"]
        acc_mk[k].append(r["mk_cal"])
        acc_ck[k].append(r["ck_cal"])
        meta[k] = (r["family"], r["perf"])
    return {k: dict(family=meta[k][0], perf=meta[k][1],
                    mean_mk=float(np.nanmean(acc_mk[k])),
                    mean_ck=float(np.nanmean(acc_ck[k])))
            for k in acc_mk}


# ── scatter primitive ──────────────────────────────────────────────────────────────
def _scatter(ax, xs, ys, fams):
    """Scatter plot with EEG-family colours + dashed linear fit + Spearman ρ."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for x, y, fam in zip(xs, ys, fams):
        ax.scatter(x, y, s=60, color=FAM_COLORS.get(fam, "#333"),
                   marker=FAM_MARKERS.get(fam, "o"), edgecolor="k", lw=0.4)
    m = np.isfinite(xs) & np.isfinite(ys)
    rho, p = spearmanr(xs[m], ys[m]) if m.sum() >= 3 else (np.nan, np.nan)
    if m.sum() >= 2 and np.ptp(xs[m]) > 0:
        b, a = np.polyfit(xs[m], ys[m], 1)
        xf   = np.linspace(xs[m].min(), xs[m].max(), 20)
        ax.plot(xf, a + b * xf, "k--", lw=1)
    return float(rho), float(p)


# ── aggregated plot ────────────────────────────────────────────────────────────────
def _plot_aggregated(agg, measure, modality, path):
    """One point per EEG variant; y = mean calibrated score across all partners."""
    col   = f"mean_{measure}"
    label = "mKNN (cal.)" if measure == "mk" else "CKA (cal.)"
    keys  = list(agg)
    xs    = [agg[k]["perf"]  for k in keys]
    ys    = [agg[k][col]     for k in keys]
    fams  = [agg[k]["family"] for k in keys]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    rho, p  = _scatter(ax, xs, ys, fams)
    ax.set_xlabel("performance  (mean EEG-feature R²)", fontsize=10)
    ax.set_ylabel(f"EEG ↔ {modality}  mean {label}", fontsize=10)
    ax.set_title(
        f"Cross-modal alignment ({modality}) vs EEG decodability — aggregated\n"
        f"Spearman ρ = {rho:+.3f}   p = {p:.3g}", fontsize=10)
    present = set(fams)
    handles = [plt.Line2D([], [], marker=FAM_MARKERS[f], ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in FAM_COLORS if f in present]
    ax.legend(handles=handles, fontsize=7)
    fig.tight_layout()
    IC.savefig(fig, path)


# ── disaggregated plot ─────────────────────────────────────────────────────────────
def _plot_disaggregated(rows, measure, modality, path):
    """Subplot per partner; one EEG-variant point per subplot."""
    col      = "mk_cal" if measure == "mk" else "ck_cal"
    label    = "mKNN (cal.)" if measure == "mk" else "CKA (cal.)"
    partners = sorted(set(r["partner"] for r in rows))
    nc       = min(4, len(partners))
    nr       = int(np.ceil(len(partners) / nc))

    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.0 * nr), squeeze=False)
    for pi, partner in enumerate(partners):
        ax  = axes[pi // nc][pi % nc]
        sub = [r for r in rows if r["partner"] == partner]
        xs  = [r["perf"] for r in sub]
        ys  = [r[col]    for r in sub]
        fms = [r["family"] for r in sub]
        rho, p = _scatter(ax, xs, ys, fms)
        ax.set_title(f"{partner}\nρ = {rho:+.3f}  p = {p:.2g}", fontsize=8)
        if pi // nc == nr - 1:
            ax.set_xlabel("mean EEG-feature R²", fontsize=7)
        if pi % nc == 0:
            ax.set_ylabel(label, fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(len(partners), nr * nc):
        axes[j // nc][j % nc].axis("off")

    present = set(r["family"] for r in rows)
    handles = [plt.Line2D([], [], marker=FAM_MARKERS[f], ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in FAM_COLORS if f in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
               fontsize=7, bbox_to_anchor=(0.5, -0.01))
    tag = "mKNN" if measure == "mk" else "CKA"
    fig.suptitle(
        f"EEG ↔ {modality}: calibrated {tag} vs EEG decodability  (per partner)",
        fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    IC.savefig(fig, path)


# ── per-feature × per-partner plot ────────────────────────────────────────────────
def _plot_perfeature_partner(sub_rows, perf_map_feat, feat_names, measure, modality, partner, path):
    """Grid of F subplots (one per NICE feature); y = calibrated score to this partner,
    x = per-feature R² for each EEG model.  Like crossarch_perfeature_*.png but y-axis
    is the alignment to one specific vision/LLM model rather than the cross-arch mean."""
    col   = "mk_cal" if measure == "mk" else "ck_cal"
    label = "mKNN (cal.)" if measure == "mk" else "CKA (cal.)"
    F     = len(feat_names)
    nc    = 4
    nr    = int(np.ceil(F / nc))

    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.0 * nr), squeeze=False)
    for fi, feat in enumerate(feat_names):
        ax = axes[fi // nc][fi % nc]
        xs, ys, fms = [], [], []
        for r in sub_rows:
            key = r["eeg"]
            pf  = perf_map_feat.get(key)
            if pf is None or not np.isfinite(pf[fi]):
                continue
            xs.append(float(pf[fi]))
            ys.append(r[col])
            fms.append(r["family"])
        rho, p = _scatter(ax, xs, ys, fms)
        sig = "*" if p < 0.05 else ""
        ax.set_title(f"{feat}{sig}\nρ={rho:+.2f}  p={p:.2g}", fontsize=8)
        if fi // nc == nr - 1:
            ax.set_xlabel("R²", fontsize=7)
        if fi % nc == 0:
            ax.set_ylabel(label, fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(F, nr * nc):
        axes[j // nc][j % nc].axis("off")

    present = set(r["family"] for r in sub_rows)
    handles = [plt.Line2D([], [], marker=FAM_MARKERS[f], ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in FAM_COLORS if f in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
               fontsize=7, bbox_to_anchor=(0.5, -0.01))
    tag = "mKNN" if measure == "mk" else "CKA"
    fig.suptitle(
        f"EEG ↔ {partner}:  per-feature {tag} vs EEG decodability  (each point = one EEG model)",
        fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    IC.savefig(fig, path)


# ── ANCOVA helpers ────────────────────────────────────────────────────────────────
def _ancova_scatter(ax, xs, ys, fams, beta, p):
    """Scatter of within-family residuals + dashed OLS line + β/p annotation."""
    import pandas as pd
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for x, y, fam in zip(xs, ys, fams):
        ax.scatter(x, y, s=60, color=FAM_COLORS.get(fam, "#333"),
                   marker=FAM_MARKERS.get(fam, "o"), edgecolor="k", lw=0.4)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() >= 2 and np.ptp(xs[m]) > 0:
        b, a = np.polyfit(xs[m], ys[m], 1)
        xf = np.linspace(xs[m].min(), xs[m].max(), 20)
        ax.plot(xf, a + b * xf, "k--", lw=1)
    sig = "*" if (np.isfinite(p) and p < 0.05) else ""
    ax.set_title(ax.get_title() + f"\nβ={beta:+.3f}  p={p:.3g}{sig}", fontsize=8)


def _run_ancova(xs, ys, fams):
    """Run IC.ancova_family and return (slope_adj, p, resid_mknn, resid_r2)."""
    import pandas as pd
    df = pd.DataFrame({"R2": list(ys), "MKNN": list(xs), "family": list(fams)})
    a = IC.ancova_family(df)
    return float(a["slope_adj"]), float(a["p_mknn"]), a["resid_mknn"], a["resid_r2"]


def _plot_aggregated_ancova(agg, modality, path):
    """Aggregated partial-regression plot: within-family residuals, ANCOVA β and p."""
    keys = list(agg)
    xs   = [agg[k]["mean_mk"] for k in keys]
    ys   = [agg[k]["perf"]    for k in keys]
    fams = [agg[k]["family"]  for k in keys]
    beta, p, rx, ry = _run_ancova(xs, ys, fams)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.set_title(f"Cross-modal alignment ({modality}) vs EEG decodability — aggregated")
    _ancova_scatter(ax, ry, rx, fams, beta, p)
    ax.set_xlabel("Δ mean EEG-feature R²  (within-family residual)", fontsize=10)
    ax.set_ylabel("Δ mKNN  (within-family residual)", fontsize=10)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    present = set(fams)
    handles = [plt.Line2D([], [], marker=FAM_MARKERS[f], ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in FAM_COLORS if f in present]
    ax.legend(handles=handles, fontsize=7)
    fig.tight_layout()
    IC.savefig(fig, path)


def _plot_disaggregated_ancova(rows, modality, path):
    """Disaggregated partial-regression plot: one subplot per partner, ANCOVA per subplot."""
    partners = sorted(set(r["partner"] for r in rows))
    nc = min(4, len(partners))
    nr = int(np.ceil(len(partners) / nc))

    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.0 * nr), squeeze=False)
    for pi, partner in enumerate(partners):
        ax  = axes[pi // nc][pi % nc]
        sub = [r for r in rows if r["partner"] == partner]
        xs  = [r["mk_cal"] for r in sub]
        ys  = [r["perf"]   for r in sub]
        fms = [r["family"] for r in sub]
        beta, p, rx, ry = _run_ancova(xs, ys, fms)
        ax.set_title(f"{partner}", fontsize=8)
        _ancova_scatter(ax, ry, rx, fms, beta, p)
        if pi // nc == nr - 1:
            ax.set_xlabel("Δ EEG-feature R²", fontsize=7)
        if pi % nc == 0:
            ax.set_ylabel("Δ mKNN", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.axhline(0, color="gray", lw=0.4, ls=":")
        ax.axvline(0, color="gray", lw=0.4, ls=":")

    for j in range(len(partners), nr * nc):
        axes[j // nc][j % nc].axis("off")

    present = set(r["family"] for r in rows)
    handles = [plt.Line2D([], [], marker=FAM_MARKERS[f], ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in FAM_COLORS if f in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
               fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"EEG ↔ {modality}: mKNN vs EEG decodability  (within-family residuals)",
                 fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    IC.savefig(fig, path)


def _plot_perfeature_partner_ancova(sub_rows, perf_map_feat, feat_names, modality, partner, path):
    """Per-feature ANCOVA partial-regression plot for one partner."""
    F  = len(feat_names)
    nc = 4
    nr = int(np.ceil(F / nc))

    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.0 * nr), squeeze=False)
    for fi, feat in enumerate(feat_names):
        ax = axes[fi // nc][fi % nc]
        xs, ys, fms = [], [], []
        for r in sub_rows:
            pf = perf_map_feat.get(r["eeg"])
            if pf is None or not np.isfinite(pf[fi]):
                continue
            xs.append(r["mk_cal"])
            ys.append(float(pf[fi]))
            fms.append(r["family"])
        beta, p, rx, ry = _run_ancova(xs, ys, fms)
        ax.set_title(f"{feat}", fontsize=8)
        _ancova_scatter(ax, ry, rx, fms, beta, p)
        if fi // nc == nr - 1:
            ax.set_xlabel("Δ R²", fontsize=7)
        if fi % nc == 0:
            ax.set_ylabel("Δ mKNN", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.axhline(0, color="gray", lw=0.4, ls=":")
        ax.axvline(0, color="gray", lw=0.4, ls=":")

    for j in range(F, nr * nc):
        axes[j // nc][j % nc].axis("off")

    present = set(r["family"] for r in sub_rows)
    handles = [plt.Line2D([], [], marker=FAM_MARKERS[f], ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in FAM_COLORS if f in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
               fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"EEG ↔ {partner}:  per-feature mKNN vs decodability  (within-family residuals)",
        fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    IC.savefig(fig, path)


# ── --from-hf: build everything from pre-computed gridsearch rows on HuggingFace ───
def _plots_from_hf_gridsearch():
    """Load gridsearch/rows__eeg_{vision,llm}.npz + eeg_alignment/summary.npz from
    triniborrell/platonic-embeddings and generate all crossmodal plots without any
    embedding downloads or permutation tests.

    Alignment axis: raw cross_mknn from the gridsearch (same as our computed mKNN but
    without aristotelian calibration — the correction is small, ~4-8%).
    Performance axis: mean NICE-feature R² from eeg_alignment/summary.npz.
    """
    from huggingface_hub import hf_hub_download

    REPO = IC.REPO_4S

    # ── load per-feature NICE R² ─────────────────────────────────────────────
    dl = hf_hub_download(REPO, "eeg_alignment/summary.npz", repo_type="dataset",
                         token=C.HF_TOKEN_CACHE)
    z_sum = np.load(dl, allow_pickle=True)
    feat_names    = list(z_sum["feat_names"])
    perf_map_feat = {str(k): v for k, v in zip(z_sum["keys"], z_sum["perf"])}
    perf_mean_map = {str(k): float(v) for k, v in zip(z_sum["keys"], z_sum["perf_mean"])}

    modality_files = {
        "vision": "gridsearch/rows__eeg_vision.npz",
        "llm":    "gridsearch/rows__eeg_llm.npz",
    }

    all_rows = {}
    for mod, hf_path in modality_files.items():
        print(f"\n=== Loading {mod} from HF ===")
        dl = hf_hub_download(REPO, hf_path, repo_type="dataset", token=C.HF_TOKEN_CACHE)
        z  = np.load(dl, allow_pickle=True)

        # build rows in the same schema as compute_pairs()
        n = len(z["model"])
        rows = []
        for i in range(n):
            eeg_key = str(z["model"][i])
            partner = str(z["partner"][i])
            fam     = str(z["family"][i])
            mk_raw  = float(z["cross_mknn"][i])
            perf    = perf_mean_map.get(eeg_key, np.nan)
            if not np.isfinite(perf):
                continue
            rows.append(dict(
                eeg=eeg_key, eeg_model=str(z["arch"][i]),
                eeg_size=eeg_key.split("-", 1)[1] if "-" in eeg_key else "",
                family=fam, partner=partner,
                partner_family=partner.split("-")[0],
                perf=perf,
                # use raw mKNN as both observed and "calibrated" (no permutation null here)
                mk_obs=mk_raw, mk_cal=mk_raw, mk_tau=np.nan, mk_p=np.nan,
                # CKA not available in gridsearch rows
                ck_obs=np.nan, ck_cal=np.nan, ck_tau=np.nan, ck_p=np.nan,
            ))

        all_rows[mod] = rows
        agg = aggregate(rows)
        print(f"  {len(rows)} pairs across {len(agg)} EEG models")
        for k, v in sorted(agg.items()):
            print(f"  {k:<20} perf={v['perf']:+.3f}  mean_mKNN={v['mean_mk']:.4f}")

        partners_all = sorted(set(r["partner"] for r in rows))
        # only mKNN (CKA not available in gridsearch rows)
        for measure in ("mk",):
            tag = "mknn"
            _plot_aggregated(agg, measure, mod,
                             f"eeg_alignment/{mod}_crossmodal_{tag}_aggregated.png")
            _plot_disaggregated(rows, measure, mod,
                                f"eeg_alignment/{mod}_crossmodal_{tag}_disaggregated.png")
            _plot_aggregated_ancova(agg, mod,
                             f"eeg_alignment/{mod}_crossmodal_{tag}_aggregated_no-arch-confounders.png")
            _plot_disaggregated_ancova(rows, mod,
                                f"eeg_alignment/{mod}_crossmodal_{tag}_disaggregated_no-arch-confounders.png")
            for partner in partners_all:
                sub  = [r for r in rows if r["partner"] == partner]
                safe = partner.replace("/", "_").replace(" ", "_")
                _plot_perfeature_partner(
                    sub, perf_map_feat, feat_names, measure, mod, partner,
                    f"eeg_alignment/{mod}_{safe}_crossmodal_{tag}_disaggregated.png")
                _plot_perfeature_partner_ancova(
                    sub, perf_map_feat, feat_names, mod, partner,
                    f"eeg_alignment/{mod}_{safe}_crossmodal_{tag}_disaggregated_no-arch-confounders.png")

    # save merged summary
    save_kw: dict = {}
    existing = OUT / "summary_crossmodal.npz"
    if existing.exists():
        z_ex = np.load(existing, allow_pickle=True)
        save_kw.update({k: z_ex[k] for k in z_ex.files})
    for mod, rows in all_rows.items():
        if not rows:
            continue
        for col in rows[0]:
            vals = [r[col] for r in rows]
            try:
                save_kw[f"{mod}__{col}"] = np.array(vals)
            except (ValueError, TypeError):
                save_kw[f"{mod}__{col}"] = np.array(vals, dtype=object)
    if save_kw:
        IC.save_npz("eeg_alignment/summary_crossmodal.npz", **save_kw)
    print("\nDone.")


# ── plot-only mode: reconstruct rows from saved npz and regenerate all plots ───────
def _replot_from_npz():
    """Load summary_crossmodal.npz + summary.npz and regenerate every plot without
    rerunning any HF downloads or permutation tests."""
    npz_path = OUT / "summary_crossmodal.npz"
    if not npz_path.exists():
        raise SystemExit(f"summary_crossmodal.npz not found at {npz_path}. Run without --plot-only first.")
    z = np.load(npz_path, allow_pickle=True)

    z_sum = np.load(SUMMARY_IN, allow_pickle=True)
    feat_names    = list(z_sum["feat_names"])
    perf_map_feat = {str(k): v for k, v in zip(z_sum["keys"], z_sum["perf"])}

    # detect which modalities are stored
    mods = sorted({k.split("__")[0] for k in z.files})
    print(f"Replotting modalities: {mods}")

    for mod in mods:
        prefix = f"{mod}__"
        cols   = [k[len(prefix):] for k in z.files if k.startswith(prefix)]
        if not cols:
            continue
        n    = len(z[f"{prefix}{cols[0]}"])
        rows = []
        for i in range(n):
            row = {}
            for col in cols:
                val = z[f"{prefix}{col}"][i]
                row[col] = val.item() if hasattr(val, "item") else str(val)
            rows.append(row)
        # re-cast numeric fields
        for col in ("perf", "mk_obs", "mk_cal", "mk_tau", "mk_p",
                    "ck_obs", "ck_cal", "ck_tau", "ck_p"):
            for r in rows:
                if col in r:
                    r[col] = float(r[col])

        agg = aggregate(rows)
        partners_all = sorted(set(r["partner"] for r in rows))
        for measure in ("mk", "ck"):
            tag = "mknn" if measure == "mk" else "cka"
            _plot_aggregated(agg, measure, mod,
                             f"eeg_alignment/{mod}_crossmodal_{tag}_aggregated.png")
            _plot_disaggregated(rows, measure, mod,
                                f"eeg_alignment/{mod}_crossmodal_{tag}_disaggregated.png")
            for partner in partners_all:
                sub  = [r for r in rows if r["partner"] == partner]
                safe = partner.replace("/", "_").replace(" ", "_")
                _plot_perfeature_partner(
                    sub, perf_map_feat, feat_names, measure, mod, partner,
                    f"eeg_alignment/{mod}_{safe}_crossmodal_{tag}_disaggregated.png")
    print("Done.")


# ── main ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Cross-modal alignment (aristotelian) vs NICE-feature decodability.")
    ap.add_argument("--modality", choices=["vision", "llm", "both"], default="both",
                    help="which partner modality to evaluate")
    ap.add_argument("--k",     type=int,   default=C.K_MKNN_DEFAULT,
                    help="kNN k for mKNN computation (default %(default)s)")
    ap.add_argument("--K",     type=int,   default=100,
                    help="number of permutation draws for calibration (default %(default)s)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance level for aristotelian calibration (default %(default)s)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick smoke test: 3 EEG × 2 vision × 2 LLM models, K=10")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip computation; regenerate all plots from existing summary_crossmodal.npz")
    ap.add_argument("--from-hf", action="store_true",
                    help="build plots from gridsearch rows + summary.npz already on HuggingFace "
                         "(no embedding downloads or permutation tests required)")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.K = 10
        print("=== SMOKE MODE: 3 EEG × 2 vision × 2 LLM, K=10 ===")

    if args.plot_only:
        _replot_from_npz()
        return

    if args.from_hf:
        _plots_from_hf_gridsearch()
        return

    modalities = ["vision", "llm"] if args.modality == "both" else [args.modality]
    all_rows: dict[str, list] = {}

    for mod in modalities:
        print(f"\n{'=' * 60}\n  MODALITY: {mod.upper()}  "
              f"(k={args.k}  K={args.K}  α={args.alpha})\n{'=' * 60}")
        rows = compute_pairs(mod, args)
        all_rows[mod] = rows
        if not rows:
            print(f"  no data for {mod}"); continue

        agg = aggregate(rows)
        print(f"\n=== AGGREGATED ({mod}) ===")
        for k, v in sorted(agg.items()):
            print(f"  {k:<20} perf={v['perf']:+.3f}  "
                  f"mean_mKNN={v['mean_mk']:.3f}  mean_CKA={v['mean_ck']:.3f}")

        # load per-feature R² from summary.npz for perfeature plots
        z_sum = np.load(SUMMARY_IN, allow_pickle=True)
        feat_names    = list(z_sum["feat_names"])
        perf_map_feat = {str(k): v for k, v in zip(z_sum["keys"], z_sum["perf"])}

        partners_all = sorted(set(r["partner"] for r in rows))
        for measure in ("mk", "ck"):
            tag = "mknn" if measure == "mk" else "cka"
            _plot_aggregated(agg, measure, mod,
                             f"eeg_alignment/{mod}_crossmodal_{tag}_aggregated.png")
            _plot_disaggregated(rows, measure, mod,
                                f"eeg_alignment/{mod}_crossmodal_{tag}_disaggregated.png")
            # per-partner × per-feature grids
            for partner in partners_all:
                sub = [r for r in rows if r["partner"] == partner]
                safe = partner.replace("/", "_").replace(" ", "_")
                _plot_perfeature_partner(
                    sub, perf_map_feat, feat_names, measure, mod, partner,
                    f"eeg_alignment/{mod}_{safe}_crossmodal_{tag}_disaggregated.png")

    # persist pairwise results — merge with any existing data (other modality may have run first)
    save_kw: dict = {}
    existing = OUT / "summary_crossmodal.npz"
    if existing.exists():
        z_ex = np.load(existing, allow_pickle=True)
        save_kw.update({k: z_ex[k] for k in z_ex.files})
    for mod, rows in all_rows.items():
        if not rows:
            continue
        for col in rows[0]:
            vals = [r[col] for r in rows]
            try:
                save_kw[f"{mod}__{col}"] = np.array(vals)
            except (ValueError, TypeError):
                save_kw[f"{mod}__{col}"] = np.array(vals, dtype=object)
    if save_kw:
        IC.save_npz("eeg_alignment/summary_crossmodal.npz", **save_kw)

    print("\nDone.")


if __name__ == "__main__":
    main()
