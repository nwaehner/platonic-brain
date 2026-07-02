"""
feature_linear_probing.py — Component 3: does alignment predict feature-decodability?

For each (EEG variant × vision model) pair we take that pair's highest-alignment EEG layer
(`best_layer_pair`), probe its embedding (Ridge, temporal-leakage-safe block CV — reused from
attribute_probing) to predict every named feature, and average R² within each feature tier.

Plot (one figure, subplot grid rows = vision models, cols = feature tier):
  • points = EEG variants, colour + marker by family (the 5 EEG architectures), no size glyph
  • x = mean R²,  y = mKNN
  3a  y = cross-modal mKNN (this pair), per-subplot linear fit  R2 ~ MKNN
  3b  family-confound ANCOVA  R2 ~ MKNN + C(family): added-variable plot + adjusted slope/η²
  …-intra  same as 3a/3b but y = the intramodal (adjacent-size) mKNN.

Usage:
  python src/interp/feature_linear_probing.py --smoke
  python src/interp/feature_linear_probing.py --eeg-model all --vision-arch dinov2 vjepa2
"""

from __future__ import annotations

# bootstrap: add interp root + all study subdirs to sys.path
import sys as _sys; from pathlib import Path as _Path
_INTERP = _Path(__file__).resolve().parent.parent
for _d in ([_INTERP] + [p for p in _INTERP.iterdir() if p.is_dir() and p.name[0] not in "._o"]):
    if str(_d) not in _sys.path: _sys.path.insert(0, str(_d))

import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import _interp_common as IC
import _common as C
import interp_features as F
from attribute_probing import block_cv_r2, _zscore
from neighbour_enrichment import _clean

PROBE_DIR = IC.OUT_DIR / "probing"
TIERS = ["low", "highvis", "sem"]
FAM_COLORS = {"femba": "#1b9e77", "luna": "#d95f02", "neurolm": "#7570b3",
              "reve": "#e7298a", "steegformer": "#66a61e",
              "dinov2": "#1b9e77", "videomae": "#d95f02", "videomae_ft": "#7570b3",
              "vjepa2": "#e7298a"}
FAM_MARKERS = {"femba": "o", "luna": "s", "neurolm": "^", "reve": "D", "steegformer": "P",
               "dinov2": "o", "videomae": "s", "videomae_ft": "^", "vjepa2": "D"}


def _eeg_variants(smoke):
    if smoke:                                          # 4 points / 2 families → ANCOVA fits
        return [("reve", "base"), ("reve", "large"), ("neurolm", "b"), ("neurolm", "l")]
    return [(m, s) for m in C.EEG for s in C.EEG[m]["sizes"]]


def _vision_variants(archs, smoke):
    if smoke:
        return [("dinov2", "large")]
    return [(a, s) for a in archs for s in C.VISION[a]["sizes"]]


def _select_features(feats, max_per_tier):
    """Top-variance feature indices per tier (bounds probing cost)."""
    feat = _clean(feats["feat"]); tiers = feats["feat_tier"]
    sel = {}
    for t in TIERS:
        cols = np.where(tiers == t)[0]
        if len(cols) == 0:
            continue
        v = np.nanvar(feat[:, cols], axis=0)
        keep = cols[np.argsort(v)[::-1][:max_per_tier]]
        sel[t] = keep
    return feat, sel


def _eeg_cache(smoke):
    """Lazy per-variant EEG embedding + layerwise kNN, keyed by (model,size)."""
    cache = {}

    def get(model, size, W, k):
        key = (model, size)
        if key not in cache:
            emb = IC.load_eeg_emb(model, size)
            if emb is None:
                cache[key] = None
            else:
                emb = emb[:, :W]
                cache[key] = dict(emb=emb, nn=C.precompute_knn_layers(emb, k))
        return cache[key]
    return get


def intramodal_mknn(get, model, size, W, k):
    sizes = C.EEG[model]["sizes"]
    i = sizes.index(size)
    j = IC.intramodal_partner(sizes, i)
    if j is None:
        return np.nan
    a, b = get(model, size, W, k), get(model, sizes[j], W, k)
    if a is None or b is None:
        return np.nan
    return IC.best_layer_pair(a["nn"], b["nn"])[2]


def build_table(eeg_variants, vision_variants, smoke, k, folds, max_per_tier):
    rows = []
    grids = {m: F.load_features(m, smoke) for m, _ in eeg_variants}
    getf = {m: (_select_features(grids[m], max_per_tier) if grids[m] else None)
            for m in grids}
    get = _eeg_cache(smoke)
    for (em, esz) in eeg_variants:
        if grids[em] is None:
            print(f"  [skip] no feature cache for EEG grid '{em}'")
            continue
        W = int(grids[em]["W"]); feat, sel = getf[em]
        e = get(em, esz, W, k)
        if e is None:
            continue
        fam = C.EEG[em]["family"]                      # registered family (femba_luna)
        intra = intramodal_mknn(get, em, esz, W, k)
        for (va, vsz) in vision_variants:
            vemb = IC.load_vision_grid(va, vsz, fam)
            if vemb is None:
                continue
            nn_v = C.precompute_knn_layers(vemb[:, :W], k)
            le, lv, cross = IC.best_layer_pair(e["nn"], nn_v)
            X = _zscore(C.l2(e["emb"][le]))[:W]
            rec = dict(eeg=f"{em}-{esz}", eeg_model=em, family=em,   # family = 5 archs
                       vision=f"{va}-{vsz}", cross_mknn=float(cross),
                       intra_mknn=float(intra))
            for t, cols in sel.items():
                r2s = [block_cv_r2(X, feat[:, c], folds) for c in cols]
                rec[f"R2_{t}"] = float(np.nanmean(r2s)) if r2s else np.nan
            rows.append(rec)
            print(f"  probe {rec['eeg']} × {rec['vision']}: cross={cross:.3f} "
                  + " ".join(f"{t}={rec.get(f'R2_{t}', np.nan):.3f}" for t in sel))
    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────────
def _present_tiers(df):
    return [t for t in TIERS if f"R2_{t}" in df.columns and df[f"R2_{t}"].notna().any()]


def plot_fit(df, ycol, title, path, row_col="vision", fam_pool=None):
    """3a: scatter x=meanR², y=mKNN; rows=`row_col`, cols=tier; per-subplot linear fit.
    Points coloured by `family`; `fam_pool` lists the families shown in the legend."""
    visions = sorted(df[row_col].unique()); tiers = _present_tiers(df)
    nr, nc = len(visions), max(1, len(tiers))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
    for ri, vis in enumerate(visions):
        for ci, t in enumerate(tiers):
            ax = axes[ri][ci]
            sub = df[df[row_col] == vis]
            x = sub[f"R2_{t}"].to_numpy(); y = sub[ycol].to_numpy()
            for _, row in sub.iterrows():
                fam = row["family"]
                ax.scatter(row[f"R2_{t}"], row[ycol], color=FAM_COLORS.get(fam, "#333"),
                           marker=FAM_MARKERS.get(fam, "o"), s=55, edgecolor="k", lw=0.3)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 2 and np.ptp(x[m]) > 0:
                b, a = np.polyfit(x[m], y[m], 1)
                xs = np.linspace(x[m].min(), x[m].max(), 20)
                ax.plot(xs, a + b * xs, color="k", lw=1)
                r = np.corrcoef(x[m], y[m])[0, 1]
                ax.set_title(f"{vis} · {t}\nslope={b:.2f} r={r:.2f}", fontsize=7)
            else:
                ax.set_title(f"{vis} · {t}\n(n={int(m.sum())})", fontsize=7)
            if ri == nr - 1:
                ax.set_xlabel("mean R²", fontsize=8)
            if ci == 0:
                ax.set_ylabel(ycol, fontsize=8)
            ax.tick_params(labelsize=6)
    pool = fam_pool or [f for f in FAM_COLORS if f in C.EEG]
    handles = [plt.Line2D([], [], marker=FAM_MARKERS.get(f, "o"), ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in pool]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)), fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, y=1.0, fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    IC.savefig(fig, path)


def plot_ancova(df, ycol, title, path, row_col="vision"):
    """3b: family-confound ANCOVA R2~MKNN+C(family). Added-variable (partial-regression)
    plot of resid(R2|family) vs resid(MKNN|family) per subplot, with adjusted slope/η²."""
    visions = sorted(df[row_col].unique()); tiers = _present_tiers(df)
    nr, nc = len(visions), max(1, len(tiers))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
    for ri, vis in enumerate(visions):
        for ci, t in enumerate(tiers):
            ax = axes[ri][ci]
            sub = df[df[row_col] == vis]
            d = pd.DataFrame(dict(R2=sub[f"R2_{t}"].to_numpy(),
                                  MKNN=sub[ycol].to_numpy(),
                                  family=sub["family"].to_numpy()))
            res = IC.ancova_family(d)
            if len(res["resid_r2"]):
                # axes match the other Comp 3 plots: x = resid R² | family, y = resid mKNN | family
                ax.scatter(res["resid_r2"], res["resid_mknn"], s=45, color="#377eb8",
                           edgecolor="k", lw=0.3)
                if np.ptp(res["resid_r2"]) > 0:
                    b, a = np.polyfit(res["resid_r2"], res["resid_mknn"], 1)
                    xs = np.linspace(res["resid_r2"].min(), res["resid_r2"].max(), 20)
                    ax.plot(xs, a + b * xs, color="k", lw=1)
                ax.set_title(f"{vis} · {t}\nβ_adj={res['slope_adj']:.2f} "
                             f"p={res['p_mknn']:.2g} η²={res['partial_eta2']:.2f}", fontsize=6.5)
            else:
                ax.set_title(f"{vis} · {t}\nn={res['n']} (need ≥2 families)", fontsize=6.5)
            if ri == nr - 1:
                ax.set_xlabel("resid R² | family", fontsize=8)
            if ci == 0:
                ax.set_ylabel("resid mKNN | family", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(title, y=1.0, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    IC.savefig(fig, path)


def main():
    ap = argparse.ArgumentParser(description="Component 3: feature linear-probing vs alignment.")
    ap.add_argument("--eeg-model", nargs="*", default=None)
    ap.add_argument("--vision-arch", nargs="*", default=list(C.VISION), choices=list(C.VISION))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-per-tier", type=int, default=12)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)

    eeg_variants = _eeg_variants(args.smoke)
    if args.eeg_model and not args.smoke:
        eeg_variants = [(m, s) for (m, s) in eeg_variants if m in args.eeg_model]
    vision_variants = _vision_variants(args.vision_arch, args.smoke)
    mpt = 4 if args.smoke else args.max_per_tier
    folds = 3 if args.smoke else args.folds

    df = build_table(eeg_variants, vision_variants, args.smoke, args.k, folds, mpt)
    if df.empty:
        print("No probe results."); return
    IC.save_npz("probing/feature_probing.npz",
                **{c: df[c].to_numpy() for c in df.columns})

    plot_fit(df, "cross_mknn", "Probe R² vs cross-modal mKNN  (3a)",
             "probing/probe_cross__a_fit.png")
    plot_ancova(df, "cross_mknn", "Probe R² vs cross-modal mKNN — family ANCOVA  (3b)",
                "probing/probe_cross__b_ancova.png")
    if df["intra_mknn"].notna().any():
        plot_fit(df, "intra_mknn", "Probe R² vs intramodal mKNN  (3a-intra)",
                 "probing/probe_intra__a_fit.png")
        plot_ancova(df, "intra_mknn", "Probe R² vs intramodal mKNN — family ANCOVA  (3b-intra)",
                    "probing/probe_intra__b_ancova.png")
    print("Done.")


if __name__ == "__main__":
    main()
