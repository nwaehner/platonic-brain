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
