"""
eeg_feature_alignment.py — does cross-architectural alignment among EEG foundation models
predict how well each model's embedding decodes EEG-DERIVED (NICE) features?

Mirrors the Platonic-Universe "cross-architectural alignment vs performance" analysis, but
within the EEG modality and with EEG-derived targets (from eeg_features.py / combine):

  PERFORMANCE  per (model, feature) = best-decoding-layer block-CV ridge R² of that EEG
               feature from the model's embedding (native grid). Overall performance =
               mean over features.
  ALIGNMENT    per model = mean cross-architectural MKNN and (linear) CKA to every OTHER
               EEG model. Models live on different window grids, so each is downsampled into
               common 10 s bins (average of the windows whose centre falls in the bin) before
               the geometry is compared; best layer-pair over MKNN / CKA.

Outputs (outputs/eeg_alignment/):
  decodability.csv/png      per-model × per-feature R² (are EEG features decodable at last?)
  crossarch_mknn.png        per-model mean MKNN vs performance  (+ Spearman ρ,p, fit, NO error bars)
  crossarch_cka.png         per-model mean CKA  vs performance
  perfeature_rho.png        per-feature Spearman ρ of (decodability vs MKNN / CKA) across models
  summary.npz               all arrays

Usage:  python src/interp/eeg_feature_alignment.py
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
from scipy.stats import spearmanr

import _interp_common as IC
import _common as C
from feature_gridsearch import ridge_path_r2

OUT = IC.OUT_DIR / "eeg_alignment"
EEGFEAT = IC.OUT_DIR / "eeg_features"
COMMON_WIN = 10
N_COMMON = 1080
FAM_COLORS = {"femba": "#1b9e77", "luna": "#d95f02", "neurolm": "#7570b3",
              "reve": "#e7298a", "steegformer": "#66a61e"}
FAM_MARKERS = {"femba": "o", "luna": "s", "neurolm": "^", "reve": "D", "steegformer": "P"}


_SMOKE_EEG = [("reve", "base"), ("neurolm", "b"), ("femba", "base")]


def eeg_variants(smoke=False):
    if smoke:
        return _SMOKE_EEG
    return [(m, s) for m in C.EEG for s in C.EEG[m]["sizes"]]


def load_eeg_feat(model):
    local = EEGFEAT / f"{model}__eegfeat.npz"
    if local.exists():
        z = np.load(local, allow_pickle=True)
    else:
        try:
            from huggingface_hub import hf_hub_download
            dl = hf_hub_download(IC.REPO_4S, f"eeg_features/{model}__eegfeat.npz",
                                 repo_type="dataset", token=C.HF_TOKEN_CACHE)
            z = np.load(dl, allow_pickle=True)
            print(f"  [{model}] loaded features from HuggingFace")
        except Exception as e:
            print(f"  [skip] no local or HF features for grid {model}: {e}")
            return None, None
    return z["feat"].astype(np.float64), list(z["feat_names"])


def _clean(feat):
    F = np.asarray(feat, np.float64).copy()
    cm = np.nanmean(F, axis=0)
    cm = np.where(np.isfinite(cm), cm, 0.0)
    bad = np.where(~np.isfinite(F))
    F[bad] = np.take(cm, bad[1])
    return F


# ── common-grid downsampling (different window schemes → shared 10 s bins) ─────────
def to_common(emb, win_sec, common_win=COMMON_WIN, n_common=N_COMMON):
    L, W, D = emb.shape
    centers = (np.arange(W) + 0.5) * win_sec
    b = np.clip((centers // common_win).astype(int), 0, n_common - 1)
    out = np.zeros((L, n_common, D), dtype=np.float64)
    cnt = np.zeros(n_common)
    for w in range(W):
        out[:, b[w]] += emb[:, w]
        cnt[b[w]] += 1
    out /= np.maximum(cnt, 1)[None, :, None]
    return out


def norm_gram(X):
    """Feature-centred, Frobenius-normalised Gram matrix XcXcᵀ for linear CKA."""
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T
    n = np.linalg.norm(G)
    return G / (n + 1e-20)


def best_cka(grams_a, grams_b):
    return max(float(np.sum(Ga * Gb)) for Ga in grams_a for Gb in grams_b)


# ── per-model decodability (native grid, best-decoding layer) ─────────────────────
def model_decodability(model, size, folds=5):
    emb = IC.load_eeg_emb(model, size)
    feat, names = load_eeg_feat(model)
    if emb is None or feat is None:
        return None, names
    W = min(emb.shape[1], feat.shape[0])
    Y = _clean(feat[:W])
    best = np.full(Y.shape[1], -np.inf)
    for l in range(emb.shape[0]):
        X = C.l2(emb[l])[:W]
        r = ridge_path_r2(X, Y, "block", folds)
        best = np.where(np.isfinite(r), np.maximum(best, r), best)
    best[~np.isfinite(best)] = np.nan
    return best, names


def main():
    ap = argparse.ArgumentParser(description="EEG cross-arch alignment vs NICE-feature decodability.")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="quick smoke test: 3 EEG models (reve-base, neurolm-b, femba-base)")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        print("=== SMOKE MODE: 3 EEG models ===")

    variants = eeg_variants(args.smoke)
    # ── load embeddings (native + common grid), decodability, grams/knn ───────────
    models, names = [], None
    common, knn, grams, perf = {}, {}, {}, {}
    for (m, s) in variants:
        emb = IC.load_eeg_emb(m, s)
        if emb is None:
            print(f"  [skip] {m}-{s}: no embedding"); continue
        r2, nm = model_decodability(m, s, args.folds)
        if r2 is None:
            print(f"  [skip] {m}-{s}: no features for grid {m}"); continue
        names = names or nm
        key = f"{m}-{s}"
        models.append((key, m, s))
        perf[key] = r2
        win = C.EEG_WINDOWS[m]["win_sec"]
        ce = to_common(emb, win)
        common[key] = ce
        knn[key] = C.precompute_knn_layers(ce, args.k)
        grams[key] = [norm_gram(ce[l]) for l in range(ce.shape[0])]
        print(f"  {key}: decod meanR2={np.nanmean(r2):+.3f}  (best feat {names[int(np.nanargmax(r2))]}="
              f"{np.nanmax(r2):+.3f})", flush=True)

    keys = [k for k, _, _ in models]
    n = len(keys)
    if n < 3:
        print("too few models with data; aborting"); return

    # ── pairwise cross-arch MKNN + CKA, per-model mean ────────────────────────────
    MK = np.full((n, n), np.nan); CK = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            mk = IC.best_layer_pair(knn[keys[i]], knn[keys[j]])[2]
            ck = best_cka(grams[keys[i]], grams[keys[j]])
            MK[i, j] = MK[j, i] = mk
            CK[i, j] = CK[j, i] = ck
    mean_mknn = np.nanmean(MK, axis=1)
    mean_cka = np.nanmean(CK, axis=1)
    perf_mat = np.vstack([perf[k] for k in keys])          # (n, F)
    perf_mean = np.nanmean(perf_mat, axis=1)               # overall performance per model

    # ── correlations ──────────────────────────────────────────────────────────────
    rho_mk, p_mk = spearmanr(mean_mknn, perf_mean)
    rho_ck, p_ck = spearmanr(mean_cka, perf_mean)
    print(f"\n=== CROSS-ARCH (n={n} EEG models) ===")
    print(f"  mean MKNN vs performance:  rho={rho_mk:+.3f}  p={p_mk:.3g}")
    print(f"  mean CKA  vs performance:  rho={rho_ck:+.3f}  p={p_ck:.3g}")

    fams = [m for _, m, _ in models]
    IC.save_npz("eeg_alignment/summary.npz", keys=np.array(keys), family=np.array(fams),
                feat_names=np.array(names), perf=perf_mat, perf_mean=perf_mean,
                mean_mknn=mean_mknn, mean_cka=mean_cka, MK=MK, CK=CK,
                rho_mknn=rho_mk, p_mknn=p_mk, rho_cka=rho_ck, p_cka=p_ck)

    # decodability table
    df = pd.DataFrame(perf_mat, columns=names, index=keys)
    df.to_csv(OUT / "decodability.csv")
    print("\n=== DECODABILITY (mean over models, per feature) ===")
    order = np.argsort(np.nanmean(perf_mat, axis=0))[::-1]
    for fi in order:
        col = perf_mat[:, fi]
        print(f"  {names[fi]:<18} meanR2={np.nanmean(col):+.3f}  best={np.nanmax(col):+.3f}  "
              f"pos {int(np.nansum(col > 0))}/{n}")

    # ── per-feature Spearman across models ────────────────────────────────────────
    rho_f_mk, rho_f_ck, p_f_mk = [], [], []
    for fi in range(len(names)):
        col = perf_mat[:, fi]
        rmk, pmk = spearmanr(mean_mknn, col, nan_policy="omit")
        rck, _ = spearmanr(mean_cka, col, nan_policy="omit")
        rho_f_mk.append(rmk); rho_f_ck.append(rck); p_f_mk.append(pmk)

    _plot_crossarch(perf_mean, mean_mknn, fams, "mean cross-arch MKNN",
                    rho_mk, p_mk, "eeg_alignment/crossarch_mknn.png")
    _plot_crossarch(perf_mean, mean_cka, fams, "mean cross-arch CKA",
                    rho_ck, p_ck, "eeg_alignment/crossarch_cka.png")
    _plot_perfeature(names, rho_f_mk, rho_f_ck, p_f_mk)
    _plot_decodability(df)
    import plot_eeg_perfeature as PF                      # per-feature scatter grids
    PF._grid(perf_mat, mean_mknn, fams, names, "mean cross-arch MKNN",
             OUT / "crossarch_perfeature_mknn.png")
    PF._grid(perf_mat, mean_cka, fams, names, "mean cross-arch CKA",
             OUT / "crossarch_perfeature_cka.png")
    print("\nDone.")


# ── plots (paper style, no error bars) ────────────────────────────────────────────
def _plot_crossarch(perf, align, fams, ylab, rho, p, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for x, y, fam in zip(perf, align, fams):
        ax.scatter(x, y, s=70, color=FAM_COLORS.get(fam, "#333"),
                   marker=FAM_MARKERS.get(fam, "o"), edgecolor="k", lw=0.4)
    m = np.isfinite(perf) & np.isfinite(align)
    if m.sum() >= 2 and np.ptp(perf[m]) > 0:
        b, a = np.polyfit(perf[m], align[m], 1)
        xs = np.linspace(perf[m].min(), perf[m].max(), 20)
        ax.plot(xs, a + b * xs, "k--", lw=1)
    ax.set_xlabel("performance  (mean EEG-feature R²)", fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(f"Cross-architectural alignment vs performance\nSpearman ρ={rho:+.3f}  p={p:.3g}",
                 fontsize=10)
    handles = [plt.Line2D([], [], marker=FAM_MARKERS[f], ls="", color=FAM_COLORS[f],
               label=f, markeredgecolor="k") for f in FAM_COLORS if f in fams]
    ax.legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    IC.savefig(fig, path)


def _plot_perfeature(names, rho_mk, rho_ck, p_mk):
    idx = np.argsort(rho_mk)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh(y - 0.2, np.array(rho_mk)[idx], height=0.4, color="#2c7fb8", label="MKNN")
    ax.barh(y + 0.2, np.array(rho_ck)[idx], height=0.4, color="#d95f02", label="CKA")
    for yi, fi in enumerate(idx):
        if p_mk[fi] < 0.05:
            ax.text(rho_mk[fi], yi - 0.2, "*", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(np.array(names)[idx], fontsize=7)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Spearman ρ across models (decodability vs alignment)", fontsize=9)
    ax.set_title("Per-feature: does cross-arch alignment track decodability?\n(* p<0.05 for MKNN)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    IC.savefig(fig, "eeg_alignment/perfeature_rho.png")


def _plot_decodability(df):
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * df.shape[1]), 0.5 * df.shape[0] + 1))
    im = ax.imshow(df.to_numpy(), aspect="auto", cmap="RdBu_r",
                   vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025).set_label("block-CV R²")
    ax.set_title("EEG-feature decodability  (model × feature, best-decoding layer)", fontsize=10)
    fig.tight_layout()
    IC.savefig(fig, "eeg_alignment/decodability.png")


if __name__ == "__main__":
    main()
