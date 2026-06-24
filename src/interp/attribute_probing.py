"""
attribute_probing.py — Study (4): what visual vs semantic content does each model encode?

Reuses Study 1b's per-window descriptors as PROBE TARGETS and asks how well a linear probe
recovers each one from every model's representation. This says what is *linearly decodable*
from a space — complementary to 1b (which asks what drives alignment):

  VISUAL targets  (model-free classical CV): motion · spatial-freq · luminance · contrast · edges
  SEMANTIC targets (Qwen captions):          person · face · indoor · outdoor · action · talk · tension

Probe = ridge regression with CONTIGUOUS-BLOCK cross-validation (no temporally adjacent windows
across folds → no leakage from temporal autocorrelation), scored by test R² and compared to a
label-shuffled baseline. All models are placed on one shared grid via `--eeg-ref` (its window
scheme defines the labels), exactly like model_stitching. No model inference.

Figures (→ outputs/):
  probe_heatmap            — best-layer R² for every attribute × every model
  probe_vs_depth           — decoding R² vs relative depth (visual vs semantic, per modality)
  probe_visual_vs_semantic — per-model mean visual- vs semantic-decodability scatter
  probe_vs_scale           — decodability vs #params

Usage:
  python src/interp/attribute_probing.py --eeg-ref neurolm
  python src/interp/attribute_probing.py --eeg-ref neurolm --models neurolm vjepa2 open_llama_13b
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

import _interp_common as IC
import _common as C

MOD_COLOR = {"eeg": "#1a9850", "vision": "#d95f02", "llm": "#7570b3"}


def _r2(y, yh):
    ss = ((y - yh) ** 2).sum()
    st = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss / (st + 1e-12))


def block_cv_r2(X, y, n_folds=5):
    """Ridge R² under contiguous-block CV (temporal-leakage-safe)."""
    W = len(y)
    bounds = np.linspace(0, W, n_folds + 1).astype(int)
    scores = []
    for f in range(n_folds):
        te = np.arange(bounds[f], bounds[f + 1])
        tr = np.setdiff1d(np.arange(W), te)
        if len(te) < 2 or len(tr) < 2:
            continue
        rg = Ridge(alpha=10.0).fit(X[tr], y[tr])
        scores.append(_r2(y[te], rg.predict(X[te])))
    return float(np.mean(scores)) if scores else float("nan")


def _zscore(M):
    return (M - M.mean(0, keepdims=True)) / (M.std(0, keepdims=True) + 1e-9)


def _clean_labels(M):
    M = np.asarray(M, dtype=np.float64)
    col_mean = np.nanmean(M, axis=0)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(col_mean, inds[1])
    return _zscore(M)


# ── label construction on the reference grid ─────────────────────────────────────
# Probe TARGETS = the Component-1 feature stack (same tiers as neighbour_enrichment):
# low-level visual / high-level visual (CLIP) / semantic. Requires the cached feature set
# for the reference grid (build with `interp_features.py --grid <eeg_ref>`).
TIERS = ["low", "highvis", "sem"]
TIER_LABEL = {"low": "low-level visual", "highvis": "high-level visual", "sem": "semantic"}


def build_labels(eeg_ref, smoke=False):
    import interp_features as F
    feats = F.load_features(eeg_ref, smoke)
    if feats is None:
        raise SystemExit(
            f"No Component-1 feature cache for grid '{eeg_ref}'. Build it first:\n"
            f"  python src/interp/interp_features.py --grid {eeg_ref}"
            f"{' --smoke' if smoke else ''}")
    Y = _clean_labels(feats["feat"])                   # z-scored (W, F)
    names = [str(n) for n in feats["feat_names"]]
    kinds = [str(t) for t in feats["feat_tier"]]       # low / highvis / sem
    return Y, names, kinds


# ── model specs on the shared grid (mirrors model_stitching) ─────────────────────
def build_specs(eeg_ref, restrict):
    vis_family = C.EEG[eeg_ref]["family"]
    specs = []
    for m, cfg in C.EEG.items():
        if cfg["family"] == vis_family:
            specs.append(dict(kind="eeg", name=m, size=cfg["sizes"][-1], mod="eeg",
                              label=f"{C.DISPLAY[m]}-{cfg['sizes'][-1]}",
                              params=C.PARAMS[m][cfg["sizes"][-1]]))
    for arch, cfg in C.VISION.items():
        s = cfg["sizes"][-1]
        specs.append(dict(kind="vision", name=arch, size=s, mod="vision",
                          vis_family=vis_family, label=C.vision_label(arch, s),
                          params=C.VISION_PARAMS.get((arch, s), np.nan)))
    for fam, stems in C.LLM.items():
        st = stems[-1]
        specs.append(dict(kind="llm", name=fam, size=st, mod="llm", llm_dir=eeg_ref,
                          label=C.LLM_LABEL.get(st, st), params=C.LLM_PARAMS.get(st, np.nan)))
    if restrict:
        specs = [s for s in specs if s["name"] in restrict]
    return specs


def load_spec(sp):
    if sp["kind"] == "eeg":
        e = IC.try_load_emb(C.EEG[sp["name"]]["fname"](sp["size"]))
        return None if e is None else e.mean(axis=2)
    if sp["kind"] == "vision":
        e = IC.try_load_emb(C.VISION[sp["name"]]["path"](sp["size"], sp["vis_family"]))
    else:
        e = IC.load_llm_grid(sp["llm_dir"], sp["size"])    # triniborrell (nitrox639 lacks most)
    if e is None:
        return None
    return e[None] if e.ndim == 2 else e


# ── probing ──────────────────────────────────────────────────────────────────────
def probe_model(emb, Y, n_layers_probe, n_folds, W):
    """Returns scores (n_layers_used, A), rel (n_layers_used,), baseline (A,)."""
    L = emb.shape[0]
    layers = np.unique(np.linspace(0, L - 1, min(n_layers_probe, L)).astype(int))
    A = Y.shape[1]
    scores = np.full((len(layers), A), np.nan)
    for li, l in enumerate(layers):
        X = _zscore(C.l2(emb[l]))[:W]
        for a in range(A):
            scores[li, a] = block_cv_r2(X, Y[:, a], n_folds)
    # shuffled baseline at the last probed layer
    rng = np.random.default_rng(0)
    Xb = _zscore(C.l2(emb[layers[-1]]))[:W]
    baseline = np.array([block_cv_r2(Xb, rng.permutation(Y[:, a]), n_folds)
                         for a in range(A)])
    rel = layers / max(L - 1, 1)
    return scores, rel, baseline


def run(args):
    IC.set_token(args.hf_token)
    Y, names, kinds = build_labels(args.eeg_ref, args.smoke)
    W = Y.shape[0]
    kinds = np.array(kinds)
    print(f"labels: {W} windows × {len(names)} attributes ("
          + ", ".join(f"{(kinds == t).sum()} {t}" for t in TIERS) + ")")

    specs = build_specs(args.eeg_ref, args.models)
    results = {}
    for sp in specs:
        emb = load_spec(sp)
        if emb is None or emb.shape[1] < W:
            if emb is not None:
                print(f"  skip {sp['label']}: W={emb.shape[1]} < {W}")
            continue
        emb = emb[:, :W]                                # match the (possibly smoke) label grid
        print(f"  probing {sp['label']}: {emb.shape}")
        scores, rel, base = probe_model(emb, Y, args.n_layers, args.folds, W)
        results[sp["label"]] = dict(scores=scores, rel=rel, baseline=base,
                                    mod=sp["mod"], params=sp["params"])

    if not results:
        print("No models probed.")
        return
    plot_heatmap(results, names, kinds)
    plot_vs_depth(results, kinds)
    plot_visual_vs_semantic(results, kinds)
    plot_vs_scale(results, kinds)
    save(results, names, kinds, args.eeg_ref)
    print("Done.")


def _best_layer(scores):
    return np.nanmax(scores, axis=0)               # (A,) best-over-layers R²


# ── plots ────────────────────────────────────────────────────────────────────────
def plot_heatmap(results, names, kinds):
    keys = list(results)
    M = np.column_stack([_best_layer(results[k]["scores"]) for k in keys])  # (A, models)
    fig, ax = plt.subplots(figsize=(max(12, 0.7 * len(keys)), max(8, 0.4 * len(names))))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=max(0.05, np.nanmax(M)), aspect="auto")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03).set_label("best-layer R²")
    ax.set_title("Linear decodability of each attribute from each model")
    fig.tight_layout()
    IC.savefig(fig, "attribute_probing/probe_heatmap.png")


def plot_vs_depth(results, kinds):
    tiers = [t for t in TIERS if (kinds == t).any()]
    fig, axes = plt.subplots(1, len(tiers), figsize=(7 * len(tiers), 6.5), squeeze=False)
    grid = np.linspace(0, 1, 11)
    for ax, kind in zip(axes[0], tiers):
        cols = np.where(kinds == kind)[0]
        for mod in ["eeg", "vision", "llm"]:
            curves = []
            for r in results.values():
                if r["mod"] != mod:
                    continue
                per_layer = np.nanmean(r["scores"][:, cols], axis=1)
                fin = np.isfinite(per_layer)
                if fin.sum() < 2:
                    continue
                curves.append(np.interp(grid, r["rel"][fin], per_layer[fin]))
            if curves:
                ax.plot(grid, np.mean(curves, 0), color=MOD_COLOR[mod], lw=2, label=mod)
        ax.set_xlabel("relative depth"); ax.set_ylabel(f"mean {TIER_LABEL[kind]} R²")
        ax.set_title(f"{TIER_LABEL[kind]} decodability vs depth")
        ax.legend(title="modality"); ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "attribute_probing/probe_vs_depth.png")


def plot_visual_vs_semantic(results, kinds):
    # "visual" = low-level ∪ high-level (CLIP); "semantic" = the sem tier.
    vis_c = np.where((kinds == "low") | (kinds == "highvis"))[0]
    sem_c = np.where(kinds == "sem")[0]
    fig, ax = plt.subplots(figsize=(11, 11))
    for k, r in results.items():
        best = _best_layer(r["scores"])
        x, y = np.nanmean(best[vis_c]), np.nanmean(best[sem_c])
        ax.scatter(x, y, color=MOD_COLOR[r["mod"]], s=70, edgecolor="black", lw=0.4)
        ax.annotate(k, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    lim = [0, max(0.05, ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color="grey", ls="--", lw=1)
    ax.set_xlabel("mean visual decodability — low+high (R²)")
    ax.set_ylabel("mean semantic decodability (R²)")
    ax.set_title("What does each model encode — visual (low+high) vs semantic?")
    handles = [plt.Line2D([], [], marker="o", ls="", color=MOD_COLOR[m], label=m)
               for m in MOD_COLOR]
    ax.legend(handles=handles, title="modality"); ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "attribute_probing/probe_visual_vs_semantic.png")


def plot_vs_scale(results, kinds):
    tiers = [t for t in TIERS if (kinds == t).any()]
    fig, axes = plt.subplots(1, len(tiers), figsize=(7 * len(tiers), 6.5), squeeze=False)
    for ax, kind in zip(axes[0], tiers):
        cols = np.where(kinds == kind)[0]
        for k, r in results.items():
            p = r["params"]
            if not np.isfinite(p):
                continue
            y = np.nanmean(_best_layer(r["scores"])[cols])
            ax.scatter(np.log10(p), y, color=MOD_COLOR[r["mod"]], s=60,
                       edgecolor="black", lw=0.4)
            ax.annotate(k, (np.log10(p), y), fontsize=5, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel("log10 #params (M)"); ax.set_ylabel(f"mean {TIER_LABEL[kind]} R²")
        ax.set_title(f"{TIER_LABEL[kind]} decodability vs scale"); ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "attribute_probing/probe_vs_scale.png")


def save(results, names, kinds, eeg_ref):
    keys = list(results)
    flat = dict(model_keys=np.array(keys), attr_names=np.array(names),
                attr_kinds=np.array(kinds),
                modality=np.array([results[k]["mod"] for k in keys]),
                params=np.array([results[k]["params"] for k in keys], dtype=float))
    for k in keys:
        flat[f"{k}__scores"] = results[k]["scores"]
        flat[f"{k}__rel"] = results[k]["rel"]
        flat[f"{k}__baseline"] = results[k]["baseline"]
    IC.save_npz(f"attribute_probing/attribute_probing__{eeg_ref}.npz", **flat)


def main():
    ap = argparse.ArgumentParser(description="Study 4: attribute probing.")
    ap.add_argument("--eeg-ref", default="neurolm", choices=list(C.EEG),
                    help="reference EEG model whose window scheme defines the labels/grid.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these family/arch names.")
    ap.add_argument("--n-layers", type=int, default=8,
                    help="number of (evenly spaced) layers probed per model.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="use the Component-1 __smoke feature cache (few windows).")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
