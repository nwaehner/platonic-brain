"""
DEPRECATED (kept for reference only — do not run as part of the pipeline).

This OLS-on-mKNN study (variance_partition of per-window mKNN onto hand-coded visual /
keyword-lexicon / time blocks) has been REPLACED by the Component 1-5 redesign:
  • interp_features.py        — library-grounded low/high-visual + semantic feature stack
  • neighbour_enrichment.py   — what mediates the neighbourhood (no regression on y)
  • feature_linear_probing.py — probe best-aligned-layer embedding → features, vs mKNN + ANCOVA
  • caption_similarity.py     — neighbour-caption cosine vs background
  • vis_lang_probing.py       — the probing study for vision↔language (4 s grid)
The `neighbour_content` idea here folded into neighbour_enrichment.py. `variance_partition`
and `r2_ols` remain in _interp_common.py for any ad-hoc use.

────────────────────────────────────────────────────────────────────────────────────────
semantic_or_visual_features.py — Study (1b): is the alignment VISUAL or SEMANTIC?

Given the per-window cross-modal alignment from Study 1a, this asks *what* about each video
chunk drives it — low-level perceptual content or high-level meaning — for every vision model
(each at its best alignment layer pair), organised per EEG model.

  VISUAL  (model-free, classical CV on cached frames; NO CNN):
      motion energy · spatial frequency · luminance · contrast · edge density
  SEMANTIC (from the Qwen captions; no inference):
      interpretable keyword attributes (person/face/indoor/outdoor/action/talk/tension)
      + principal components of the already-extracted caption-LLM embedding (when on-grid)

The per-window mKNN is regressed on these blocks plus a temporal nuisance basis (the repo's
central confound: both signals are smooth in time). Outputs per EEG model:
  outputs/<eeg_model>/
    sem_vs_vis_variance_partition.png  commonality decomposition, one group per vision model
    sem_vs_vis_scatter_<arch-size>.png mKNN vs the visual / semantic OLS predictions
    sem_vs_vis_neighbour_content.png   shared-neighbour similarity (visual vs semantic) vs random
    sem_vs_vis_<arch-size>.npz          per-(eeg,vision) arrays
    sem_vs_vis_features.npz             the shared visual/semantic feature matrices

Usage:
  python src/interp/semantic_or_visual_features.py --eeg-model reve
  python src/interp/semantic_or_visual_features.py --eeg-model neurolm --llm open_llama_3b
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _interp_common as IC
import _common as C
import high_alignment as HA

N_LLM_PC = 10


def _clean(M):
    """Replace NaNs (empty boundary windows) with column means; z-score columns."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        M = M[:, None]
    col_mean = np.nanmean(M, axis=0)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(col_mean, inds[1])
    return (M - M.mean(0, keepdims=True)) / (M.std(0, keepdims=True) + 1e-9)


def _time_basis(W, degree=3):
    t = np.linspace(-1, 1, W)
    return np.vstack([t ** d for d in range(1, degree + 1)]).T


def _fit_score(X, y):
    """OLS prediction ŷ = [1 X]·β̂ — the block's best linear guess of the per-window mKNN."""
    Xd = np.hstack([np.ones((len(y), 1)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return Xd @ beta


def _llm_pcs(model, llm_stem, W):
    # The caption-LLM embedding MUST be on this EEG model's own window grid — never fall
    # back to another family (it would temporally misalign the windows). femba/luna share
    # the femba_luna grid, so luna may borrow femba's directory.
    dirs = [model] + (["femba"] if model == "luna" else [])
    emb = None
    for d in dirs:
        emb = IC.try_load_emb(C.llm_path(d, llm_stem))
        if emb is not None:
            break
    if emb is None:
        print(f"    (no caption-LLM '{llm_stem}' on the {model} grid — "
              f"semantic block = caption attributes only)")
        return np.zeros((W, 0))
    if emb.ndim == 3:
        emb = emb[-1]
    elif emb.ndim == 1:
        emb = emb[None]
    X = emb - emb.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return (U[:, :N_LLM_PC] * S[:N_LLM_PC])


def neighbour_content(hap, V, S, seed=0):
    """For each top-mKNN anchor, mean visual- vs semantic-similarity of its shared
    EEG∩vision neighbours, vs random windows."""
    rng = np.random.default_rng(seed)
    nn_e, nn_v = hap["nn_eeg_best"], hap["nn_vis_best"]
    W = V.shape[0]

    def sim(a, idxs, M):
        if len(idxs) == 0:
            return np.nan
        va = M[a] / (np.linalg.norm(M[a]) + 1e-9)
        vb = M[idxs] / (np.linalg.norm(M[idxs], axis=1, keepdims=True) + 1e-9)
        return float((vb @ va).mean())

    vs, ss, vr, sr = [], [], [], []
    for a in hap["top_idx"]:
        shared = np.intersect1d(nn_e[a], nn_v[a])
        if len(shared) == 0:
            continue
        vs.append(sim(a, shared, V)); ss.append(sim(a, shared, S))
        rnd = rng.integers(0, W, size=len(shared) * 4)
        vr.append(sim(a, rnd, V)); sr.append(sim(a, rnd, S))
    return dict(vis_shared=np.nanmean(vs), sem_shared=np.nanmean(ss),
                vis_rand=np.nanmean(vr), sem_rand=np.nanmean(sr))


def analyze_eeg(model, vision_archs, llm_stem, k):
    loaded = HA.load_eeg(model, k)
    if loaded is None:
        return None
    nn_eeg, win_sec, W, fam, _ = loaded

    # shared per-EEG feature blocks (visual depends only on the grid; semantic too)
    vis = IC.compute_visual_features(model)
    V = _clean(vis["scalars"][:W])
    caps = IC.load_captions()
    texts, _ = IC.build_window_texts(caps, win_sec, C.EEG_WINDOWS[model]["n_windows"])
    A, attr_names = IC.caption_attributes(texts[:W])
    A = _clean(A)
    Lpc = _clean(_llm_pcs(model, llm_stem, W))[:W] if N_LLM_PC else np.zeros((W, 0))
    S = np.hstack([A, Lpc]) if Lpc.shape[1] else A
    T = _time_basis(W)

    per_vision = {}
    for arch, size in HA.vision_specs(vision_archs):
        hap = HA.analyze_pair(model, nn_eeg, win_sec, W, arch, size, fam, k)
        if hap is None:
            continue
        y = hap["per_window"]
        vp = IC.variance_partition(y, {"visual": V, "semantic": S, "time": T})
        partials = dict(
            visual_given_time=IC.partial_corr(y, _fit_score(V, y), T),
            semantic_given_time=IC.partial_corr(y, _fit_score(S, y), T),
            visual_given_sem_time=IC.partial_corr(y, _fit_score(V, y), np.hstack([S, T])),
            semantic_given_vis_time=IC.partial_corr(y, _fit_score(S, y), np.hstack([V, T])))
        per_vision[hap["vlabel"]] = dict(
            vlabel=hap["vlabel"], vision_label=hap["vision_label"], y=y, vp=vp,
            partials=partials, nbr=neighbour_content(hap, V, S),
            vis_score=_fit_score(V, y), sem_score=_fit_score(S, y))
        print(f"    {hap['vlabel']:18s} R²_full={vp['R2_full']:.3f}  "
              f"uV={vp['unique_visual']:.3f}  uS={vp['unique_semantic']:.3f}  "
              f"uT={vp['unique_time']:.3f}")
    if not per_vision:
        return None
    return dict(model=model, V=V, S=S, A=A, Lpc=Lpc, vis_names=list(vis["scalar_names"]),
                attr_names=list(attr_names), per_vision=per_vision, W=W)


# ── plots (per EEG model) ────────────────────────────────────────────────────────
def plot_variance_partition(res):
    """Lollipop: per vision model, the per-window-mKNN variance (R²) split into unique
    visual / semantic / temporal + shared; open diamond = R²_full (total variance
    explained — NOT the mKNN itself)."""
    model = res["model"]
    vis = list(res["per_vision"].values())
    comps = [("unique_visual", "unique visual", "#d95f02", "o"),
             ("unique_semantic", "unique semantic", "#7570b3", "o"),
             ("unique_time", "unique time (nuisance)", "#999999", "o"),
             ("shared", "shared", "#1a9850", "o"),
             ("R2_full", "R² total explained", "black", "D")]
    rows_per = len(comps) + 0.8
    fig, ax = plt.subplots(figsize=(13, max(5, 1.0 * rows_per * len(vis))))
    yt, ytl = [], []
    for vi, v in enumerate(vis):
        vp = v["vp"]
        shared = vp["R2_full"] - (vp["unique_visual"] + vp["unique_semantic"]
                                  + vp["unique_time"])
        vals = dict(unique_visual=vp["unique_visual"], unique_semantic=vp["unique_semantic"],
                    unique_time=vp["unique_time"], shared=shared, R2_full=vp["R2_full"])
        base = -vi * rows_per
        for ci, (key, lab, col, mk) in enumerate(comps):
            y = base - ci
            val = vals[key]
            if key != "R2_full":
                ax.hlines(y, 0, val, color=col, lw=2.2, zorder=1)
            ax.plot(val, y, mk, ms=11, markerfacecolor="none" if key == "R2_full" else col,
                    markeredgecolor=col, mew=1.8, zorder=3)
            ax.annotate(f"{val:.3f}", (val, y), fontsize=8, color=col, va="center",
                        ha="left" if val >= 0 else "right",
                        xytext=(6 if val >= 0 else -6, 0), textcoords="offset points")
            yt.append(y); ytl.append(lab)
        ax.text(0.0, base + 0.6, v["vision_label"], fontsize=12, fontweight="bold",
                va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.axvline(0, color="black", lw=0.8, zorder=0)
    ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=8)
    ax.set_xlabel("contribution to R² of per-window mKNN")
    ax.set_title(f"{C.DISPLAY[model]} — what explains the alignment? (per vision model)")
    ax.grid(alpha=0.3, axis="x"); ax.margins(x=0.14)
    fig.tight_layout()
    IC.savefig(fig, f"{model}/sem_vs_vis_variance_partition.png")


def plot_scatter(model, v):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    for ax, (score, name, col) in zip(axes[0],
                                      [(v["vis_score"], "visual", "#d95f02"),
                                       (v["sem_score"], "semantic", "#7570b3")]):
        ax.scatter(score, v["y"], s=7, alpha=0.3, color=col)
        pc = v["partials"][f"{name}_given_time"]
        ax.set_xlabel(f"{name} OLS prediction of mKNN")
        ax.set_ylabel("per-window mKNN")
        ax.set_title(f"{name}:  corr(mKNN, {name} | time) = {pc:.2f}")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{C.DISPLAY[model]} × {v['vision_label']} — "
                 f"per-window mKNN vs block predictions", y=1.0)
    fig.tight_layout()
    IC.savefig(fig, f"{model}/sem_vs_vis_scatter_{v['vlabel']}.png")


def plot_neighbour_content(res):
    model = res["model"]
    vis = list(res["per_vision"].values())
    x = np.arange(len(vis)); w = 0.2
    fig, ax = plt.subplots(figsize=(max(11, 1.4 * len(vis)), 8))
    for off, key, col, lab in [(-1.5 * w, "vis_shared", "#d95f02", "visual · shared"),
                               (-0.5 * w, "vis_rand", "#f4a582", "visual · random"),
                               (0.5 * w, "sem_shared", "#7570b3", "semantic · shared"),
                               (1.5 * w, "sem_rand", "#b2abd2", "semantic · random")]:
        ax.bar(x + off, [v["nbr"][key] for v in vis], w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([v["vision_label"] for v in vis],
                                         rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean similarity to anchor chunk")
    ax.set_title(f"{C.DISPLAY[model]} — are shared neighbours of high-mKNN chunks "
                 f"visually or semantically similar?")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    IC.savefig(fig, f"{model}/sem_vs_vis_neighbour_content.png")


def save(res):
    model = res["model"]
    IC.save_npz(f"{model}/sem_vs_vis_features.npz", V=res["V"], S=res["S"], A=res["A"],
                Lpc=res["Lpc"], vis_names=np.array(res["vis_names"]),
                attr_names=np.array(res["attr_names"]))
    for vlabel, v in res["per_vision"].items():
        vp = {f"vp__{k}": np.array(val) for k, val in v["vp"].items()}
        IC.save_npz(f"{model}/sem_vs_vis_{vlabel}.npz", y=v["y"],
                    partials=np.array(list(v["partials"].items()), dtype=object),
                    nbr=np.array(list(v["nbr"].items()), dtype=object), **vp)


def main():
    ap = argparse.ArgumentParser(description="Study 1b: visual vs semantic drivers.")
    ap.add_argument("--eeg-model", nargs="*", default=list(C.EEG), choices=list(C.EEG))
    ap.add_argument("--vision-arch", nargs="*", default=list(C.VISION),
                    choices=list(C.VISION))
    ap.add_argument("--llm", default="open_llama_3b",
                    help="caption-LLM whose embedding PCs form part of the semantic block.")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    IC.set_token(args.hf_token)
    for model in args.eeg_model:
        print(f"[{model}] visual vs semantic across vision models…")
        res = analyze_eeg(model, args.vision_arch, args.llm, args.k)
        if res is None:
            print(f"  (no data for {model})")
            continue
        plot_variance_partition(res)
        for v in res["per_vision"].values():
            plot_scatter(model, v)
        plot_neighbour_content(res)
        save(res)
    print("Done.")


if __name__ == "__main__":
    main()
