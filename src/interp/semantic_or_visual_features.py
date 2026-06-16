"""
semantic_or_visual_features.py — Study (1b): is the alignment VISUAL or SEMANTIC?

Given the per-window cross-modal alignment from Study 1a, this asks *what* about each
video chunk drives it — low-level perceptual content or high-level meaning — along a
gradient from purely visual to purely semantic:

  VISUAL  (model-free, classical CV on cached frames; NO CNN):
      motion energy · spatial frequency · luminance · contrast · edge density
  SEMANTIC (from the Qwen captions; no inference):
      interpretable keyword attributes (person/face/indoor/outdoor/action/talk/tension)
      + principal components of the already-extracted caption-LLM embedding

We regress the per-window mKNN on these blocks (and on a temporal nuisance basis, because
the repo's central confound is that both signals are smooth in time) and report:

  1 variance partition / commonality  — unique(visual), unique(semantic), shared, time
  2 mKNN vs visual-score and vs semantic-score scatters
  3 visual→semantic gradient           — R² carried by each block in order
  4 neighbour-content test             — are the shared EEG∩vision neighbours of high-mKNN
                                         anchors more visually- or semantically-similar than
                                         temporally-matched random windows?

Usage:
  python src/interp/semantic_or_visual_features.py --eeg-model reve
  python src/interp/semantic_or_visual_features.py                      # all EEG models
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
    M = (M - M.mean(0, keepdims=True)) / (M.std(0, keepdims=True) + 1e-9)
    return M


def _time_basis(W, degree=3):
    t = np.linspace(-1, 1, W)
    return np.vstack([t ** d for d in range(1, degree + 1)]).T


def _llm_pcs(model, llm_stem, W):
    # The caption-LLM embedding MUST be on this EEG model's own window grid — never
    # fall back to a different family (that would temporally misalign the windows).
    # femba/luna share the femba_luna grid, so luna may borrow femba's directory.
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
        emb = emb[-1]                               # last layer (W, D)
    elif emb.ndim == 1:
        emb = emb[None]
    X = emb - emb.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :N_LLM_PC] * S[:N_LLM_PC])         # (W, N_LLM_PC) score matrix


def analyze_eeg(model, vision_archs, llm_stem, k):
    ha = HA.analyze_eeg(model, vision_archs, k, verbose=False)
    if ha is None:
        return None
    y = ha["per_window"]
    W = len(y)
    win_sec = ha["win_sec"]

    # ── visual block (5 interpretable scalars) ──
    vis = IC.compute_visual_features(model)
    V = _clean(vis["scalars"][:W])
    vis_names = vis["scalar_names"]

    # ── semantic block: caption attributes + caption-LLM PCs ──
    caps = IC.load_captions()
    texts, _ = IC.build_window_texts(caps, win_sec, C.EEG_WINDOWS[model]["n_windows"])
    texts = texts[:W]
    A, attr_names = IC.caption_attributes(texts)
    A = _clean(A)
    Lpc = _clean(_llm_pcs(model, llm_stem, W)) if N_LLM_PC else np.zeros((W, 0))
    Lpc = Lpc[:W]
    S = np.hstack([A, Lpc]) if Lpc.shape[1] else A

    # ── temporal nuisance ──
    T = _time_basis(W)

    # 1) variance partition
    vp = IC.variance_partition(y, {"visual": V, "semantic": S, "time": T})

    # 2) single visual/semantic scores (OLS fit) for scatters + partials
    def fit_score(X):
        Xd = np.hstack([np.ones((W, 1)), X])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        return Xd @ beta
    vis_score = fit_score(V)
    sem_score = fit_score(S)
    partials = dict(
        visual_given_sem_time=IC.partial_corr(y, vis_score, np.hstack([S, T])),
        semantic_given_vis_time=IC.partial_corr(y, sem_score, np.hstack([V, T])),
        visual_given_time=IC.partial_corr(y, vis_score, T),
        semantic_given_time=IC.partial_corr(y, sem_score, T))

    # 3) visual→semantic gradient: own-R² of each ordered block
    gradient = [("low-level visual", IC.r2_ols(y, V)),
                ("caption attributes", IC.r2_ols(y, A)),
                ("caption embedding", IC.r2_ols(y, Lpc) if Lpc.shape[1] else np.nan)]

    # 4) neighbour-content test on top anchors
    nbr = neighbour_content(ha, V, S)

    return dict(model=model, y=y, V=V, S=S, A=A, Lpc=Lpc, T=T,
                vis_names=list(vis_names), attr_names=list(attr_names),
                vp=vp, partials=partials, gradient=gradient, nbr=nbr,
                vis_score=vis_score, sem_score=sem_score,
                vision_label=ha["vision_label"], top_idx=ha["top_idx"], W=W)


def neighbour_content(ha, V, S, n_random=200, seed=0):
    """For each top-mKNN anchor, mean visual- vs semantic-similarity of its shared
    EEG∩vision neighbours, compared to temporally-matched random windows."""
    rng = np.random.default_rng(seed)
    nn_e, nn_v = ha["nn_eeg_best"], ha["nn_vis_best"]
    W = V.shape[0]

    def sim(a, idxs, M):                            # mean cosine sim of M[a] to M[idxs]
        if len(idxs) == 0:
            return np.nan
        va = M[a] / (np.linalg.norm(M[a]) + 1e-9)
        vb = M[idxs] / (np.linalg.norm(M[idxs], axis=1, keepdims=True) + 1e-9)
        return float((vb @ va).mean())

    vis_shared, sem_shared, vis_rand, sem_rand = [], [], [], []
    for a in ha["top_idx"]:
        shared = np.intersect1d(nn_e[a], nn_v[a])
        if len(shared) == 0:
            continue
        vis_shared.append(sim(a, shared, V)); sem_shared.append(sim(a, shared, S))
        rnd = rng.integers(0, W, size=len(shared) * 4)
        vis_rand.append(sim(a, rnd, V)); sem_rand.append(sim(a, rnd, S))
    return dict(vis_shared=np.nanmean(vis_shared), sem_shared=np.nanmean(sem_shared),
                vis_rand=np.nanmean(vis_rand), sem_rand=np.nanmean(sem_rand))


# ── plots ────────────────────────────────────────────────────────────────────────
def plot_variance_partition(results):
    """Lollipop / dot plot. For each EEG model the per-window-mKNN variance (R²) is
    split into the unique visual / semantic / temporal parts and the shared part, each
    shown as a dot on a common R²-contribution axis (negative = suppression). The open
    black diamond marks R²_full = the total variance explained."""
    models = list(results)
    # component (key, label, colour, marker), top → bottom within each model group
    comps = [("unique_visual", "unique visual", "#d95f02", "o"),
             ("unique_semantic", "unique semantic", "#7570b3", "o"),
             ("unique_time", "unique time (nuisance)", "#999999", "o"),
             ("shared", "shared", "#1a9850", "o"),
             ("R2_full", "R² total explained", "black", "D")]
    rows_per = len(comps) + 0.8                       # +gap between model groups
    fig, ax = plt.subplots(figsize=(13, max(5, 1.0 * rows_per * len(models))))
    yt, ytl = [], []
    for mi, m in enumerate(models):
        vp = results[m]["vp"]
        shared = vp["R2_full"] - (vp["unique_visual"] + vp["unique_semantic"]
                                  + vp["unique_time"])
        vals = {"unique_visual": vp["unique_visual"],
                "unique_semantic": vp["unique_semantic"],
                "unique_time": vp["unique_time"], "shared": shared,
                "R2_full": vp["R2_full"]}
        base_y = -mi * rows_per
        for ci, (key, lab, col, mk) in enumerate(comps):
            y = base_y - ci
            v = vals[key]
            if key != "R2_full":
                ax.hlines(y, 0, v, color=col, lw=2.2, zorder=1)
            ax.plot(v, y, mk, color=col, ms=11,
                    markerfacecolor="none" if key == "R2_full" else col,
                    markeredgecolor=col, mew=1.8, zorder=3)
            ax.annotate(f"{v:.3f}", (v, y), fontsize=8, color=col,
                        xytext=(6 if v >= 0 else -6, 0), textcoords="offset points",
                        va="center", ha="left" if v >= 0 else "right")
            yt.append(y); ytl.append(lab)
        ax.text(0.0, base_y + 0.6, C.DISPLAY[m], fontsize=13, fontweight="bold",
                va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.axvline(0, color="black", lw=0.8, zorder=0)
    ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=9)
    ax.set_xlabel("contribution to R² of per-window mKNN")
    ax.set_title("What explains the alignment? Variance partition per EEG model")
    ax.grid(alpha=0.3, axis="x")
    ax.margins(x=0.12)
    fig.tight_layout()
    IC.savefig(fig, "sem_vs_vis__variance_partition.png")


def plot_scatters(results):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 5 * n), squeeze=False)
    for row, (m, r) in enumerate(results.items()):
        for col, (score, name, color) in enumerate(
                [(r["vis_score"], "visual", "#d95f02"),
                 (r["sem_score"], "semantic", "#7570b3")]):
            ax = axes[row][col]
            ax.scatter(score, r["y"], s=6, alpha=0.3, color=color)
            ax.set_xlabel(f"{name} prediction"); ax.set_ylabel("per-window mKNN")
            pc = r["partials"][f"{name}_given_time"]
            ax.set_title(f"{C.DISPLAY[m]} — {name}  (partial r|time={pc:.2f})")
            ax.grid(alpha=0.3)
    fig.suptitle("Per-window mKNN vs visual / semantic predictions", y=1.0)
    fig.tight_layout()
    IC.savefig(fig, "sem_vs_vis__scatters.png")


def plot_gradient(results):
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [g[0] for g in next(iter(results.values()))["gradient"]]
    x = np.arange(len(labels))
    width = 0.8 / max(len(results), 1)
    for i, (m, r) in enumerate(results.items()):
        vals = [g[1] for g in r["gradient"]]
        ax.bar(x + i * width, vals, width, label=C.DISPLAY[m])
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("own R² of per-window mKNN")
    ax.set_title("Visual → semantic gradient: variance carried by each descriptor block")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    IC.savefig(fig, "sem_vs_vis__gradient.png")


def plot_neighbour_content(results):
    models = list(results)
    fig, ax = plt.subplots(figsize=(max(10, 2.2 * len(models)), 8))
    x = np.arange(len(models)); w = 0.2
    for off, key, color, lab in [(-1.5 * w, "vis_shared", "#d95f02", "visual · shared"),
                                 (-0.5 * w, "vis_rand", "#f4a582", "visual · random"),
                                 (0.5 * w, "sem_shared", "#7570b3", "semantic · shared"),
                                 (1.5 * w, "sem_rand", "#b2abd2", "semantic · random")]:
        ax.bar(x + off, [results[m]["nbr"][key] for m in models], w, color=color, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([C.DISPLAY[m] for m in models])
    ax.set_ylabel("mean similarity to anchor")
    ax.set_title("Are shared neighbours of high-mKNN chunks visually or semantically similar?")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    IC.savefig(fig, "sem_vs_vis__neighbour_content.png")


def save(results):
    for m, r in results.items():
        vp = {f"vp__{k}": np.array(v) for k, v in r["vp"].items()}
        IC.save_npz(f"sem_vs_vis__{m}.npz",
                    y=r["y"], V=r["V"], S=r["S"], A=r["A"], Lpc=r["Lpc"],
                    vis_names=np.array(r["vis_names"]),
                    attr_names=np.array(r["attr_names"]),
                    gradient_labels=np.array([g[0] for g in r["gradient"]]),
                    gradient_r2=np.array([g[1] for g in r["gradient"]]),
                    partials=np.array(list(r["partials"].items()), dtype=object),
                    nbr=np.array(list(r["nbr"].items()), dtype=object), **vp)


def main():
    ap = argparse.ArgumentParser(description="Study 1b: visual vs semantic drivers.")
    ap.add_argument("--eeg-model", nargs="*", default=list(C.EEG), choices=list(C.EEG))
    ap.add_argument("--vision-arch", nargs="*", default=list(C.VISION),
                    choices=list(C.VISION))
    ap.add_argument("--llm", default="open_llama_3b",
                    help="caption-LLM whose embedding PCs form the semantic block.")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    IC.set_token(args.hf_token)
    results = {}
    for m in args.eeg_model:
        print(f"[{m}] visual vs semantic…")
        r = analyze_eeg(m, args.vision_arch, args.llm, args.k)
        if r is None:
            print(f"  (no data for {m})")
            continue
        results[m] = r
        vp = r["vp"]
        print(f"  R²_full={vp['R2_full']:.3f}  unique_visual={vp['unique_visual']:.3f}  "
              f"unique_semantic={vp['unique_semantic']:.3f}  unique_time={vp['unique_time']:.3f}")

    if not results:
        print("No results.")
        return
    plot_variance_partition(results)
    plot_scatters(results)
    plot_gradient(results)
    plot_neighbour_content(results)
    save(results)
    print("Done.")


if __name__ == "__main__":
    main()
