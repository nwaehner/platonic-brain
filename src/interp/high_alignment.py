"""
high_alignment.py — Study (1a): WHICH fractions of the video carry the alignment.

This formalises `notebooks/visualize_mknn_stimuli.ipynb`. For each EEG model it finds the
best-aligned vision model and the best (EEG-layer, vision-layer) pair (max mKNN over all
pairs), then computes the *per-window* mKNN — the shared-neighbour fraction at every video
chunk — on the EEG model's matched window grid. It then characterises how the alignment is
distributed in time:

  • concentration  — sorted curve + Lorenz/Gini ("what % of windows carry the alignment")
  • histogram      — distribution of per-window mKNN
  • temporal trace — mKNN(t) with the top windows marked
  • autocorrelation— how temporally bursty the high-alignment moments are
  • cross-model agreement — Spearman ρ of per-window mKNN across EEG models sharing a grid
  • top-window montage — the actual frames of the most-aligned chunks (needs cached video)

The per-window arrays it saves are the dependent variable consumed by Study 1b / 4.

Usage:
  python src/interp/high_alignment.py                       # all EEG models (heavy)
  python src/interp/high_alignment.py --eeg-model reve --vision-arch vjepa2
  python src/interp/high_alignment.py --eeg-model neurolm --no-video
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import _interp_common as IC
import _common as C

TOP_N = 8


def analyze_eeg(model, vision_archs, k, verbose=True):
    """Find best vision model + layer pair for one EEG model; return per-window mKNN."""
    fam = C.EEG[model]["family"]                   # vision must share EEG's window grid
    size = C.EEG[model]["sizes"][-1]               # largest EEG size
    emb = IC.try_load_emb(C.EEG[model]["fname"](size))
    if emb is None:
        return None
    eeg_avg = emb.mean(axis=2)                      # (L,W,S,D) -> (L,W,D)
    nn_eeg = C.precompute_knn_layers(eeg_avg, k)
    W = eeg_avg.shape[1]
    win_sec = IC.eeg_window_sec(model)

    best = dict(score=-1.0)
    for arch in vision_archs:
        for vs in C.VISION[arch]["sizes"]:
            vemb = IC.try_load_emb(C.VISION[arch]["path"](vs, fam))
            if vemb is None:
                continue
            if vemb.ndim == 2:
                vemb = vemb[None]
            nn_vis = C.precompute_knn_layers(vemb, k)
            la, lb, sc = IC.best_layer_pair(nn_eeg, nn_vis)
            if verbose:
                print(f"    {arch}-{vs}: max mKNN={sc:.3f} (le={la}, lv={lb})")
            if sc > best["score"]:
                best = dict(score=sc, arch=arch, size=vs, le=la, lv=lb,
                            nn_vis=nn_vis)
    if best["score"] < 0:
        return None

    per_window = IC.mknn_per_window(nn_eeg[best["le"]], best["nn_vis"][best["lv"]])
    top_idx = np.argsort(per_window)[::-1][:TOP_N]
    starts = IC.canonical_starts(win_sec)
    # temporal autocorrelation of the per-window signal
    x = per_window - per_window.mean()
    acf = np.correlate(x, x, mode="full")[len(x) - 1:]
    acf = acf / (acf[0] + 1e-12)
    return dict(model=model, eeg_size=size, vision_arch=best["arch"],
                vision_size=best["size"], le=best["le"], lv=best["lv"],
                raw_score=best["score"], per_window=per_window, top_idx=top_idx,
                starts_s=starts[:len(per_window)], win_sec=win_sec, W=W,
                gini=IC.gini(per_window), acf=acf,
                vision_label=C.vision_label(best["arch"], best["size"]),
                nn_eeg_best=nn_eeg[best["le"]],          # (W,k) — for Study 1b
                nn_vis_best=best["nn_vis"][best["lv"]])  # (W,k)


# ── plots ────────────────────────────────────────────────────────────────────────
def _grid(n):
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    return rows, cols


def plot_temporal_traces(results):
    n = len(results)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows), squeeze=False)
    for ax, (m, r) in zip(axes.ravel(), results.items()):
        t = r["starts_s"] / 60.0
        ax.plot(t, r["per_window"], color="#1a9850", lw=0.7)
        ax.scatter(t[r["top_idx"]], r["per_window"][r["top_idx"]], color="crimson",
                   s=30, zorder=5, label="top windows")
        ax.set_title(f"{C.DISPLAY[m]} × {r['vision_label']}  (raw mKNN={r['raw_score']:.3f})")
        ax.set_xlabel("time (min)"); ax.set_ylabel("per-window mKNN")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Per-window cross-modal alignment over the movie", y=1.0)
    fig.tight_layout()
    IC.savefig(fig, "high_alignment__temporal.png")


def plot_lorenz(results):
    fig, ax = plt.subplots(figsize=(11, 11))
    for m, r in results.items():
        frac, cum = IC.lorenz(r["per_window"])
        ax.plot(frac, cum, lw=1.8,
                label=f"{C.DISPLAY[m]}  (Gini={r['gini']:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="uniform")
    ax.set_xlabel("fraction of windows (most-aligned first)")
    ax.set_ylabel("cumulative share of total mKNN")
    ax.set_title("Concentration of alignment across the video")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "high_alignment__lorenz.png")


def plot_histograms(results):
    n = len(results)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.0 * rows), squeeze=False)
    for ax, (m, r) in zip(axes.ravel(), results.items()):
        ax.hist(r["per_window"], bins=40, color="#1a9850", alpha=0.8)
        ax.axvline(r["per_window"].mean(), color="crimson", lw=1.5, label="mean")
        ax.set_title(f"{C.DISPLAY[m]}"); ax.set_xlabel("per-window mKNN")
        ax.set_ylabel("count"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Distribution of per-window alignment", y=1.0)
    fig.tight_layout()
    IC.savefig(fig, "high_alignment__hist.png")


def plot_autocorr(results):
    fig, ax = plt.subplots(figsize=(12, 8))
    for m, r in results.items():
        lag = np.arange(len(r["acf"])) * r["win_sec"]
        keep = lag <= 120                           # first 2 minutes of lag
        ax.plot(lag[keep], r["acf"][keep], lw=1.5, label=C.DISPLAY[m])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("temporal lag (s)"); ax.set_ylabel("autocorrelation of per-window mKNN")
    ax.set_title("How temporally bursty are the high-alignment moments?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "high_alignment__autocorr.png")


def plot_agreement(results):
    groups = {}
    for m, r in results.items():
        groups.setdefault(r["W"], []).append(m)
    for W, models in groups.items():
        if len(models) < 2:
            continue
        M = np.eye(len(models))
        for i, a in enumerate(models):
            for j, b in enumerate(models):
                if i < j:
                    rho, _ = spearmanr(results[a]["per_window"],
                                       results[b]["per_window"])
                    M[i, j] = M[j, i] = rho
        labels = [C.DISPLAY[m] for m in models]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(models))); ax.set_yticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
        for i in range(len(models)):
            for j in range(len(models)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.045).set_label("Spearman ρ")
        ax.set_title(f"Do EEG models align on the SAME moments? (W={W})")
        fig.tight_layout()
        IC.savefig(fig, f"high_alignment__agreement_W{W}.png")


def plot_montage(r):
    IC.ensure_clips()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, w in zip(axes.ravel(), r["top_idx"]):
        t = r["starts_s"][w]
        frames = IC.get_window_frames(t, r["win_sec"])
        if frames:
            ax.imshow(frames[len(frames) // 2])
        else:
            ax.set_facecolor("#ccc")
        ax.set_title(f"w={w}  {int(t // 60)}m{int(t % 60):02d}s\nmKNN={r['per_window'][w]:.2f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{C.DISPLAY[r['model']]} × {r['vision_label']} — most-aligned chunks", y=1.0)
    fig.tight_layout()
    IC.savefig(fig, f"high_alignment__montage_{r['model']}.png")


def save(results):
    for m, r in results.items():
        IC.save_npz(f"high_alignment__{m}.npz",
                    per_window=r["per_window"], top_idx=r["top_idx"],
                    starts_s=r["starts_s"], acf=r["acf"],
                    le_star=r["le"], lv_star=r["lv"], gini=r["gini"],
                    raw_score=r["raw_score"], win_sec=r["win_sec"],
                    eeg_size=r["eeg_size"], vision_arch=r["vision_arch"],
                    vision_size=r["vision_size"], vision_label=r["vision_label"])


def main():
    ap = argparse.ArgumentParser(description="Study 1a: high-alignment characterisation.")
    ap.add_argument("--eeg-model", nargs="*", default=list(C.EEG),
                    choices=list(C.EEG))
    ap.add_argument("--vision-arch", nargs="*", default=list(C.VISION),
                    choices=list(C.VISION))
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--no-video", dest="video", action="store_false")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    IC.set_token(args.hf_token)
    results = {}
    for model in args.eeg_model:
        print(f"[{model}] searching best vision alignment…")
        r = analyze_eeg(model, args.vision_arch, args.k)
        if r is None:
            print(f"  (no data for {model})")
            continue
        results[model] = r
        print(f"  best: {r['vision_label']}  raw mKNN={r['raw_score']:.3f}  "
              f"Gini={r['gini']:.2f}")

    if not results:
        print("No results.")
        return
    plot_temporal_traces(results)
    plot_lorenz(results)
    plot_histograms(results)
    plot_autocorr(results)
    plot_agreement(results)
    if args.video:
        for r in results.values():
            plot_montage(r)
    save(results)
    print("Done.")


if __name__ == "__main__":
    main()
