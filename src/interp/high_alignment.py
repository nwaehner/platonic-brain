"""
high_alignment.py — Study (1a): WHICH fractions of the video carry the alignment.

For each EEG model and EVERY vision model, this finds the best (EEG-layer, vision-layer)
pair (max mKNN over all pairs) and computes the *per-window* mKNN — the shared-neighbour
fraction at every video chunk, on the EEG model's matched window grid. It then characterises
how the alignment is distributed in time.

Outputs are organised per EEG model, with the vision model in every filename:
  outputs/<eeg_model>/
    high_alignment_temporal_<arch-size>.png   per-window mKNN over the movie (top chunks marked)
    high_alignment_lorenz_<arch-size>.png      concentration / Lorenz curve (+ Gini)
    high_alignment_autocorr_<arch-size>.png    temporal autocorrelation of the per-window signal
    high_alignment_montage_<arch-size>.png     frames of the most-aligned chunks (cached video)
    high_alignment_<arch-size>.npz             arrays for this (eeg, vision) pair
    high_alignment_concentration_summary.png   Gini per vision model (one figure per EEG model)
    high_alignment_agreement.png               do vision models agree on the same chunks?

The per-window arrays are the dependent variable consumed by Study 1b / 4.

Usage:
  python src/interp/high_alignment.py --eeg-model reve                 # all vision models
  python src/interp/high_alignment.py --eeg-model neurolm --vision-arch vjepa2 dinov2
  python src/interp/high_alignment.py --no-video                       # skip the frame montage
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import _interp_common as IC
import _common as C

TOP_N = 8


def autocorr(x):
    """Biased autocorrelation of x at lags 0..len(x)-1, normalised so ACF(0)=1."""
    z = x - x.mean()
    a = np.correlate(z, z, mode="full")[len(z) - 1:]
    return a / (a[0] + 1e-12)


def load_eeg(model, k):
    """Load an EEG model (largest size), subject-mean, and precompute its layerwise kNN.
    Returns (nn_eeg (L,W,k), win_sec, W, fam, eeg_size) or None."""
    size = C.EEG[model]["sizes"][-1]
    emb = IC.try_load_emb(C.EEG[model]["fname"](size))
    if emb is None:
        return None
    eeg_avg = emb.mean(axis=2)                       # (L,W,S,D) -> (L,W,D)
    return (C.precompute_knn_layers(eeg_avg, k), IC.eeg_window_sec(model),
            eeg_avg.shape[1], C.EEG[model]["family"], size)


def analyze_pair(model, nn_eeg, win_sec, W, arch, size, fam, k):
    """Per-window mKNN for one (EEG model, vision model) pair at their best layer pair."""
    vemb = IC.try_load_emb(C.VISION[arch]["path"](size, fam))
    if vemb is None:
        return None
    if vemb.ndim == 2:
        vemb = vemb[None]
    nn_vis = C.precompute_knn_layers(vemb, k)
    le, lv, score = IC.best_layer_pair(nn_eeg, nn_vis)
    per_window = IC.mknn_per_window(nn_eeg[le], nn_vis[lv])
    top_idx = np.argsort(per_window)[::-1][:TOP_N]
    starts = IC.canonical_starts(win_sec)[:len(per_window)]
    return dict(model=model, arch=arch, size=size, vlabel=f"{arch}-{size}",
                vision_label=C.vision_label(arch, size), le=le, lv=lv,
                raw_score=score, per_window=per_window, top_idx=top_idx,
                starts_s=starts, win_sec=win_sec, W=W, gini=IC.gini(per_window),
                acf=autocorr(per_window),
                nn_eeg_best=nn_eeg[le], nn_vis_best=nn_vis[lv])


# ── per-pair plots ───────────────────────────────────────────────────────────────
def _title(r):
    return f"{C.DISPLAY[r['model']]} × {r['vision_label']}  (max mKNN={r['raw_score']:.3f}, le={r['le']}, lv={r['lv']})"


def plot_temporal(r):
    fig, ax = plt.subplots(figsize=(16, 6))
    t = r["starts_s"] / 60.0
    ax.plot(t, r["per_window"], color="#1a9850", lw=0.8)
    ax.scatter(t[r["top_idx"]], r["per_window"][r["top_idx"]], color="crimson", s=40,
               zorder=5, label=f"top-{TOP_N} chunks")
    ax.set_xlabel("time (min)"); ax.set_ylabel("per-window mKNN")
    ax.set_title("Per-window cross-modal alignment over the movie\n" + _title(r))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    IC.savefig(fig, f"{r['model']}/high_alignment_temporal_{r['vlabel']}.png")


def plot_lorenz(r):
    fig, ax = plt.subplots(figsize=(11, 11))
    frac, cum = IC.lorenz(r["per_window"])
    ax.plot(frac, cum, lw=2.2, color="#1a9850", label=f"Gini={r['gini']:.2f}")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="uniform")
    ax.set_xlabel("fraction of windows (most-aligned first)")
    ax.set_ylabel("cumulative share of total mKNN")
    ax.set_title("Concentration of alignment across the video\n" + _title(r))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, f"{r['model']}/high_alignment_lorenz_{r['vlabel']}.png")


def plot_autocorr(r):
    fig, ax = plt.subplots(figsize=(12, 8))
    lag = np.arange(len(r["acf"])) * r["win_sec"]
    keep = lag <= 120
    ax.plot(lag[keep], r["acf"][keep], lw=1.8, color="#1a9850")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("temporal lag τ (seconds)")
    ax.set_ylabel("autocorrelation of per-window mKNN")
    ax.set_title("Temporal structure of the alignment signal\n"
                 "slow decay ⇒ alignment comes in sustained bursts; "
                 "fast decay ⇒ isolated spikes\n" + _title(r))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, f"{r['model']}/high_alignment_autocorr_{r['vlabel']}.png")


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
    fig.suptitle("Most-aligned chunks\n" + _title(r), y=1.0)
    fig.tight_layout()
    IC.savefig(fig, f"{r['model']}/high_alignment_montage_{r['vlabel']}.png")


# ── per-EEG-model summary plots (across vision models) ───────────────────────────
def plot_concentration_summary(model, pairs):
    fig, ax = plt.subplots(figsize=(12, max(5, 0.6 * len(pairs))))
    pairs = sorted(pairs, key=lambda r: r["gini"])
    y = np.arange(len(pairs))
    colors = [C.VISION_ARCH_COLORS.get(r["arch"], "#1a9850") for r in pairs]
    for yi, r, col in zip(y, pairs, colors):
        ax.hlines(yi, 0, r["gini"], color=col, lw=2.2)
        ax.plot(r["gini"], yi, "o", color=col, ms=10)
        ax.annotate(f"max mKNN={r['raw_score']:.3f}", (r["gini"], yi), fontsize=8,
                    xytext=(8, 0), textcoords="offset points", va="center")
    ax.set_yticks(y); ax.set_yticklabels([r["vision_label"] for r in pairs], fontsize=9)
    ax.set_xlabel("Gini of per-window mKNN  (higher ⇒ alignment more concentrated)")
    ax.set_title(f"{C.DISPLAY[model]} — alignment concentration per vision model")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    IC.savefig(fig, f"{model}/high_alignment_concentration_summary.png")


def plot_agreement(model, pairs):
    if len(pairs) < 2:
        return
    labels = [r["vision_label"] for r in pairs]
    M = np.eye(len(pairs))
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            rho, _ = spearmanr(pairs[i]["per_window"], pairs[j]["per_window"])
            M[i, j] = M[j, i] = rho
    fig, ax = plt.subplots(figsize=(max(8, len(pairs)), max(7, len(pairs) * 0.9)))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(pairs))); ax.set_yticks(range(len(pairs)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045).set_label("Spearman ρ")
    ax.set_title(f"{C.DISPLAY[model]} — do vision models align on the SAME chunks?")
    fig.tight_layout()
    IC.savefig(fig, f"{model}/high_alignment_agreement.png")


def save_pair(r):
    IC.save_npz(f"{r['model']}/high_alignment_{r['vlabel']}.npz",
                per_window=r["per_window"], top_idx=r["top_idx"],
                starts_s=r["starts_s"], acf=r["acf"], le_star=r["le"], lv_star=r["lv"],
                gini=r["gini"], raw_score=r["raw_score"], win_sec=r["win_sec"],
                eeg_model=r["model"], vision_arch=r["arch"], vision_size=r["size"],
                vision_label=r["vision_label"])


def vision_specs(archs):
    for arch in archs:
        for size in C.VISION[arch]["sizes"]:
            yield arch, size


def main():
    ap = argparse.ArgumentParser(description="Study 1a: high-alignment characterisation.")
    ap.add_argument("--eeg-model", nargs="*", default=list(C.EEG), choices=list(C.EEG))
    ap.add_argument("--vision-arch", nargs="*", default=list(C.VISION),
                    choices=list(C.VISION))
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--no-video", dest="video", action="store_false")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    IC.set_token(args.hf_token)
    for model in args.eeg_model:
        loaded = load_eeg(model, args.k)
        if loaded is None:
            print(f"[{model}] no EEG data — skipping")
            continue
        nn_eeg, win_sec, W, fam, _ = loaded
        print(f"[{model}] grid family={fam}  W={W}  — scanning vision models")
        pairs = []
        for arch, size in vision_specs(args.vision_arch):
            r = analyze_pair(model, nn_eeg, win_sec, W, arch, size, fam, args.k)
            if r is None:
                continue
            print(f"    {r['vlabel']:18s} max mKNN={r['raw_score']:.3f}  Gini={r['gini']:.2f}")
            plot_temporal(r); plot_lorenz(r); plot_autocorr(r)
            if args.video:
                plot_montage(r)
            save_pair(r)
            pairs.append(r)
        if pairs:
            plot_concentration_summary(model, pairs)
            plot_agreement(model, pairs)
    print("Done.")


if __name__ == "__main__":
    main()
