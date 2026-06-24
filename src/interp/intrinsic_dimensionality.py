"""
intrinsic_dimensionality.py — Study (2): the geometry of every FM's representation.

For each EEG / vision / LLM model on `nitrox639/platonic-embeddings` this measures, at
every layer, the *intrinsic dimensionality* (ID) of the representational manifold:

  • participation ratio   — LINEAR ID (Σλ)²/Σλ²
  • TwoNN (Facco 2017)     — NON-LINEAR ID
  • MLE (Levina–Bickel)    — NON-LINEAR ID (robustness companion)
  • effective rank         — exp(spectral entropy), a soft linear rank
  • per-window local TwoNN  (last layer) — manifold "thickness" per video chunk

LLM ID is new here (the manifold_analysis notebook covered EEG+vision only). Vision and
LLM default to the 4 s `clip4s` family with graceful fallback. Everything is computed
from the uploaded embeddings — no model inference.

Figures (→ outputs/):
  1 id_vs_depth        — PR / TwoNN / MLE vs relative depth, faceted by modality
  2 id_vs_scale        — last-layer ID vs #params (log)
  3 linear_vs_nonlinear— PR vs TwoNN scatter ("nonlinearity gap")
  4 id_vs_performance  — ID vs LLM 1−BPB and vs vision K400 top-1
  5 id_cross_modality  — mean ID vs depth per modality (do modalities converge?)
  6 id_vs_alignment    — last-layer ID vs mean calibrated mKNN (if alignment_plots/outputs exist)
  7 eigenspectrum      — log-log covariance spectrum decay (last layer)
  8 localid_agreement  — cross-FM Spearman agreement on per-window local ID (per window-count)

Usage:
  python src/interp/intrinsic_dimensionality.py                 # full grid (heavy)
  python src/interp/intrinsic_dimensionality.py --modalities eeg --models reve
  python src/interp/intrinsic_dimensionality.py --no-local --max-windows 1500 --layer-stride 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import _interp_common as IC
import _common as C

MOD_COLOR = {"eeg": "#1a9850", "vision": "#d95f02", "llm": "#7570b3"}
MOD_LS = {"eeg": "-", "vision": "--", "llm": ":"}


# ── per-model ID computation ─────────────────────────────────────────────────────
def _window_idx(W, max_windows):
    if not max_windows or W <= max_windows:
        return np.arange(W)
    return np.unique(np.linspace(0, W - 1, max_windows).astype(int))


def analyze(emb, do_local, max_windows, layer_stride):
    """emb (L, W, D) → dict of per-layer ID arrays + last-layer extras."""
    L, W, D = emb.shape
    layers = list(range(0, L, layer_stride))
    if layers[-1] != L - 1:
        layers.append(L - 1)                       # always include the last layer
    idx = _window_idx(W, max_windows)
    pr = np.full(L, np.nan); tnn = np.full(L, np.nan)
    mle = np.full(L, np.nan); effr = np.full(L, np.nan)
    for l in layers:
        z = C.l2(emb[l])[idx]
        ev = IC.covariance_eigs(z)
        pr[l] = IC.pr_from_eigs(ev)
        effr[l] = IC.effrank_from_eigs(ev)
        tnn[l] = IC.twonn_id(z)
        mle[l] = IC.mle_id(z)
    rel = np.linspace(0.0, 1.0, L) if L > 1 else np.array([0.5])
    rec = dict(L=L, W=W, D=D, rel=rel, pr=pr, twonn=tnn, mle=mle, eff_rank=effr,
               eig_last=IC.eigenspectrum(C.l2(emb[-1])[idx], 64),
               layers=np.array(layers))
    if do_local:
        rec["local_id"] = IC.local_twonn(emb[-1])   # full windows for agreement
    return rec


def last(arr):
    """Last finite value of a per-layer array."""
    a = np.asarray(arr)
    fin = a[np.isfinite(a)]
    return float(fin[-1]) if len(fin) else float("nan")


# ── collection over the registries ───────────────────────────────────────────────
def collect(args):
    records = {}                                   # key -> record dict (+ meta)

    def want(mod):
        return "all" in args.modalities or mod in args.modalities

    if want("eeg"):
        for model, size in IC.iter_eeg():
            if args.models and model not in args.models:
                continue
            print(f"[eeg] {model}-{size}")
            emb = IC.try_load_emb(C.EEG[model]["fname"](size))
            if emb is None:
                continue
            emb = emb.mean(axis=2)                  # (L,W,S,D) -> (L,W,D)
            rec = analyze(emb, args.local, args.max_windows, args.layer_stride)
            rec.update(modality="eeg", label=f"{C.DISPLAY[model]}-{size}",
                       params=C.PARAMS[model][size], perf=np.nan, family=model)
            records[f"eeg|{model}|{size}"] = rec

    if want("vision"):
        for arch, size in IC.iter_vision():
            if args.models and arch not in args.models:
                continue
            print(f"[vision] {arch}-{size} ({args.vision_family})")    # nitrox639 EEG-grid
            emb, used = IC.load_vision_emb(arch, size, args.vision_family)
            if emb is None:
                continue
            if emb.ndim == 2:
                emb = emb[None]
            rec = analyze(emb, args.local, args.max_windows, args.layer_stride)
            rec.update(modality="vision", label=C.vision_label(arch, size),
                       params=C.VISION_PARAMS.get((arch, size), np.nan),
                       perf=C.VIDEO_PERF.get((arch, size)) or np.nan, family=arch,
                       used_family=used)
            records[f"vision|{arch}|{size}"] = rec

    if want("llm"):
        for famly, stem in IC.iter_llm():
            if args.models and famly not in args.models:
                continue
            print(f"[llm] {stem} ({args.llm_grid})")                    # triniborrell
            emb = IC.load_llm_grid(args.llm_grid, stem)
            if emb is None:
                continue
            rec = analyze(emb, args.local, args.max_windows, args.layer_stride)
            perf = C.LLM_PERF.get(stem)
            rec.update(modality="llm", label=C.LLM_LABEL.get(stem, stem),
                       params=C.LLM_PARAMS.get(stem, np.nan),
                       perf=perf if perf is not None else np.nan, family=famly,
                       used_family=args.llm_grid)
            records[f"llm|{famly}|{stem}"] = rec

    return records


# ── plots ────────────────────────────────────────────────────────────────────────
def plot_id_vs_depth(records):
    mods = ["eeg", "vision", "llm"]
    metrics = [("pr", "participation ratio (linear ID)"),
               ("twonn", "TwoNN (non-linear ID)"),
               ("mle", "MLE (non-linear ID)")]
    fig, axes = plt.subplots(len(metrics), len(mods), figsize=(16, 13), squeeze=False)
    for r, (mk, ylab) in enumerate(metrics):
        for c, mod in enumerate(mods):
            ax = axes[r][c]
            members = [k for k, v in records.items() if v["modality"] == mod]
            cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(members), 1)))
            for col, k in enumerate(members):
                v = records[k]
                ax.plot(v["rel"], v[mk], color=cmap[col], marker="o", ms=2.5,
                        lw=1.3, label=v["label"])
            if r == 0:
                ax.set_title(mod.upper())
            if c == 0:
                ax.set_ylabel(ylab)
            ax.set_xlabel("relative depth")
            ax.grid(alpha=0.3)
            if 0 < len(members) <= 14:
                ax.legend(fontsize=6, ncol=1, loc="best")
    fig.suptitle("Intrinsic dimensionality vs depth", y=1.0)
    fig.tight_layout()
    IC.savefig(fig, "intrinsic_dimensionality/id_vs_depth.png")


def plot_id_vs_scale(records):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), squeeze=False)
    for ax, (mk, ylab) in zip(axes[0], [("pr", "last-layer participation ratio"),
                                         ("twonn", "last-layer TwoNN ID")]):
        for k, v in records.items():
            p = v["params"]
            if not np.isfinite(p):
                continue
            ax.scatter(np.log10(p), last(v[mk]), color=MOD_COLOR[v["modality"]],
                       s=55, edgecolor="black", lw=0.4)
            ax.annotate(v["label"], (np.log10(p), last(v[mk])), fontsize=5,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("log10 #params (M)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=MOD_COLOR[m], label=m)
               for m in MOD_COLOR]
    axes[0][1].legend(handles=handles, title="modality")
    fig.suptitle("Intrinsic dimensionality vs model scale", y=1.0)
    fig.tight_layout()
    IC.savefig(fig, "intrinsic_dimensionality/id_vs_scale.png")


def plot_linear_vs_nonlinear(records):
    fig, ax = plt.subplots(figsize=(12, 12))
    xs, ys = [], []
    for k, v in records.items():
        x, y = last(v["pr"]), last(v["twonn"])
        xs.append(x); ys.append(y)
        ax.scatter(x, y, color=MOD_COLOR[v["modality"]], s=60, edgecolor="black", lw=0.4)
        ax.annotate(v["label"], (x, y), fontsize=5, xytext=(3, 3),
                    textcoords="offset points")
    lim = [0, np.nanmax(xs + ys) * 1.05]
    ax.plot(lim, lim, color="grey", ls="--", lw=1, label="linear = non-linear")
    ax.set_xlabel("participation ratio (linear ID)")
    ax.set_ylabel("TwoNN (non-linear ID)")
    ax.set_title("Linear vs non-linear ID — distance below the diagonal ≈ manifold curvature")
    handles = [plt.Line2D([], [], marker="o", ls="", color=MOD_COLOR[m], label=m)
               for m in MOD_COLOR]
    ax.legend(handles=handles + [plt.Line2D([], [], color="grey", ls="--",
              label="x = y")])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "intrinsic_dimensionality/linear_vs_nonlinear.png")


def plot_id_vs_performance(records):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), squeeze=False)
    # LLM: x = 1-BPB (or proxy)
    axL = axes[0][0]
    for k, v in records.items():
        if v["modality"] != "llm":
            continue
        stem = k.split("|")[-1]
        axL.scatter(C.llm_x(stem), last(v["twonn"]), color=MOD_COLOR["llm"], s=55,
                    edgecolor="black", lw=0.4)
        axL.annotate(v["label"], (C.llm_x(stem), last(v["twonn"])), fontsize=5,
                     xytext=(3, 3), textcoords="offset points")
    axL.set_xlabel(f"LLM x  {C.llm_axis_tag()}")
    axL.set_ylabel("last-layer TwoNN ID")
    axL.set_title("LLM ID vs performance")
    axL.grid(alpha=0.3)
    # Vision: x = K400 top-1
    axV = axes[0][1]
    for k, v in records.items():
        if v["modality"] != "vision" or not np.isfinite(v["perf"]):
            continue
        axV.scatter(v["perf"], last(v["twonn"]), color=MOD_COLOR["vision"], s=55,
                    edgecolor="black", lw=0.4)
        axV.annotate(v["label"], (v["perf"], last(v["twonn"])), fontsize=5,
                     xytext=(3, 3), textcoords="offset points")
    axV.set_xlabel("Kinetics-400 top-1 (%)")
    axV.set_ylabel("last-layer TwoNN ID")
    axV.set_title("Vision ID vs performance")
    axV.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "intrinsic_dimensionality/id_vs_performance.png")


def plot_cross_modality(records):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), squeeze=False)
    grid = np.linspace(0, 1, 11)
    for ax, (mk, ylab) in zip(axes[0], [("pr", "participation ratio"),
                                         ("twonn", "TwoNN ID")]):
        for mod in ["eeg", "vision", "llm"]:
            curves = []
            for v in records.values():
                if v["modality"] != mod:
                    continue
                y = np.asarray(v[mk]); rel = v["rel"]
                fin = np.isfinite(y)
                if fin.sum() < 2:
                    continue
                curves.append(np.interp(grid, rel[fin], y[fin]))
            if not curves:
                continue
            curves = np.array(curves)
            m, s = curves.mean(0), curves.std(0)
            ax.plot(grid, m, color=MOD_COLOR[mod], lw=2, label=mod)
            ax.fill_between(grid, m - s, m + s, color=MOD_COLOR[mod], alpha=0.15)
        ax.set_xlabel("relative depth")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(title="modality")
    fig.suptitle("Cross-modality ID vs depth (mean ± sd across models)", y=1.0)
    fig.tight_layout()
    IC.savefig(fig, "intrinsic_dimensionality/id_cross_modality.png")


def plot_eigenspectrum(records):
    fig, ax = plt.subplots(figsize=(13, 10))
    for v in records.values():
        ev = v["eig_last"]
        ev = ev[np.isfinite(ev) & (ev > 0)]
        ax.loglog(np.arange(1, len(ev) + 1), ev, color=MOD_COLOR[v["modality"]],
                  lw=1, alpha=0.7)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("normalised eigenvalue")
    ax.set_title("Last-layer covariance spectrum decay")
    handles = [plt.Line2D([], [], color=MOD_COLOR[m], label=m) for m in MOD_COLOR]
    ax.legend(handles=handles, title="modality")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    IC.savefig(fig, "intrinsic_dimensionality/eigenspectrum.png")


def plot_localid_agreement(records):
    groups = {}
    for k, v in records.items():
        if "local_id" in v:
            groups.setdefault(v["W"], []).append(k)
    for W, keys in groups.items():
        if len(keys) < 2:
            continue
        M = np.eye(len(keys))
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                if i < j:
                    rho, _ = spearmanr(records[a]["local_id"], records[b]["local_id"])
                    M[i, j] = M[j, i] = rho
        labels = [records[k]["label"] for k in keys]
        fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.9),
                                        max(7, len(keys) * 0.8)))
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(keys))); ax.set_yticks(range(len(keys)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(M[i, j]) > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.045).set_label("Spearman ρ")
        ax.set_title(f"Cross-FM agreement on per-window local ID (W={W})")
        fig.tight_layout()
        IC.savefig(fig, f"intrinsic_dimensionality/localid_agreement_W{W}.png")


def plot_id_vs_alignment(records):
    """Best-effort: scatter last-layer TwoNN vs mean calibrated mKNN read from
    src/alignment_plots/outputs/*.npz. Skips silently if nothing maps."""
    out = IC._ALIGN_DIR / "outputs"
    if not out.exists():
        print("  (no alignment_plots/outputs — skipping id_vs_alignment)")
        return
    align = {}                                     # family name -> [s_cal means]
    for npz in out.glob("*.npz"):
        try:
            z = np.load(npz, allow_pickle=True)
        except Exception:
            continue
        vals = [np.asarray(z[k], float).mean() for k in z.files if k.endswith("s_cal")]
        if not vals:
            continue
        m = float(np.nanmean(vals))
        for v in records.values():
            if v["family"] and v["family"] in npz.stem:
                align.setdefault(v["family"], []).append(m)
    pts = []
    for v in records.values():
        if v["family"] in align:
            pts.append((float(np.nanmean(align[v["family"]])), last(v["twonn"]),
                        v["modality"], v["label"]))
    if len(pts) < 3:
        print("  (insufficient alignment matches — skipping id_vs_alignment)")
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    for x, y, mod, lab in pts:
        ax.scatter(x, y, color=MOD_COLOR[mod], s=60, edgecolor="black", lw=0.4)
        ax.annotate(lab, (x, y), fontsize=5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean calibrated mKNN (alignment_plots)")
    ax.set_ylabel("last-layer TwoNN ID")
    ax.set_title("Does intrinsic dimensionality track cross-modal alignment?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "intrinsic_dimensionality/id_vs_alignment.png")


# ── persistence ──────────────────────────────────────────────────────────────────
def save(records):
    flat = {}
    keys = list(records)
    for k, v in records.items():
        for fld in ("rel", "pr", "twonn", "mle", "eff_rank", "eig_last", "layers"):
            flat[f"{k}__{fld}"] = v[fld]
        if "local_id" in v:
            flat[f"{k}__local_id"] = v["local_id"]
    flat["model_keys"] = np.array(keys)
    flat["labels"] = np.array([records[k]["label"] for k in keys])
    flat["modality"] = np.array([records[k]["modality"] for k in keys])
    flat["params"] = np.array([records[k]["params"] for k in keys], dtype=float)
    flat["perf"] = np.array([records[k]["perf"] for k in keys], dtype=float)
    flat["pr_last"] = np.array([last(records[k]["pr"]) for k in keys])
    flat["twonn_last"] = np.array([last(records[k]["twonn"]) for k in keys])
    flat["mle_last"] = np.array([last(records[k]["mle"]) for k in keys])
    IC.save_npz("intrinsic_dimensionality/intrinsic_dimensionality.npz", **flat)


def main():
    ap = argparse.ArgumentParser(description="Study 2: intrinsic dimensionality.")
    ap.add_argument("--modalities", nargs="+", default=["all"],
                    choices=["all", "eeg", "vision", "llm"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these family names (e.g. reve dinov2 bloom).")
    ap.add_argument("--vision-family", default="femba_luna",
                    help="vision window family on nitrox639 (EEG grid; default femba_luna).")
    ap.add_argument("--llm-grid", default="clip4s",
                    help="LLM caption grid on triniborrell (default clip4s; or an EEG grid).")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="subsample windows for the global ID estimators.")
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--no-local", dest="local", action="store_false",
                    help="skip per-window local TwoNN (the expensive part).")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    IC.set_token(args.hf_token)
    records = collect(args)
    if not records:
        print("No records computed — check token / model availability.")
        return
    print(f"\nComputed ID for {len(records)} models. Plotting…")
    plot_id_vs_depth(records)
    plot_id_vs_scale(records)
    plot_linear_vs_nonlinear(records)
    plot_id_vs_performance(records)
    plot_cross_modality(records)
    plot_eigenspectrum(records)
    if args.local:
        plot_localid_agreement(records)
    plot_id_vs_alignment(records)
    save(records)
    print("Done.")


if __name__ == "__main__":
    main()
