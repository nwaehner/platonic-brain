"""
intramodal.py — within-model size-scaling alignment (EEG, vision, LLM).

For each model family with ≥2 sizes, computes Aristotelian mKNN between consecutive
size pairs and plots, per family:
  (a) calibrated (Aristotelian) mKNN across consecutive size pairs (0 = chance), and
  (b) observed (uncalibrated) mKNN vs the i.i.d. permutation-null band.

Modalities (``--modalities``):
  • eeg     — FEMBA, LUNA, NeuroLM, STEEGFormer (REVE excluded: <2 sizes).
              EEG has a subject axis → curves show mean±std over subjects.
  • vision  — the 4 s (clip4s) vision encoders with >2 sizes: DINOv2, VideoMAE-K400,
              V-JEPA2 (VideoMAE excluded: only 2 sizes).
  • llm     — the 4 s (clip4s) LLM caption families with >2 sizes: BLOOM, OpenLLaMA
              (LLaMA excluded: only one size on the 4 s grid).
Vision/LLM embeddings are subject-independent, so each size pair is a single
calibrated score (no error bars), with the permutation-null band giving the spread.

Outputs (per rendered family + a per-modality combined when >1 family):
  src/alignment_plots/outputs/intramodal__<key>.png
  src/alignment_plots/outputs/intramodal__combined.png          (EEG, back-compat)
  src/alignment_plots/outputs/intramodal__vision_combined.png
  src/alignment_plots/outputs/intramodal__llm_combined.png
  src/alignment_plots/outputs/intramodal.npz

Usage:
  python src/alignment_plots/intramodal.py
  python src/alignment_plots/intramodal.py --modalities eeg
  python src/alignment_plots/intramodal.py --modalities vision,llm
  python src/alignment_plots/intramodal.py --eeg-model neurolm --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _common as C


# ── compute: EEG (subject axis) ────────────────────────────────────────────────
def compute_intra(eeg_model, k_mknn, k_perm):
    """Returns results[(a,b)] = dict(obs (S,), null_mean (S,), tau (S,), s_cal (S,))."""
    sizes = C.EEG[eeg_model]["sizes"]
    pairs = list(zip(sizes[:-1], sizes[1:]))
    nn_by_size = {}
    for sz in sizes:
        emb = C.load_npz(C.EEG[eeg_model]["fname"](sz))["embeddings"].astype(np.float32)
        print(f"  {eeg_model}-{sz}: {emb.shape}  precomputing kNN...")
        nn_by_size[sz] = C.precompute_knn(emb, k_mknn)
    S = next(iter(nn_by_size.values())).shape[1]
    out = {}
    for (a, b) in pairs:
        obs, null_mean, tau, s_cal = [], [], [], []
        for s in range(S):
            T_obs, T_null = C.aristotelian_intra_subj_full(
                nn_by_size[a], nn_by_size[b], s, K=k_perm, seed=42 + s)
            t, sc, _ = C._calibrate(T_obs, T_null, C.ALPHA_DEFAULT)   # Aristotelian calibration
            obs.append(T_obs)
            null_mean.append(float(T_null.mean()))
            tau.append(t)
            s_cal.append(sc)
            print(f"    {a}<->{b} sub-{s+1:04d}: obs={T_obs:.4f}  null={T_null.mean():.4f}  cal={sc:.4f}")
        out[(a, b)] = dict(obs=np.array(obs), null_mean=np.array(null_mean),
                           tau=np.array(tau), s_cal=np.array(s_cal))
    return out, pairs


# ── compute: vision / LLM (no subject axis, 4 s grid) ──────────────────────────
def compute_intra_layers(sizes, path_fn, k_mknn, k_perm, label_for=str):
    """Subject-free intramodal for vision/LLM (clip4s). Loads each *available* size,
    runs aristotelian_cross between consecutive available sizes. Each field is a
    length-1 array so the shared EEG plotters (which take mean/std over axis 1) work."""
    nn_by_size = {}
    for sz in sizes:
        try:
            emb = C.load_npz(path_fn(sz))["embeddings"].astype(np.float32)
        except Exception as e:
            print(f"  [skip] {label_for(sz)}: {type(e).__name__}")
            continue
        nn_by_size[sz] = C.precompute_knn_layers(emb, k_mknn)   # promotes 2-D internally
        print(f"  {label_for(sz)}: nn {nn_by_size[sz].shape}  precomputed")
    avail = [s for s in sizes if s in nn_by_size]
    pairs = list(zip(avail[:-1], avail[1:]))
    out = {}
    for (a, b) in pairs:
        T_obs, s_cal, tau, _p, null_mean = C.aristotelian_cross(
            nn_by_size[a], nn_by_size[b], K=k_perm, seed=42)
        out[(a, b)] = dict(obs=np.array([T_obs]), null_mean=np.array([null_mean]),
                           tau=np.array([tau]), s_cal=np.array([s_cal]))
        print(f"    {label_for(a)}<->{label_for(b)}: obs={T_obs:.4f}  "
              f"null={null_mean:.4f}  cal={s_cal:.4f}")
    return out, pairs


# ── labels ─────────────────────────────────────────────────────────────────────
def _vis_label(arch, sz):
    p = C.VISION_PARAMS.get((arch, sz))
    return f"{sz} ({C._fmt_params(p)})" if p is not None else sz


def _xlabels(labelfn, pairs):
    return [f"{labelfn(a)}\n↔\n{labelfn(b)}" for (a, b) in pairs]


# ── plotting (generic over display name + per-size label fn) ────────────────────
def _plot_calibrated(ax, display, labelfn, res, pairs):
    """Aristotelian-calibrated mKNN (chance already removed; 0 = chance)."""
    x = np.arange(len(pairs))
    cal = np.array([res[p]["s_cal"] for p in pairs])          # (n_pairs, S|1)
    ax.errorbar(x, cal.mean(1), yerr=cal.std(1), color="#4575b4", lw=2.6,
                marker="o", ms=8, capsize=4, label="calibrated mKNN (Aristotelian)", zorder=4)
    ax.axhline(0.0, color="black", lw=1.0, ls=":", label="calibrated chance (0)")
    C.tight_ylim(ax, [min(0.0, (cal.mean(1) - cal.std(1)).min())],
                 [(cal.mean(1) + cal.std(1)).max()])
    ax.set_xticks(x); ax.set_xticklabels(_xlabels(labelfn, pairs), fontsize=11)
    ax.set_title(f"{display} — calibrated", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def _plot_observed(ax, display, labelfn, res, pairs):
    """Observed (uncalibrated) mKNN vs the i.i.d. permutation baseline band."""
    x = np.arange(len(pairs))
    obs = np.array([res[p]["obs"] for p in pairs])
    nullm = np.array([res[p]["null_mean"] for p in pairs])
    tau = np.array([res[p]["tau"] for p in pairs])
    ax.errorbar(x, obs.mean(1), yerr=obs.std(1), color="#1a9850", lw=2.6,
                marker="o", ms=8, capsize=4, label="observed mKNN", zorder=4)
    ax.plot(x, nullm.mean(1), color="black", lw=1.8, ls="--", marker="s", ms=5,
            label="permutation baseline (null mean)", zorder=3)
    ax.fill_between(x, nullm.mean(1), tau.mean(1), color="grey", alpha=0.20,
                    label="null spread (mean→95th pct)", zorder=1)
    C.tight_ylim(ax, [nullm.mean(1).min(), (obs.mean(1) - obs.std(1)).min()],
                 [(obs.mean(1) + obs.std(1)).max(), tau.mean(1).max()])
    ax.set_xticks(x); ax.set_xticklabels(_xlabels(labelfn, pairs), fontsize=11)
    ax.set_title(f"{display} — observed vs null", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def _render_entity(out_dir, key, display, labelfn, res, pairs, k_mknn, k_perm, payload):
    for (a, b), d in res.items():
        for fld, arr in d.items():
            payload[f"{key}__{a}__{b}__{fld}"] = arr
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _plot_calibrated(axes[0], display, labelfn, res, pairs)
    _plot_observed(axes[1], display, labelfn, res, pairs)
    axes[0].set_ylabel("calibrated mKNN"); axes[1].set_ylabel("mKNN (observed)")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.suptitle(f"Intramodal mKNN — {display} (k={k_mknn}, K_perm={k_perm})", fontsize=14)
    fig.tight_layout()
    p = out_dir / f"intramodal__{key}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")


_CAT_TITLE = {"eeg": "EEG models", "vision": "vision models (4 s)",
              "llm": "LLM families (4 s)"}


def _render_combined(out_dir, cat, items, k_mknn, k_perm):
    fig, axes = plt.subplots(2, len(items), figsize=(6 * len(items), 10), squeeze=False)
    for j, (display, labelfn, res, pairs) in enumerate(items):
        _plot_calibrated(axes[0][j], display, labelfn, res, pairs)
        _plot_observed(axes[1][j], display, labelfn, res, pairs)
    axes[0][0].set_ylabel("calibrated mKNN"); axes[1][0].set_ylabel("mKNN (observed)")
    axes[0][-1].legend(fontsize=7); axes[1][-1].legend(fontsize=7)
    fig.suptitle(f"Intramodal mKNN across {_CAT_TITLE[cat]} (k={k_mknn}, K_perm={k_perm})  "
                 f"— top: calibrated (Aristotelian),  bottom: observed vs permutation null",
                 fontsize=15)
    fig.tight_layout()
    fname = "intramodal__combined.png" if cat == "eeg" else f"intramodal__{cat}_combined.png"
    p = out_dir / fname
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")


def main():
    ap = argparse.ArgumentParser(description="Intramodal size-scaling mKNN (EEG/vision/LLM).")
    ap.add_argument("--modalities", default="eeg,vision,llm",
                    help="comma list of {eeg,vision,llm} to render (default all).")
    ap.add_argument("--eeg-model", choices=[*C.INTRA_MODELS, "all"], default="all")
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--k-perm", type=int, default=C.K_PERM_DEFAULT)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--smoke", action="store_true", help="k_perm=20, fast sanity run.")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    C.HF_TOKEN_CACHE = C.resolve_hf_token(args.hf_token)
    k_perm = 20 if args.smoke else args.k_perm
    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    payload = {}
    rendered = {"eeg": [], "vision": [], "llm": []}

    # ── EEG (subject axis) ──────────────────────────────────────────────────────
    if "eeg" in mods:
        models = C.INTRA_MODELS if args.eeg_model == "all" else [args.eeg_model]
        for m in models:
            print(f"\n=== intramodal eeg · {m} ===")
            res, pairs = compute_intra(m, args.k_mknn, k_perm)
            display = C.DISPLAY[m]
            labelfn = (lambda m: lambda sz: C.fmt_size(m, sz))(m)
            _render_entity(out_dir, m, display, labelfn, res, pairs,
                           args.k_mknn, k_perm, payload)
            rendered["eeg"].append((display, labelfn, res, pairs))

    # ── vision (4 s clip4s, archs with >2 sizes) ────────────────────────────────
    if "vision" in mods:
        for arch in C.VISION:
            if len(C.VISION[arch]["sizes"]) <= 2:
                continue
            print(f"\n=== intramodal vision · {arch} (4 s) ===")
            labelfn = (lambda arch: lambda sz: _vis_label(arch, sz))(arch)
            path_fn = (lambda arch: lambda sz: C.VISION[arch]["path"](sz, "clip4s"))(arch)
            res, pairs = compute_intra_layers(C.VISION[arch]["sizes"], path_fn,
                                              args.k_mknn, k_perm, labelfn)
            if not pairs:
                print(f"  [skip] {arch}: <2 clip4s sizes available"); continue
            display = f"{C.VISION[arch]['display']} (4 s)"
            _render_entity(out_dir, arch, display, labelfn, res, pairs,
                           args.k_mknn, k_perm, payload)
            rendered["vision"].append((display, labelfn, res, pairs))

    # ── LLM (4 s clip4s, families with >2 sizes) ────────────────────────────────
    if "llm" in mods:
        for fam in C.LLM:
            if len(C.LLM[fam]) <= 2:
                continue
            print(f"\n=== intramodal llm · {fam} (4 s) ===")
            labelfn = lambda st: C.LLM_LABEL.get(st, st)
            path_fn = lambda st: C.llm_path("clip4s", st)
            res, pairs = compute_intra_layers(C.LLM[fam], path_fn,
                                              args.k_mknn, k_perm, labelfn)
            if not pairs:
                print(f"  [skip] {fam}: <2 clip4s stems available"); continue
            display = f"{fam} (4 s)"
            _render_entity(out_dir, fam, display, labelfn, res, pairs,
                           args.k_mknn, k_perm, payload)
            rendered["llm"].append((display, labelfn, res, pairs))

    # ── per-modality combined figures ───────────────────────────────────────────
    for cat, items in rendered.items():
        if len(items) > 1:
            _render_combined(out_dir, cat, items, args.k_mknn, k_perm)

    np.savez(out_dir / "intramodal.npz", **payload)
    print(f"  Saved {out_dir / 'intramodal.npz'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
