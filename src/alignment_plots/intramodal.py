"""
intramodal.py — within-EEG-model size-scaling alignment.

For each EEG model with ≥2 sizes (FEMBA, LUNA, NeuroLM, STEEGFormer; REVE excluded),
computes Aristotelian mKNN between consecutive size pairs and plots:

  (a) one figure per model: observed (uncalibrated) mKNN across consecutive size
      pairs, PLUS a baseline curve = the i.i.d. permutation null (the alignment you
      get by permuting one size's window order before comparing). The gap between the
      two is the calibrated signal.
  (b) one combined figure: a single row with one subplot per model.

Outputs:
  src/alignment_plots/outputs/intramodal__<model>.png
  src/alignment_plots/outputs/intramodal__combined.png
  src/alignment_plots/outputs/intramodal.npz

Usage:
  python src/alignment_plots/intramodal.py
  python src/alignment_plots/intramodal.py --eeg-model neurolm
  python src/alignment_plots/intramodal.py --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _common as C


def compute_intra(eeg_model, k_mknn, k_perm):
    """Returns results[(a,b)] = dict(obs (S,), null_mean (S,), tau (S,))."""
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


def _xlabels(eeg_model, pairs):
    return [f"{C.fmt_size(eeg_model, a)}\n↔\n{C.fmt_size(eeg_model, b)}" for (a, b) in pairs]


def _plot_calibrated(ax, eeg_model, res, pairs, title_prefix=""):
    """Aristotelian-calibrated mKNN (chance already removed; 0 = chance)."""
    x = np.arange(len(pairs))
    cal = np.array([res[p]["s_cal"] for p in pairs])          # (n_pairs, S)
    ax.errorbar(x, cal.mean(1), yerr=cal.std(1), color="#4575b4", lw=2.6,
                marker="o", ms=8, capsize=4, label="calibrated mKNN (Aristotelian)", zorder=4)
    ax.axhline(0.0, color="black", lw=1.0, ls=":", label="calibrated chance (0)")
    C.tight_ylim(ax, [min(0.0, (cal.mean(1) - cal.std(1)).min())],
                 [(cal.mean(1) + cal.std(1)).max()])
    ax.set_xticks(x); ax.set_xticklabels(_xlabels(eeg_model, pairs), fontsize=11)
    ax.set_title(f"{title_prefix}{C.DISPLAY[eeg_model]} — calibrated", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def _plot_observed(ax, eeg_model, res, pairs, title_prefix=""):
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
    ax.set_xticks(x); ax.set_xticklabels(_xlabels(eeg_model, pairs), fontsize=11)
    ax.set_title(f"{title_prefix}{C.DISPLAY[eeg_model]} — observed vs null", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description="Intramodal EEG size-scaling mKNN.")
    ap.add_argument("--eeg-model", choices=[*C.INTRA_MODELS, "all"], default="all")
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--k-perm", type=int, default=C.K_PERM_DEFAULT)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--smoke", action="store_true", help="k_perm=20, fast sanity run.")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    C.HF_TOKEN_CACHE = C.resolve_hf_token(args.hf_token)
    k_perm = 20 if args.smoke else args.k_perm
    models = C.INTRA_MODELS if args.eeg_model == "all" else [args.eeg_model]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    all_res = {}
    payload = {}
    for m in models:
        print(f"\n=== intramodal {m} ===")
        res, pairs = compute_intra(m, args.k_mknn, k_perm)
        all_res[m] = (res, pairs)
        for (a, b), d in res.items():
            for key, arr in d.items():
                payload[f"{m}__{a}__{b}__{key}"] = arr

        # per-model: left = calibrated (Aristotelian), right = observed vs permutation null
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        _plot_calibrated(axes[0], m, res, pairs)
        _plot_observed(axes[1], m, res, pairs)
        axes[0].set_ylabel("calibrated mKNN"); axes[1].set_ylabel("mKNN (observed)")
        axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
        fig.suptitle(f"Intramodal mKNN — {C.DISPLAY[m]} (k={args.k_mknn}, K_perm={k_perm})",
                     fontsize=14)
        fig.tight_layout()
        p = out_dir / f"intramodal__{m}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")

    if len(models) > 1:
        # combined: row 0 = calibrated, row 1 = observed vs permutation null
        fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 10), squeeze=False)
        for j, m in enumerate(models):
            res, pairs = all_res[m]
            _plot_calibrated(axes[0][j], m, res, pairs)
            _plot_observed(axes[1][j], m, res, pairs)
        axes[0][0].set_ylabel("calibrated mKNN"); axes[1][0].set_ylabel("mKNN (observed)")
        axes[0][-1].legend(fontsize=7); axes[1][-1].legend(fontsize=7)
        fig.suptitle(f"Intramodal mKNN across EEG models (k={args.k_mknn}, K_perm={k_perm})  "
                     f"— top: calibrated (Aristotelian),  bottom: observed vs permutation null",
                     fontsize=15)
        fig.tight_layout()
        p = out_dir / "intramodal__combined.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")

    np.savez(out_dir / "intramodal.npz", **payload)
    print(f"  Saved {out_dir / 'intramodal.npz'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
