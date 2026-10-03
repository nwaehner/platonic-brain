"""
circular_shift_llm_full.py — EVERY circular rotation, every EEG family x every LLM stem.

The language counterpart of circular_shift_vision_full.py, and it imports that module's
machinery rather than restating it, so the two analyses cannot drift apart: same
`curves_all_shifts` (FFT over neighbour offsets), same `stats_for` / `stats_per_subject`,
same conventions.

WHY THE FULL ROTATION SET
-------------------------
The step-10 sweep left 31 of 60 panels pinned at the permutation floor 2/84 = 0.024,
unable to distinguish "barely passes" from "overwhelming". Using every one of the W-1
non-identity rotations makes the test exact and drops the floor to ~2/(W-199+2).

A DATA CONSISTENCY FIX
----------------------
The step-10 LLM sweep read NeuroLM's captions from `nitrox639/platonic-embeddings` and
the other four families from `triniborrell/platonic-embeddings`. Those files are NOT the
same — all eight neurolm stems differ by sha256, because Trinidad re-extracted them with
the generic `extract_llm_captions.py` while nitrox639 still holds the output of the
frozen NeuroLM-only extractor. The old grid figure therefore compared one row against a
different data source than the other four. This script reads every family from ONE repo
(`--llm-repo`, default triniborrell, which is the only one carrying all five grids), so
the comparison is finally like-for-like. `--compare-repos` quantifies the difference on
NeuroLM instead of assuming it is negligible.

llama-13b is excluded: it is the only llama size on HF, so it contributes an isolated
point with no within-family scaling, which is what these figures are about.

Outputs, one subfolder per EEG family:

    outputs_vs_llm/<model>/circular_shift_full__<model>.npz
    outputs_vs_llm/<model>/shift_full__<model>__{bloom,openllama}.png

Usage:
    python circular_shift_llm_full.py                       # all five families
    python circular_shift_llm_full.py --validate femba      # FFT vs the saved step-10 run
    python circular_shift_llm_full.py --compare-repos       # neurolm: nitrox639 vs triniborrell
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import circular_shift_null as CS  # noqa: E402
import circular_shift_vision_full as V  # noqa: E402  (shared FFT + stats machinery)
import fast_shift as FS  # noqa: E402
import _common as C  # noqa: E402

OUT = _HERE / "outputs_vs_llm"
LLM_FAMILIES = {"bloom": C.LLM["bloom"], "openllama": C.LLM["openllama"]}
STEMS = LLM_FAMILIES["bloom"] + LLM_FAMILIES["openllama"]
DEFAULT_LLM_REPO = "triniborrell/platonic-embeddings"
EPISODE_S = 1080.0
TAIL_MIN = V.TAIL_MIN


def short(stem):
    return stem.replace("bloomz-", "").replace("open_llama_", "")


def plot_family(model, llm_fam, z, d_signed, out_path):
    """rows = EEG size, cols = LLM stem of this family."""
    W = C.EEG_WINDOWS[model]["n_windows"]
    ws = C.EEG_WINDOWS[model]["win_sec"]
    ep_win = int(round(EPISODE_S / ws))
    ep_lines = [n * ep_win for n in range(-(W // 2) // ep_win, W // 2 // ep_win + 1) if n]

    sizes = C.EEG[model]["sizes"]
    stems = LLM_FAMILIES[llm_fam]
    order = np.argsort(d_signed)
    ds = d_signed[order]

    fig, axes = plt.subplots(len(sizes), len(stems),
                             figsize=(4.3 * len(stems), 3.2 * len(sizes)), squeeze=False)
    ntail = 0
    for r, sz in enumerate(sizes):
        for c, st in enumerate(stems):
            ax = axes[r][c]
            key = f"{sz}__{st}__curve"
            if key not in z.files:
                ax.axis("off")
                continue
            curve = z[key]
            m = curve.mean(axis=0)[order]
            for j, x in enumerate(ep_lines):
                ax.axvline(x, color="#b2182b", lw=1.0, ls="--", alpha=0.55, zorder=2,
                           label=f"episode boundary (every {ep_win} win = 1080 s)"
                           if j == 0 else None)
            ax.plot(ds, m, color="#4477aa", lw=0.6, alpha=0.85, zorder=3,
                    label=f"all {W} rotations")
            ax.plot([0], [m[ds == 0]], "*", ms=15, color="crimson",
                    mec="black", mew=0.6, zorder=8, label="d = 0")
            zz, pp, ntail = V.stats_for(curve, d_signed)
            ax.set_title(f"{C.DISPLAY[model]}-{sz.upper()} × {short(st)}",
                         fontsize=9.5, fontweight="bold", pad=15)
            ax.text(0.5, 1.012, f"z = {zz:+.2f}   p = {pp:.4f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color="#b2182b" if pp <= 0.05 else "#666666")
            ax.set_xlim(-W // 2, W // 2)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)
            if r == len(sizes) - 1:
                ax.set_xlabel("circular shift d (windows)", fontsize=9)
            if c == 0:
                ax.set_ylabel("mKNN (uncal.)", fontsize=9)
    axes[0][0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.suptitle(
        f"{C.DISPLAY[model]} × {llm_fam.upper()} captions — every one of the {W} circular "
        f"rotations ({ws}s windows)\n"
        f"Dashed red = episode boundaries, every {ep_win} windows = 1080 s = 18 min.  "
        f"Exact permutation p over {ntail} valid rotations (|d| ≥ {TAIL_MIN}), "
        f"floor {2 / (ntail + 2):.4f}.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path.name}")


def sweep(model, replot=False):
    W = C.EEG_WINDOWS[model]["n_windows"]
    k = C.K_MKNN_DEFAULT
    sizes = C.EEG[model]["sizes"]
    sub = OUT / model
    sub.mkdir(parents=True, exist_ok=True)
    npz_path = sub / f"circular_shift_full__{model}.npz"
    d_signed = V.signed_shifts(W)

    if not replot:
        print(f"\n=== {C.DISPLAY[model]}  W={W}  ({C.EEG_WINDOWS[model]['win_sec']}s "
              f"windows) — {W - 1} non-identity rotations  [{C.LLM_REPO}] ===")
        nn_llm = {}
        for st in STEMS:
            try:
                emb = CS.load_retry(C.llm_path(model, st))["embeddings"].astype(np.float32)
            except Exception as e:
                print(f"  [skip] {st}: {type(e).__name__}: {str(e)[:100]}")
                continue
            nn_llm[st] = C.precompute_knn_layers(emb, k)
            del emb
        print(f"  {len(nn_llm)} LLM stems loaded")

        payload = {"shifts": d_signed}
        for sz in sizes:
            emb = CS.load_retry(C.EEG[model]["fname"](sz))["embeddings"].astype(np.float32)
            nn_eeg = C.precompute_knn(emb, k).transpose(1, 0, 2, 3)
            LE = nn_eeg.shape[1]
            del emb
            for st, nv in nn_llm.items():
                if nv.shape[1] != W:
                    print(f"    [skip] {st}: W={nv.shape[1]} != {W}")
                    continue
                t0 = time.perf_counter()
                curve = V.curves_all_shifts(nn_eeg, nv, W, k)
                zz, pp, _ = V.stats_for(curve, d_signed)
                zsub, psub = V.stats_per_subject(curve, d_signed)
                payload[f"{sz}__{st}__curve"] = curve
                payload[f"{sz}__{st}__z_subj"] = zsub
                payload[f"{sz}__{st}__p_subj"] = psub
                payload[f"{sz}__{st}__z_mean"] = zz
                payload[f"{sz}__{st}__p_mean"] = pp
                print(f"    {model}-{sz:6s} × {st:16s} LE={LE:2d} LV={nv.shape[0]:2d}  "
                      f"z={zz:+5.2f} p={pp:.4f}  | subj z: "
                      f"{' '.join(f'{v:+.1f}' for v in zsub)}  ({time.perf_counter() - t0:4.1f}s)")
            del nn_eeg
        np.savez_compressed(npz_path, **payload)
        print(f"  Saved {npz_path}")

    z = np.load(npz_path, allow_pickle=True)
    for fam in LLM_FAMILIES:
        plot_family(model, fam, z, d_signed, sub / f"shift_full__{model}__{fam}.png")


def validate(model):
    """FFT over all rotations vs the saved step-10 direct run, on the shifts they share."""
    old_p = OUT / f"circular_shift_null__{model}.npz"
    if not old_p.exists():
        sys.exit(f"no {old_p} to validate against")
    old = np.load(old_p, allow_pickle=True)
    grid10 = old["shifts"]
    W = C.EEG_WINDOWS[model]["n_windows"]
    k = C.K_MKNN_DEFAULT
    sz = C.EEG[model]["sizes"][0]
    st = STEMS[0]
    print(f"\n=== validation: {model}-{sz} × {st}  ({C.LLM_REPO}) ===")

    emb = CS.load_retry(C.llm_path(model, st))["embeddings"].astype(np.float32)
    nv = C.precompute_knn_layers(emb, k); del emb
    emb = CS.load_retry(C.EEG[model]["fname"](sz))["embeddings"].astype(np.float32)
    ne = C.precompute_knn(emb, k).transpose(1, 0, 2, 3); del emb

    fast = V.curves_all_shifts(ne, nv, W, k)
    ref = old[f"{sz}__{st}__curve"]
    err = np.abs(fast[:, grid10 % W] - ref).max()
    print(f"  max |FFT(all {W}) − direct(step 10)| = {err:.3e}"
          f"   {'OK' if err < 1e-6 else 'MISMATCH'}")
    return err < 1e-6


def compare_repos(model="neurolm"):
    """How much do the two HF copies of the same captions actually differ?"""
    W = C.EEG_WINDOWS[model]["n_windows"]
    k = C.K_MKNN_DEFAULT
    sz = C.EEG[model]["sizes"][0]
    d_signed = V.signed_shifts(W)
    emb = CS.load_retry(C.EEG[model]["fname"](sz))["embeddings"].astype(np.float32)
    ne = C.precompute_knn(emb, k).transpose(1, 0, 2, 3); del emb
    print(f"\n=== {model}-{sz}: nitrox639 vs triniblorrell captions ===")
    print(f"{'stem':16s} {'z (nitrox)':>11s} {'z (trini)':>11s} {'Δz':>7s}")
    for st in STEMS:
        out = {}
        for repo in ("nitrox639/platonic-embeddings", DEFAULT_LLM_REPO):
            C.LLM_REPO = repo
            try:
                emb = CS.load_retry(C.llm_path(model, st))["embeddings"].astype(np.float32)
            except Exception:
                out[repo] = np.nan; continue
            nv = C.precompute_knn_layers(emb, k); del emb
            out[repo] = V.stats_for(V.curves_all_shifts(ne, nv, W, k), d_signed)[0]
        a, b = out["nitrox639/platonic-embeddings"], out[DEFAULT_LLM_REPO]
        print(f"{st:16s} {a:+11.2f} {b:+11.2f} {b - a:+7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-model", nargs="+", default=list(C.EEG.keys()),
                    choices=list(C.EEG.keys()))
    ap.add_argument("--llm-repo", default=DEFAULT_LLM_REPO)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--validate", default=None, choices=list(C.EEG.keys()))
    ap.add_argument("--compare-repos", action="store_true")
    args = ap.parse_args()

    if not args.replot and not FS.self_test():
        sys.exit("fast_shift self-test failed")
    C.HF_TOKEN_CACHE = C.resolve_hf_token()
    C.LLM_REPO = args.llm_repo

    if args.compare_repos:
        compare_repos()
        return
    if args.validate:
        sys.exit(0 if validate(args.validate) else 1)

    t0 = time.perf_counter()
    for m in args.eeg_model:
        C.LLM_REPO = args.llm_repo
        sweep(m, replot=args.replot)
    print(f"\nTOTAL {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
