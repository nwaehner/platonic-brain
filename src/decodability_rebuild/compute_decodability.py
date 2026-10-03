#!/usr/bin/env python
"""
compute_decodability.py — recompute the NICE-decodability axis from scratch, per subject.

Rebuilds the x axis of RESULTS.md panel (b) starting only from
  * EEG embeddings          nitrox639/platonic-embeddings   eeg/<model>/<...>.npz  (L,W,S,D)
  * per-subject NICE targets triniborrell/...  eeg_features/raw/<sub>__<grid>__eegfeat.npz
Nothing is copied from eeg_alignment/summary.npz — the ridge is run again here.

WHAT IS SAVED, and why it is the full array rather than the summary.

The published pipeline stores one scalar per model. It reaches that scalar by three
collapses — max over LAYERS, mean over 16 FEATURES, and (in the published run) a mean
over SUBJECTS applied to the data *before* fitting. Each collapse throws away something
a later question needs, and each is contestable:

  * max over layers      is selection on the CV statistic, and L varies 2..48 across the
                         models being compared, so the inflation is unequal.
  * mean over features   R2 is capped at +1 but unbounded below, so the mean is dragged
                         by failures; and the 16 features are not 16 independent things.
  * mean over subjects   averaging 6 subjects on both sides of the ridge cuts noise ~6x.

So this script stores the UNCOLLAPSED array

    R2_subj      (S=6, L, F=16)     one R2 per subject x layer x feature
    R2_meanview  (L, F=16)          the published design: subject-mean X and y, fitted once

from which every collapse can be applied afterwards — including a fixed layer budget or a
per-feature analysis — without recomputing anything.

Grids: 5 model families share 4 window grids; luna reuses femba's 5 s / 2160-window grid,
exactly as combine_eeg_features.py does. Verified present for all 24 subject x grid files.

Outputs -> cache/decodability__<model>-<size>.npz   (checkpointed per model)

Usage:
    python compute_decodability.py                 # all 14 models
    python compute_decodability.py --models neurolm-b steegformer-large
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import _common as C  # noqa: E402

CACHE = _HERE / "cache"
REPO_EEG = "nitrox639/platonic-embeddings"
REPO_FEAT = "triniborrell/platonic-embeddings"
SUBJECTS = [f"sub-{i:04d}" for i in range(1, 7)]
# 5 families, 4 grids: luna is recorded on femba's 5 s tiling.
GRID_OF = {"femba": "femba", "luna": "femba", "neurolm": "neurolm",
           "reve": "reve", "steegformer": "steegformer"}
ALPHAS = np.logspace(-1, 4, 10)
N_FOLDS = 5


def _tok():
    p = _HERE.parents[1] / "tokens" / "hf_token.txt"
    return p.read_text().strip() if p.exists() else None


def fetch(repo, path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", token=_tok())


def clean(F):
    """Column-mean impute non-finite entries — the pipeline's _clean, reproduced."""
    F = np.array(F, float)
    cm = np.nanmean(F, axis=0)
    cm = np.where(np.isfinite(cm), cm, 0.0)
    bad = np.where(~np.isfinite(F))
    F[bad] = np.take(cm, bad[1])
    return F


def ridge_path_r2(X, Y, alphas=ALPHAS, n_folds=N_FOLDS):
    """Best-alpha mean-fold CV R2 per column of Y. (F,) with NaN where unfittable.

    Identical maths to src/interp/feature_search/feature_gridsearch.ridge_path_r2:
    5 CONTIGUOUS folds (adjacent windows are autocorrelated, so interleaved folds would
    leak neighbours into training), X standardised on train-fold statistics only, one SVD
    per fold shared across every alpha and every feature, per-fold R2 clipped at -10, and
    features with no test-block variance dropped from that fold's average.
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    W, F = Y.shape
    b = np.linspace(0, W, n_folds + 1).astype(int)
    folds = [np.arange(b[i], b[i + 1]) for i in range(n_folds)]
    acc = np.zeros((len(alphas), F))
    n_ok = np.zeros(F)
    for te in folds:
        tr = np.setdiff1d(np.arange(W), te)
        if len(te) < 3 or len(tr) < 5:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = Y[tr].mean(0)
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        UtY = U.T @ (Y[tr] - ym)
        yvar = Y[te].var(0)
        valid = yvar > 1e-9
        sstot = yvar * len(te) + 1e-12
        n_ok += valid
        for ai, al in enumerate(alphas):
            B = Vt.T @ ((s / (s ** 2 + al))[:, None] * UtY)
            pred = Xte @ B + ym
            r2 = np.clip(1.0 - ((Y[te] - pred) ** 2).sum(0) / sstot, -10.0, 1.0)
            acc[ai] += np.where(valid, r2, 0.0)
    gvar = Y.var(0)
    n = np.where(n_ok > 0, n_ok, np.nan)
    best = (acc / n).max(0)
    return np.where((n_ok > 0) & (gvar > 1e-9), best, np.nan)


def run_model(model, size, overwrite=False):
    key = f"{model}-{size}"
    out = CACHE / f"decodability__{key}.npz"
    if out.exists() and not overwrite:
        print(f"[{key}] cached, skipping")
        return
    grid = GRID_OF[model]
    t0 = time.time()
    emb = np.load(fetch(REPO_EEG, C.EEG[model]["fname"](size)),
                  allow_pickle=True)["embeddings"]
    L, We, S, D = emb.shape

    feats, names = [], None
    for sub in SUBJECTS:
        z = np.load(fetch(REPO_FEAT, f"eeg_features/raw/{sub}__{grid}__eegfeat.npz"),
                    allow_pickle=True)
        feats.append(np.asarray(z["feat"], float))
        names = [str(x) for x in z["feat_names"]]
    Wf = min(f.shape[0] for f in feats)
    W = min(We, Wf)
    Y_subj = np.stack([clean(f[:W]) for f in feats])        # (S,W,F)
    F = Y_subj.shape[2]
    print(f"[{key}] emb {emb.shape} grid={grid} -> W={W}, L={L}, D={D}, F={F} "
          f"({time.time()-t0:.0f}s to load)", flush=True)

    R2_subj = np.full((S, L, F), np.nan)
    for s in range(S):
        ts = time.time()
        for l in range(L):
            R2_subj[s, l] = ridge_path_r2(C.l2(emb[l, :W, s, :]), Y_subj[s])
        print(f"  [{key}] subject {s+1}/{S} done ({time.time()-ts:.0f}s) "
              f"decodability={np.nanmean(np.nanmax(R2_subj[s], axis=0)):+.4f}", flush=True)

    # the published design, for comparison: average BOTH sides first, then fit once
    emb_mean = emb[:, :W].mean(axis=2)
    Y_mean = clean(Y_subj.mean(axis=0))
    R2_meanview = np.stack([ridge_path_r2(C.l2(emb_mean[l]), Y_mean) for l in range(L)])

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, R2_subj=R2_subj, R2_meanview=R2_meanview,
                        feat_names=np.array(names), model=model, size=size,
                        family=C.EEG[model]["family"], grid=grid,
                        n_layers=L, n_windows=W, dim=D,
                        subjects=np.array(SUBJECTS))
    ps = np.nanmean(np.nanmax(R2_subj, axis=1), axis=1)     # (S,) per-subject scalar
    mv = float(np.nanmean(np.nanmax(R2_meanview, axis=0)))
    print(f"[{key}] SAVED  per-subject={np.round(ps,3).tolist()} "
          f"mean={ps.mean():+.4f} | subject-mean-view={mv:+.4f} "
          f"({time.time()-t0:.0f}s)\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    all_keys = [f"{m}-{s}" for m in C.EEG for s in C.EEG[m]["sizes"]]
    ap.add_argument("--models", nargs="+", default=all_keys, choices=all_keys)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    C.HF_TOKEN_CACHE = _tok()
    print(f"recomputing decodability for {len(args.models)} models\n")
    for k in args.models:
        m, s = k.rsplit("-", 1)
        run_model(m, s, args.overwrite)
    print(f"done — cache in {CACHE}")


if __name__ == "__main__":
    main()
