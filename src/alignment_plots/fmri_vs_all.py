"""
fmri_vs_all.py — the raw fMRI representation vs every foundation model.

The Platonic Representation Hypothesis, tested with the *brain itself* on one axis:
we treat the preprocessed CineBrain fMRI (per-TR masked-voxel vector, built by
`extract_fmri.py`) as another representation of each stimulus window, then measure
its cross-modal alignment — mKNN and linear CKA — against every already-computed
foundation-model embedding:

    fMRI  vs  EEG foundation models   (femba, luna, neurolm, reve, steegformer)
    fMRI  vs  vision models           (dinov2, videomae, videomae_ft, vjepa2)
    fMRI  vs  language models          (bloom, openllama, llama)

Nothing here runs a GPU or re-extracts a model: EEG + vision load from
`nitrox639/platonic-embeddings`, LLMs from `triniborrell/…` (see _common.fetch),
and the only new artifact is the fMRI matrix on disk.

fMRI is per-subject, shape mirrors EEG: we build (1, W, S=6, D) so the exact same
mKNN primitives (`precompute_knn`, `aristotelian_cross`) apply. fMRI has no layers,
so the "layer" axis is length 1; alignment is still max-over-layer-pairs on the model
side. fMRI-vs-EEG compares the SAME subject s; fMRI-vs-vision/LLM compares each fMRI
subject against the (subject-free) model embedding and averages the S scores.

HRF ablation: BOLD lags neural activity ~4-6 s, so we pool the fMRI TRs of the window
shifted forward by `h` seconds and sweep h (default 0..8 s). The h that maximises
alignment is the empirical hemodynamic delay; the sweep is reported as a curve.

Window alignment is identical to every other extractor: window w of length L spans
stimulus time [wL, (w+1)L); Season 7 = 10,800 s; fMRI TR = 0.8 s, Season-7 TR index
0..13499 == EEG segment index.

Usage:
    # one family, default shift sweep, all model comparisons:
    python fmri_vs_all.py --family neurolm
    # a single fixed HRF shift, faster:
    python fmri_vs_all.py --family neurolm --shifts 4.8
    # every family:
    python fmri_vs_all.py --family all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import _common as C

# ── fMRI raw-matrix location ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FMRI_DIR = Path(os.environ.get("FMRI_DIR", _PROJECT_ROOT / "embeddings" / "fmri"))
OUT = Path(__file__).resolve().parent / "outputs"
SUBJECTS = [f"sub-{i:04d}" for i in range(1, 7)]

TR_SECONDS = 0.8
SEASON7_SEC = 10800

# EEG-family → the window scheme and which EEG models / LLM family key it carries.
# (vision NPZs are tagged by these same family names; LLMs live under llms/<eeg_model>/.)
FAMILIES = {
    "femba_luna":  {"win_sec": 5,  "eeg": ["femba", "luna"],   "llm_key": "femba"},
    "steegformer": {"win_sec": 6,  "eeg": ["steegformer"],     "llm_key": "steegformer"},
    "neurolm":     {"win_sec": 8,  "eeg": ["neurolm"],         "llm_key": "neurolm"},
    "reve":        {"win_sec": 10, "eeg": ["reve"],            "llm_key": "reve"},
    "clip4s":      {"win_sec": 4,  "eeg": [],                  "llm_key": "clip4s"},
}
# HRF shift(s), seconds. The preprocessed fMRI is already hemodynamically aligned
# (0 s empirically maximised alignment), so the default is a single no-shift pass.
# Pass e.g. --shifts 0 2 4 6 8 to re-run the ablation.
DEFAULT_SHIFTS = [0.0]


# ── linear CKA (feature-centred, Frobenius-normalised Gram) ────────────────────
def norm_gram(X):
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T
    return G / (np.linalg.norm(G) + 1e-20)


def best_cka(grams_a, grams_b):
    return max(float(np.sum(Ga * Gb)) for Ga in grams_a for Gb in grams_b)


# ── fMRI window embedding ──────────────────────────────────────────────────────
def load_fmri_raw():
    """Return a list of per-subject (13500, D) float32 memmaps (never all resident) and
    the subject list. Memmapping keeps the 6 GB raw stack off the heap — only the pooled
    window embedding (~0.6 GB) is ever materialised."""
    mms, subs = [], []
    for s in SUBJECTS:
        p = FMRI_DIR / f"fmri_raw_{s}.npy"
        if not p.exists():
            print(f"  [warn] missing {p} — run extract_fmri.py first")
            continue
        mms.append(np.load(p, mmap_mode="r"))
        subs.append(s)
    if not mms:
        raise SystemExit(f"No fMRI matrices in {FMRI_DIR}. Run extract_fmri.py.")
    T, D = mms[0].shape
    for m in mms:
        assert m.shape == (T, D), "inconsistent fMRI matrix shapes"
    return mms, subs


def window_pool(raw_mms, win_sec, shift_sec):
    """raw_mms: list of S per-subject (T=13500, D) memmaps → fMRI window embedding
    (1, W, S, D), mean-pooling the TRs whose centre time falls in the HRF-shifted
    window [wL+h, (w+1)L+h). Reads only the pooled rows off disk per window."""
    S = len(raw_mms)
    T, D = raw_mms[0].shape
    W = SEASON7_SEC // win_sec
    centres = (np.arange(T) + 0.5) * TR_SECONDS               # TR centre times (s)
    out = np.zeros((1, W, S, D), dtype=np.float32)
    for w in range(W):
        lo = w * win_sec + shift_sec
        hi = (w + 1) * win_sec + shift_sec
        sel = np.where((centres >= lo) & (centres < hi))[0]
        if sel.size == 0:                                     # shifted past the end
            sel = np.array([min(T - 1, int(round(lo / TR_SECONDS)))])
        for s in range(S):
            out[0, w, s] = np.asarray(raw_mms[s][sel, :]).mean(axis=0)
    return out


# ── comparison helpers ─────────────────────────────────────────────────────────
def _knn_fmri(fmri_emb, k):
    """(1,W,S,D) → nn (1,S,W,k)."""
    return C.precompute_knn(fmri_emb, k)


# ── driver for one family ──────────────────────────────────────────────────────
def _enumerate_targets(fam):
    """(kind, label, loader) for every model that exists for this family."""
    spec = FAMILIES[fam]
    targets = []
    for m in spec["eeg"]:
        for sz in C.EEG[m]["sizes"]:
            targets.append(("eeg", f"{m}-{sz}",
                            (lambda mm, ss: lambda: C.load_npz(C.EEG[mm]["fname"](ss))
                             ["embeddings"].astype(np.float32))(m, sz)))
    for arch, meta in C.VISION.items():
        for sz in meta["sizes"]:
            targets.append(("vision", f"{arch}-{sz}",
                            (lambda a, s: lambda: C.load_npz(C.VISION[a]["path"](s, fam))
                             ["embeddings"].astype(np.float32))(arch, sz)))
    llm_key = spec["llm_key"]
    for _famly, stems in C.LLM.items():
        for stem in stems:
            targets.append(("llm", C.LLM_LABEL.get(stem, stem),
                            (lambda kk, st: lambda: C.load_npz(C.llm_path(kk, st))
                             ["embeddings"].astype(np.float32))(llm_key, stem)))
    return targets


def run_family(fam, raw, shifts, k, k_perm):
    """Cost-aware structure: the fMRI side depends on the HRF shift, the model side does
    NOT — so precompute the fMRI kNN + Gram for every shift ONCE, then load each model
    exactly once and inner-loop over shifts (model kNN/Gram computed once, not per shift).
    Returns results[shift][label] = dict(mknn, mknn_cal, cka, kind)."""
    win = FAMILIES[fam]["win_sec"]
    W = SEASON7_SEC // win
    S = len(raw)
    print(f"\n=== family {fam} (win={win}s, W={W}, S={S}) ===", flush=True)

    # 1. fMRI side, cached per shift.  nn: (1,S,W,k);  grams: one W×W Gram per subject.
    fmri = {}
    for h in shifts:
        emb = window_pool(raw, win, h)                       # (1,W,S,D)
        nn = C.precompute_knn(emb, k)                        # (1,S,W,k)
        grams = [norm_gram(C.l2(emb[0, :, s, :])) for s in range(S)]
        fmri[h] = dict(nn=nn, grams=grams)
        del emb
        print(f"  [fmri] shift {h:>4}s ready", flush=True)

    # 2. each model loaded once → inner-loop over shifts
    results = {h: {} for h in shifts}
    for kind, label, loader in _enumerate_targets(fam):
        try:
            other = loader()
        except Exception as e:
            print(f"  [skip] {label}: {type(e).__name__} {str(e)[:60]}", flush=True)
            continue
        if kind == "eeg":                                    # (LE,W,S,D), same-subject
            nn_o = C.precompute_knn(other, k)                # (LE,S,W,k)
            grams_o = [[norm_gram(C.l2(other[l, :, s, :])) for l in range(other.shape[0])]
                       for s in range(S)]
            for h in shifts:
                fm = fmri[h]; mk, mkc, ck = [], [], []
                for s in range(S):
                    T, sc, _, _, _ = C.aristotelian_cross(fm["nn"][:, s], nn_o[:, s], K=k_perm)
                    mk.append(T); mkc.append(sc)
                    ck.append(best_cka([fm["grams"][s]], grams_o[s]))
                results[h][label] = dict(mknn=np.array(mk), mknn_cal=np.array(mkc),
                                         cka=np.array(ck), kind=kind)
        else:                                                # (LV,W,D), subject-free
            nn_o = C.precompute_knn_layers(other, k)
            grams_o = [norm_gram(C.l2(other[l])) for l in range(other.shape[0])]
            for h in shifts:
                fm = fmri[h]; mk, mkc, ck = [], [], []
                for s in range(S):
                    T, sc, _, _, _ = C.aristotelian_cross(fm["nn"][:, s], nn_o, K=k_perm)
                    mk.append(T); mkc.append(sc)
                    ck.append(best_cka([fm["grams"][s]], grams_o))
                results[h][label] = dict(mknn=np.array(mk), mknn_cal=np.array(mkc),
                                         cka=np.array(ck), kind=kind)
        bh = max(shifts, key=lambda hh: results[hh][label]["mknn_cal"].mean())
        print(f"  {label:<22} mKNN(cal)={results[bh][label]['mknn_cal'].mean():.3f}@{bh}s "
              f"CKA={results[bh][label]['cka'].mean():.3f}", flush=True)
    return results


def save_scores(fam, results, shifts):
    """Persist ONLY the alignment numbers (no plots — plotting lives in fmri_scaling.py).
    Per model: subject-mean calibrated mKNN + linear CKA at the best HRF shift."""
    OUT.mkdir(parents=True, exist_ok=True)
    def mean_mknn(h):
        v = [r["mknn_cal"].mean() for r in results[h].values()]
        return float(np.mean(v)) if v else np.nan
    best_h = max(shifts, key=mean_mknn)
    labels = list(results[best_h].keys())
    kinds = [results[best_h][l]["kind"] for l in labels]
    mk = np.array([results[best_h][l]["mknn_cal"].mean() for l in labels])
    ck = np.array([results[best_h][l]["cka"].mean() for l in labels])
    npz = OUT / f"fmri_vs_all__{fam}.npz"
    np.savez(npz, family=fam, best_shift=best_h,
             labels=np.array(labels), kinds=np.array(kinds), mknn_cal=mk, cka=ck)
    print(f"  saved {npz}  (HRF shift = {best_h}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="neurolm",
                    choices=list(FAMILIES) + ["all"])
    ap.add_argument("--shifts", nargs="+", type=float, default=DEFAULT_SHIFTS)
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--k-perm", type=int, default=C.K_PERM_DEFAULT)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    # Load .env if present (run_all.sh sources it on Scaleway; do it here so the script
    # runs standalone). Needs HF_TOKEN (gated nitrox639 auth) AND PLATONIC_LLM_REPO —
    # the LLM caption embeddings for every family except neurolm live only in the
    # triniborrell repo, so without this the loader defaults to nitrox639 and 404s.
    envf = _PROJECT_ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
    # _common.LLM_REPO is bound at import from PLATONIC_LLM_REPO; refresh it now.
    C.LLM_REPO = os.environ.get("PLATONIC_LLM_REPO", C.LLM_REPO)
    if args.hf_token:
        C.HF_TOKEN_CACHE = args.hf_token.strip()
    elif C.resolve_hf_token():
        C.HF_TOKEN_CACHE = C.resolve_hf_token()

    raw, subs = load_fmri_raw()
    print(f"loaded fMRI raw: {len(raw)}×{raw[0].shape} (mmap) subjects {subs}")

    fams = list(FAMILIES) if args.family == "all" else [args.family]
    for fam in fams:
        res = run_family(fam, raw, args.shifts, args.k, args.k_perm)
        save_scores(fam, res, args.shifts)


if __name__ == "__main__":
    main()
