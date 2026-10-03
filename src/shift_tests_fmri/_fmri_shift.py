"""
_fmri_shift.py — shared machinery for the fMRI circular-shift sweeps.

Both `circular_shift_fmri_vision.py` and `circular_shift_fmri_llm.py` are thin drivers
over this: they only differ in which embeddings they enumerate. Everything else — the
fMRI side, the FFT sweep, the statistics, the npz layout — is identical, and lives here
so the two modalities cannot silently drift apart.

CELL METADATA. Each npz stores an explicit table

    cells   (N,)  cell key, e.g. "dinov2-large" or "bloomz-3b"
    groups  (N,)  the scaling family: vision architecture, or LLM family
    ranks   (N,)  ordinal size rank WITHIN the group (0 = smallest)
    labels  (N,)  x-tick label, e.g. "LARGE\\n300M"

rather than making the plots re-derive it by splitting the key on a hyphen. That parsing
worked for "dinov2-large" but breaks on "bloomz-560m" / "open_llama_3b", where the family
is not a hyphen-delimited prefix. The plots read this table and stay modality-agnostic.

THE TEMPORAL-ADJACENCY TRAP (why `excl` matters here more than it did for EEG).
fMRI and vision/LLM embeddings are both smooth in time, so "nearest neighbour" largely
means "temporally adjacent window". That structure is TRANSLATION-INVARIANT: rotating one
modality preserves it, so the circular null inherits it. Measured on this data, mKNN was
0.284 against a chance floor of 0.00185, with the far-tail null at 0.269 — the null was
95% of the observed value, and its size-scaling curve had the same shape as d=0's.
Excluding a +-10 window band (+-40 s) drops mKNN 4x and makes the scaling monotonic.
Always run both; the exclusion changes the answer, so reporting only one is a choice.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "shift_tests"))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import circular_shift_null as CS  # noqa: E402
import fast_shift as FS  # noqa: E402
import circular_shift_vision_full as V  # noqa: E402  (knn_excl helpers)
import _common as C  # noqa: E402
import fmri_windows as FW  # noqa: E402

GRID = "clip4s"
WIN_SEC = 4
W_EXPECT = 2700
EPISODE_S = 1080.0
TAIL_MIN = 100


# ── stats ─────────────────────────────────────────────────────────────────────
def signed_shifts(W):
    d = np.arange(W)
    return np.where(d <= W // 2, d, d - W)


def stats_for(curve, d_signed, tail_min=TAIL_MIN):
    """Subject-mean statistics — the headline numbers."""
    m = curve.mean(axis=0)
    tail = np.abs(d_signed) >= tail_min
    t = m[tail]
    z = (m[0] - t.mean()) / t.std(ddof=1)
    p = (1 + int((t >= m[0]).sum()) + 1) / (int(tail.sum()) + 2)
    return float(z), float(p), float(t.mean()), float(t.std(ddof=1)), int(tail.sum())


def stats_per_subject(curve, d_signed, tail_min=TAIL_MIN):
    """The same test run independently inside each subject."""
    tail = np.abs(d_signed) >= tail_min
    n = int(tail.sum())
    zs, ps = [], []
    for c in curve:
        t = c[tail]
        zs.append((c[0] - t.mean()) / t.std(ddof=1))
        ps.append((1 + int((t >= c[0]).sum()) + 1) / (n + 2))
    return np.array(zs, dtype=np.float64), np.array(ps, dtype=np.float64)


# ── sweep ─────────────────────────────────────────────────────────────────────
def curves_all_shifts(nn_fmri, nn_other, W, k):
    """(S,1,W,k) x (LO,W,k) -> (S,W) max-over-layer-pairs mKNN at every rotation."""
    S = nn_fmri.shape[0]
    Fo = FS.precompute_fft(nn_other, W)
    out = np.empty((S, W), dtype=np.float64)
    for s in range(S):
        Ff = FS.precompute_fft(nn_fmri[s], W)
        out[s] = FS.max_mknn_all_shifts(Ff, Fo, W, k)
        del Ff
    del Fo
    return out


def validate_fft(nn_fmri, nn_other, W, k, curve_fft, n_check=3, seed=0):
    """Recompute a few rotations by literal relabel+recount and compare to the FFT."""
    rng = np.random.default_rng(seed)
    ds = [0] + [int(x) for x in rng.integers(1, W, n_check)]
    worst = 0.0
    for d in ds:
        perm, inv = CS.roll_maps(W, d)
        nn_d = CS.relabel_nn(nn_other, perm, inv)
        for s in range(nn_fmri.shape[0]):
            worst = max(worst, abs(CS.mknn_matrix(nn_fmri[s], nn_d).max()
                                   - curve_fft[s, d % W]))
    print(f"  [validate] max |FFT - direct| over {len(ds)} rotations x "
          f"{nn_fmri.shape[0]} subjects = {worst:.2e}")
    return worst < 1e-6


def build_fmri_knn(k, excl):
    """fMRI -> (S,1,W,k) neighbour sets on the clip4s grid."""
    emb, subs = FW.fmri_embedding(WIN_SEC, shift_sec=0.0)
    _, W, S, _ = emb.shape
    if W != W_EXPECT:
        raise SystemExit(f"fMRI W={W}, expected {W_EXPECT} for the {GRID} grid — abort.")
    print(f"  chance mKNN floor = k/(W-1) = {k}/{W-1} = {k/(W-1):.5f}")
    if excl:
        print(f"  temporal exclusion: +-{excl} windows (+-{excl*WIN_SEC}s) barred from kNN")
    t0 = time.time()
    if excl:
        nn = V.precompute_knn_excl(emb, k, excl).transpose(1, 0, 2, 3)
    else:
        nn = C.precompute_knn(emb, k).transpose(1, 0, 2, 3)
    del emb
    print(f"  fMRI kNN {nn.shape} in {time.time()-t0:.0f}s", flush=True)
    return nn, subs, W


def sweep(targets, npz_path, k=None, tail_min=TAIL_MIN, excl=0, do_validate=True,
          modality="vision"):
    """Run every target against the fMRI and persist.

    targets: list of (key, group, rank, label, path) — `path` is the HF relative path.
    """
    k = k or C.K_MKNN_DEFAULT
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    nn_fmri, subs, W = build_fmri_knn(k, excl)
    d_signed = signed_shifts(W)
    store = {"shifts": d_signed, "subjects": np.array(subs), "excl": excl,
             "grid": GRID, "win_sec": WIN_SEC, "k_mknn": k, "tail_min": tail_min,
             "modality": modality}
    meta = []

    def checkpoint():
        if meta:
            store["cells"] = np.array([m[0] for m in meta])
            store["groups"] = np.array([m[1] for m in meta])
            store["ranks"] = np.array([m[2] for m in meta])
            store["labels"] = np.array([m[3] for m in meta])
        np.savez_compressed(npz_path, **store)

    validated = not do_validate
    for key, group, rank, label, path in targets:
        try:
            emb = CS.load_retry(path)["embeddings"].astype(np.float32)
        except Exception as e:
            print(f"  [skip] {key}: {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
        if emb.ndim == 2:
            emb = emb[None]
        if emb.shape[1] != W:
            print(f"  [skip] {key}: W={emb.shape[1]} != fMRI W={W}", flush=True)
            continue
        t0 = time.time()
        nn_o = (V.precompute_knn_layers_excl(emb, k, excl) if excl
                else C.precompute_knn_layers(emb, k))
        del emb
        curve = curves_all_shifts(nn_fmri, nn_o, W, k)
        if not validated:
            validated = validate_fft(nn_fmri, nn_o, W, k, curve)
            if not validated:
                raise SystemExit("FFT validation FAILED — refusing to continue.")
        del nn_o

        z, p, tmu, tsd, n_tail = stats_for(curve, d_signed, tail_min)
        zs, ps = stats_per_subject(curve, d_signed, tail_min)
        store[f"{key}__curve"] = curve
        store[f"{key}__obs"] = curve.mean(axis=0)[0]
        store[f"{key}__z_tail"] = z
        store[f"{key}__p_tail"] = p
        store[f"{key}__tail_mean"] = tmu
        store[f"{key}__tail_sd"] = tsd
        store[f"{key}__z_subj"] = zs
        store[f"{key}__p_subj"] = ps
        meta.append((key, group, rank, label))
        checkpoint()
        print(f"  {key:22} obs={curve.mean(axis=0)[0]:.5f} tail={tmu:.5f} "
              f"z={z:+.2f} p={p:.4f}  subj z={np.round(zs,2).tolist()} "
              f"({time.time()-t0:.0f}s)", flush=True)

    checkpoint()
    print(f"\nsaved {npz_path}")


# ── env ───────────────────────────────────────────────────────────────────────
def setup_env(hf_token=None, default_repo=None):
    """Resolve the HF token and the repo holding the clip4s embeddings.

    `_common.fetch` routes both llms/ and vision/*__clip4s.npz to C.LLM_REPO, which
    defaults to nitrox639 — where neither exists. .env is gitignored and may be absent,
    so fall back to the repo that actually holds them.
    """
    envf = _HERE.parents[1] / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k_, v_ = line.split("=", 1)
                os.environ.setdefault(k_.strip(), v_.strip())
    C.LLM_REPO = os.environ.get("PLATONIC_LLM_REPO") or default_repo or CS.DEFAULT_LLM_REPO
    print(f"  clip4s embedding repo = {C.LLM_REPO}")
    tok = hf_token or C.resolve_hf_token()
    if tok:
        C.HF_TOKEN_CACHE = tok


def run_self_tests():
    print("── self-tests (synthetic, no network) ──")
    if not (CS.self_test() and FS.self_test()):
        raise SystemExit("self-test FAILED — refusing to run on real data.")
