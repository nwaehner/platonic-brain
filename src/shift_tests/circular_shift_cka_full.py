#!/usr/bin/env python
"""
circular_shift_cka_full.py — the CKA twin of circular_shift_vision_full.py.

Same pipeline, same cells, CKA instead of mKNN:

    * PER SUBJECT. The EEG Gram is built from emb[l, :, s, :] for each subject
      separately, and the (S, W) curve is stored whole — no subject averaging anywhere.
    * NATIVE GRID. Each EEG family keeps its own W (2160 / 1800 / 1350 / 1080) and is
      paired with the partner file published on that same grid: the vision file for
      C.EEG[m]["family"], and llms/<m>/ for the LLM side. Nothing is resampled.
    * The complete cyclic group as the null: all W-1 non-identity rotations.
    * The layer pair re-maximised independently at every rotation.

This exists because decodability_rebuild/cache/cyclic_cka_*.npz is subject-mean on the
shared 10 s grid, so a CKA grid built from it is not the same quantity as the mKNN grid
and the two cannot be read side by side. This script closes that gap.

CKA under rotation is a circular cross-correlation over the Gram's diagonals (derived and
self-tested in decodability_rebuild/compute_fmri_cka_cyclic.py), so one rFFT per layer
gives every rotation. Cost is dominated by building the Grams, which is why the EEG side
is held resident for all six subjects while the partner side streams: that way each
partner's Grams are built once per family rather than once per (family, subject).

Saved -> outputs_cka/<model>/circular_shift_cka__<model>.npz
    <size>__<partner>__curve   (S, W) float32, index 0 == d = 0
    shifts (W,) signed

Usage:
    python circular_shift_cka_full.py
    python circular_shift_cka_full.py --eeg-model reve --no-llm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))
sys.path.insert(0, str(_HERE.parent / "decodability_rebuild"))

import circular_shift_null as CS  # noqa: E402
import _common as C  # noqa: E402
from compute_cka_cyclic_eeg import grams_fft, max_cka_all_shifts  # noqa: E402
from compute_fmri_cka_cyclic import self_test  # noqa: E402

OUT = _HERE / "outputs_cka"
ARCHS = ["dinov2", "videomae", "videomae_ft", "vjepa2"]


def signed_shifts(W):
    d = np.arange(W)
    return np.where(d > W // 2, d - W, d)


def partner_specs(model, with_llm=True):
    """-> [(key, loader), ...] for one EEG family, on that family's own grid."""
    fam = C.EEG[model]["family"]
    out = [(f"{a}-{vs}", (lambda a=a, vs=vs: CS.load_retry(C.VISION[a]["path"](vs, fam))))
           for a in ARCHS for vs in C.VISION[a]["sizes"]]
    if with_llm:
        for f in ("bloom", "openllama"):
            for stem in C.LLM[f]:
                out.append((stem, (lambda s=stem: C.load_npz(C.llm_path(model, s)))))
    return out


def run_model(model, with_llm=True):
    t0 = time.time()
    store = {}
    # ── EEG side, every size and every subject, held resident ────────────────
    Fe, W = {}, None
    for sz in C.EEG[model]["sizes"]:
        emb = CS.load_retry(C.EEG[model]["fname"](sz))["embeddings"]
        L, We, S, D = emb.shape
        W = We if W is None else min(W, We)
        for s in range(S):
            Fe[(sz, s)] = grams_fft(np.asarray(emb[:, :W, s, :], float), W)
        del emb
        print(f"  [eeg {model}-{sz:6}] L={L} S={S} W={W}  "
              f"({sum(v.nbytes for v in Fe.values())/1e9:.2f} GB resident, "
              f"{time.time()-t0:.0f}s)", flush=True)
    sizes = C.EEG[model]["sizes"]
    S = len({s for _, s in Fe})

    # ── partner side, streamed: Grams built once per (family, partner) ───────
    for key, load in partner_specs(model, with_llm):
        tp = time.time()
        emb = np.asarray(load()["embeddings"], float)[:, :W]
        Fp = grams_fft(emb, W)
        del emb
        for sz in sizes:
            cur = np.zeros((S, W), dtype=np.float32)
            for s in range(S):
                cur[s] = max_cka_all_shifts(Fe[(sz, s)], Fp, W)
            store[f"{sz}__{key}__curve"] = cur
        del Fp
        print(f"    [{key:18}] x {len(sizes)} sizes x {S} subjects "
              f"({time.time()-tp:.0f}s)", flush=True)

    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"circular_shift_cka__{model}.npz"
    np.savez_compressed(f, shifts=signed_shifts(W), **store)
    print(f"  [{model}] saved {f.name} ({f.stat().st_size/1e6:.1f} MB, "
          f"{len(store)} cells, {time.time()-t0:.0f}s)\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eeg-model", nargs="+", default=list(C.EEG), choices=list(C.EEG))
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    print("── self-test (synthetic, no network) ──")
    if not self_test(verbose=False):
        raise SystemExit("cka_all_shifts self-test FAILED.")
    print("  PASSED")
    C.HF_TOKEN_CACHE = (_HERE.parents[1] / "tokens" / "hf_token.txt").read_text().strip()
    for m in args.eeg_model:
        print(f"=== {m} ===", flush=True)
        run_model(m, not args.no_llm)


if __name__ == "__main__":
    main()
