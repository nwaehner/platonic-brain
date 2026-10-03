#!/usr/bin/env python
"""
compute_fmri_cka_partners.py — fMRI x vision / language linear CKA at every rotation.

The CKA twin of circular_shift_fmri_{vision,llm}.py, which sweep mKNN only. Same cells,
same clip4s 4 s grid (W = 2700), same five fMRI subjects matched by ID, same complete
cyclic null with every non-identity rotation, and the layer pair re-maximised at every
rotation. Only the metric changes.

Written in the format circular_shift_fmri_vision.npz already uses — the same `cells`,
`groups`, `ranks` and `labels` tables, copied verbatim from the mKNN file so the two
sweeps label and order their cells identically — which lets plot_fmri_scaling.load read
it with no changes.

Saved -> outputs_vs_vision/circular_shift_fmri_vision_cka.npz
         outputs_vs_llm/circular_shift_fmri_llm_cka.npz
    <key>__curve  (S, W) float32, index 0 == d = 0

Usage:
    python compute_fmri_cka_partners.py
    python compute_fmri_cka_partners.py --modality llm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "geometry"))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))
sys.path.insert(0, str(_HERE.parent / "decodability_rebuild"))

import _geom as GE  # noqa: E402
import _common as C  # noqa: E402
from compute_cka_cyclic_eeg import grams_fft, max_cka_all_shifts  # noqa: E402
from compute_fmri_cka_cyclic import norm_gram, diag_rows, self_test  # noqa: E402

SRC = {"vision": ("outputs_vs_vision", "circular_shift_fmri_vision"),
       "llm": ("outputs_vs_llm", "circular_shift_fmri_llm")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", nargs="+", default=["vision", "llm"],
                    choices=list(SRC))
    args = ap.parse_args()
    C.HF_TOKEN_CACHE = GE.token()

    print("── self-test ──")
    if not self_test(verbose=False):
        raise SystemExit("cka_all_shifts self-test FAILED.")
    print("  cka_all_shifts PASSED")

    for mod in args.modality:
        sub, stem = SRC[mod]
        ref = np.load(_HERE / sub / f"{stem}.npz", allow_pickle=True)
        # carry the cell table over verbatim, so both metrics label cells identically
        meta = {k: ref[k] for k in ("shifts", "subjects", "excl", "grid", "win_sec",
                                    "tail_min", "modality", "cells", "groups",
                                    "ranks", "labels") if k in ref.files}
        want = {str(c) for c in ref["cells"]}
        store = {}
        for key, win, loader in GE.targets("vision" if mod == "vision" else "llm"):
            if key not in want:
                continue
            t0 = time.time()
            model, fm, subs = GE.pair(mod, key, win, loader)
            W, S = model.shape[1], fm.shape[1]
            Fm = grams_fft(np.asarray(model, float), W)
            cur = np.zeros((S, W), np.float32)
            for s in range(S):
                G = norm_gram(C.l2(np.asarray(fm[:, s, :], np.float64)))
                Ff = np.fft.rfft(diag_rows(G), axis=1).astype(np.complex64)[None]
                del G
                cur[s] = max_cka_all_shifts(Ff, Fm, W)
                del Ff
            store[f"{key}__curve"] = cur
            del model, Fm
            print(f"  [{key:20}] W={W} S={S}  d0 mean={cur[:, 0].mean():.4f} "
                  f"null mean={cur[:, 1:].mean():.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        out = _HERE / sub / f"{stem}_cka.npz"
        np.savez_compressed(out, **meta, **store)
        print(f"[done] saved {out.name} ({out.stat().st_size/1e6:.1f} MB, "
              f"{len(store)} cells)\n")


if __name__ == "__main__":
    main()
