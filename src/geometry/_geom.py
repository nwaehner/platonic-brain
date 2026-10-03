#!/usr/bin/env python
"""
_geom.py — shared loaders for the two geometry probes.

Both probes ask the same question in different coordinates: WHERE in a representation
does its alignment with fMRI live? The mKNN probe widens the neighbourhood (k); the CKA
probe deletes the strongest directions (r). Everything else is held at the project's
standard settings — the complete cyclic group C_n as the null, subjects paired by ID,
the layer pair re-maximised at every rotation.

WHICH GRID EACH GROUP USES, AND WHY THAT IS FINE. The vision and language embeddings are
published on the caption-native 4 s grid (clip4s, W = 2700) and meet fMRI pooled to 4 s.
The EEG embeddings have no 4 s version, so an EEG model meets fMRI pooled to ITS OWN
window length, exactly as compute_fmri_mknn_cyclic.py does (W = 2160 / 1800 / 1350 /
1080). The groups therefore sit on different grids — which does not matter here, because
every curve in the figure is a RATIO to its own k = 5 or r = 0 baseline. Absolute values
are never compared across groups.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))
sys.path.insert(0, str(_HERE.parent / "shift_tests"))
sys.path.insert(0, str(_HERE.parent / "shift_tests_fmri"))
sys.path.insert(0, str(_HERE.parent / "decodability_rebuild"))

import _common as C  # noqa: E402
import fmri_windows as FW  # noqa: E402
import _llm_grids as G  # noqa: E402

CLIP_WIN = 4          # the grid vision and LLM embeddings are published on
OUT = _HERE / "cache"
K_GRID = [5, 10, 20, 50, 100, 200]
R_GRID = [0, 1, 2, 3, 5, 10, 20]


def token():
    p = _HERE.parents[1] / "tokens" / "hf_token.txt"
    return p.read_text().strip() if p.exists() else None


def targets(group):
    """-> [(key, win_sec, loader)] for one model group. loader() -> (L, W, D)."""
    if group == "eeg":
        return [(f"{m}-{s}", C.EEG_WINDOWS[m]["win_sec"],
                 (lambda m=m, s=s: C.load_npz(C.EEG[m]["fname"](s))["embeddings"]))
                for m in C.EEG for s in C.EEG[m]["sizes"]]
    if group == "vision":
        return [(f"{a}-{vs}", CLIP_WIN,
                 (lambda a=a, vs=vs: C.load_npz(C.VISION[a]["path"](vs, "clip4s"))
                  ["embeddings"]))
                for a in C.VISION for vs in C.VISION[a]["sizes"]]
    return [(stem, CLIP_WIN, (lambda s=stem: G.load_llm(s, "clip4s")))
            for fam in ("bloom", "openllama") for stem in C.LLM[fam]]


def fmri_for(win_sec, _cache={}):
    """fMRI pooled to `win_sec`. -> (W, S, D) float32 and the subject ids."""
    if win_sec not in _cache:
        emb, subs = FW.fmri_embedding(win_sec, shift_sec=0.0)
        _cache[win_sec] = (np.ascontiguousarray(emb[0]), subs)   # (W, S, D)
        del emb
    return _cache[win_sec]


def pair(group, key, win_sec, loader):
    """Align one model with fMRI on a common W and subject list.

    -> model (L, W, D) float32, fmri (W, S, D) float32, subject ids.
    EEG carries a subject axis of its own and is matched to fMRI BY ID; vision and
    language do not, and every fMRI subject sees the same embedding.
    """
    fm, fsubs = fmri_for(win_sec)
    emb = np.asarray(loader())
    if group == "eeg":
        m = key.rsplit("-", 1)[0]
        z = C.load_npz(C.EEG[m]["fname"](key.rsplit("-", 1)[1]))
        esubs = [str(s) for s in z["subjects"]]
        keep = [(fsubs.index(s), esubs.index(s)) for s in fsubs if s in esubs]
        W = min(emb.shape[1], fm.shape[0])
        model = np.stack([emb[:, :W, ei, :] for _, ei in keep], axis=0)  # (S,L,W,D)
        return (model.astype(np.float32),
                fm[:W][:, [fi for fi, _ in keep]].astype(np.float32),
                [fsubs[fi] for fi, _ in keep])
    W = min(emb.shape[1], fm.shape[0])
    return emb[:, :W].astype(np.float32), fm[:W].astype(np.float32), fsubs
