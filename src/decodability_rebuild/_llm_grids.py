#!/usr/bin/env python
"""
_llm_grids.py — which LLM caption embeddings exist, and on which window grid to read them.

Shared by compute_mknn_cyclic_llm.py and compute_cka_cyclic_llm.py so the two metrics see
exactly the same model list and the same grid, the way the vision pair shares
compute_mknn_cyclic_vision.vision_keys().

Keys here are the bare LLM stems ("bloomz-560m", "open_llama_3b", ...); the family comes
from C.LLM_FAMILY_OF, not from splitting the string — "open_llama_3b" has no separator a
split could use.

WHICH GRID, AND WHY IT MATTERS — the LLM analogue of the note in
compute_mknn_cyclic_vision.py. The caption embeddings are published once per EEG WINDOW
GRID (llms/femba, llms/luna, llms/neurolm, llms/reve, llms/steegformer, plus a 4 s
llms/clip4s): the same captions pooled six ways. Everything here is read from ONE grid —
femba, 5 s, the finest of the EEG five — and downsampled once to the shared 10 s grid, so
each LLM gets exactly one common-grid representation, as each EEG and vision model does.

Pairing each EEG model with the LLM file on its own family's grid would instead give a
single LLM five different representations, and the difference between them would enter y
as a family effect on the LANGUAGE side — precisely the confound the family-centring on
the EEG side removes.

WHICH MODELS. C.LLM lists llama-13b/30b/65b, but only llama-13b was ever extracted; the
other two have no file in either embeddings repo. `available()` asks the repo rather than
assuming, so a later extraction is picked up with no edit here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

import _common as C  # noqa: E402

# win_sec of each published LLM grid. The five EEG names come from C.EEG_WINDOWS;
# clip4s is the caption-native 4 s tiling and is not an EEG family.
GRID_WIN = {m: C.EEG_WINDOWS[m]["win_sec"] for m in C.EEG_WINDOWS}
GRID_WIN["clip4s"] = 4

DEFAULT_GRID = "femba"
FAMILIES = ["bloom", "openllama", "llama"]
DISPLAY = {"bloom": "BLOOMZ", "openllama": "OpenLLaMA", "llama": "LLaMA"}


def all_stems():
    """Every LLM stem in the registry, families in FAMILIES order."""
    return [s for fam in FAMILIES for s in C.LLM[fam]]


def available(grid=DEFAULT_GRID, token=None):
    """The subset of all_stems() that actually has a file on `grid`. One repo listing."""
    from huggingface_hub import list_repo_files
    have = set(list_repo_files(C.LLM_REPO, repo_type="dataset", token=token))
    return [s for s in all_stems() if C.llm_path(grid, s) in have]


def load_llm(stem, grid=DEFAULT_GRID):
    """(L, W, D) raw caption embedding for `stem` on `grid`."""
    return C.load_npz(C.llm_path(grid, stem))["embeddings"]
