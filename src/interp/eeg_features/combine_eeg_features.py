"""
combine_eeg_features.py — average the per-subject NICE EEG features into per-grid,
subject-mean targets (matching the subject-mean EEG embeddings).

Reads outputs/eeg_features/raw/<subject>__<grid>__eegfeat.npz (from eeg_features.py, 6
subjects) → writes outputs/eeg_features/<grid>__eegfeat.npz with the across-subject mean.
luna shares femba's 5 s grid, so it is copied.

Usage:  python src/interp/combine_eeg_features.py
"""

from __future__ import annotations

# bootstrap: add interp root + all study subdirs to sys.path
import sys as _sys; from pathlib import Path as _Path
_INTERP = _Path(__file__).resolve().parent.parent
for _d in ([_INTERP] + [p for p in _INTERP.iterdir() if p.is_dir() and p.name[0] not in "._o"]):
    if str(_d) not in _sys.path: _sys.path.insert(0, str(_d))

import shutil

import numpy as np

import _interp_common as IC

RAW = IC.OUT_DIR / "eeg_features" / "raw"
OUT = IC.OUT_DIR / "eeg_features"
SUBJECTS = [f"sub-{i:04d}" for i in range(1, 7)]
GRIDS = ["femba", "neurolm", "steegformer", "reve"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for grid in GRIDS:
        mats, names = [], None
        for s in SUBJECTS:
            p = RAW / f"{s}__{grid}__eegfeat.npz"
            if not p.exists():
                print(f"  [warn] missing {p.name}")
                continue
            z = np.load(p, allow_pickle=True)
            mats.append(z["feat"]); names = z["feat_names"]
        if not mats:
            print(f"  [skip] {grid}: no subjects"); continue
        W = min(m.shape[0] for m in mats)
        M = np.mean([m[:W] for m in mats], axis=0).astype(np.float32)
        np.savez_compressed(OUT / f"{grid}__eegfeat.npz", feat=M, feat_names=names,
                            n_subjects=len(mats), grid=grid)
        print(f"  {grid}: {M.shape} from {len(mats)} subjects → {grid}__eegfeat.npz")
    femba = OUT / "femba__eegfeat.npz"
    if femba.exists():
        shutil.copy(femba, OUT / "luna__eegfeat.npz")
        print("  copied femba → luna (shared 5 s grid)")


if __name__ == "__main__":
    main()
