"""
circular_shift_fmri_vision.py — fMRI x VISION, every circular rotation.

The `shift_tests/` question with the brain side held FIXED. There the EEG model scaled;
fMRI has no "sizes", so the VISION model scales instead:

  H1  fMRI<->vision mKNN increases with vision model size.
  H2  that increase is specific to d=0. Under circular rotation the scaling slope should
      collapse. If it survives rotation, "bigger models align better" is a shared temporal
      geometry (both signals smooth in time), not shared content.

H2 is the point — H1 alone is not evidence, because the null can produce a size trend on
its own. It did, in fact, until temporal exclusion was applied: see `--excl` and the
module docstring of `_fmri_shift.py`.

Grid: clip4s (4 s, W=2700), one window per Season-7 clip. Stimulus-native, so nothing
inherits an EEG model's arbitrary window length; all 12 vision variants exist on it.

Outputs -> <out-dir>/circular_shift_fmri_vision.npz  (layout in `_fmri_shift.sweep`)

Usage:
    python circular_shift_fmri_vision.py --arch dinov2            # smoke, 4 cells
    python circular_shift_fmri_vision.py                          # all 12
    python circular_shift_fmri_vision.py --excl 10 \
        --out-dir outputs_vs_vision_excl10                        # +-40 s exclusion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _fmri_shift as FS_  # noqa: E402
from _fmri_shift import C, CS  # noqa: E402

OUT = _HERE / "outputs_vs_vision"
ARCHS = ["dinov2", "videomae", "videomae_ft", "vjepa2"]


def build_targets(archs, sizes_filter):
    """(key, group, rank, label, hf_path) for every requested vision variant."""
    out = []
    for arch in archs:
        sizes = [s for s in C.VISION[arch]["sizes"]
                 if sizes_filter is None or s in sizes_filter]
        for rank, size in enumerate(sizes):
            p = C.VISION_PARAMS.get((arch, size))
            out.append((f"{arch}-{size}", arch, rank,
                        f"{size.upper()}\n{p}M" if p else size.upper(),
                        C.VISION[arch]["path"](size, FS_.GRID)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="+", default=ARCHS, choices=ARCHS)
    ap.add_argument("--sizes", nargs="+", default=None)
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--tail-min", type=int, default=FS_.TAIL_MIN)
    ap.add_argument("--excl", type=int, default=0,
                    help="Bar neighbours within +-N windows (N*4 s). Without it, "
                         "temporal adjacency dominates mKNN and — being translation-"
                         "invariant — survives rotation, so the null tracks d=0.")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--vision-repo", default=CS.DEFAULT_LLM_REPO)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    FS_.run_self_tests()
    if args.self_test_only:
        return
    FS_.setup_env(args.hf_token, args.vision_repo)

    targets = build_targets(args.arch, args.sizes)
    print(f"  {len(targets)} vision targets")
    FS_.sweep(targets, Path(args.out_dir) / "circular_shift_fmri_vision.npz",
              k=args.k_mknn, tail_min=args.tail_min, excl=args.excl,
              do_validate=not args.no_validate, modality="vision")


if __name__ == "__main__":
    main()
