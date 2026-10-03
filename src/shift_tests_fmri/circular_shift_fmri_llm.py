"""
circular_shift_fmri_llm.py — fMRI x LANGUAGE, every circular rotation.

Identical machinery to the vision sweep (`_fmri_shift.sweep`); only the targets differ.
The LLMs embed the Qwen-2.5-VL caption of each 4 s Season-7 clip, so on the clip4s grid
one window = one caption — no concatenation, which is the cleanest version of the
language comparison available in this project.

  H1  fMRI<->LLM mKNN increases with LLM size.
  H2  that increase is specific to d=0.

Scaling families on the clip4s grid:
    bloom      bloomz-560m .. bloomz-7b1        5 sizes
    openllama  open_llama_3b .. open_llama_13b  3 sizes
    llama      llama-13b only                   1 size -> NO within-family scaling

llama is therefore excluded from the scaling figures by default (`--families`), matching
`shift_tests/circular_shift_null.py`, which also drops llama-13b for this reason. It is
still runnable with `--families bloom openllama llama` if you want the cell itself.

Note the two families are NOT on a common capability scale — `_common.LLM_PERF` holds
measured 1-BPB, but this package deliberately uses ordinal size rank WITHIN family, as
`shift_tests` does, so no cross-family capability claim is implied.

Outputs -> <out-dir>/circular_shift_fmri_llm.npz

Usage:
    python circular_shift_fmri_llm.py
    python circular_shift_fmri_llm.py --excl 10 --out-dir outputs_vs_llm_excl10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _fmri_shift as FS_  # noqa: E402
from _fmri_shift import C, CS  # noqa: E402

OUT = _HERE / "outputs_vs_llm"
# Families with >=2 sizes on clip4s. llama ships only llama-13b there, so it cannot
# contribute a within-family slope and is off by default.
DEFAULT_FAMILIES = ["bloom", "openllama"]


def build_targets(families, stems_filter):
    """(key, group, rank, label, hf_path) for every requested LLM stem.

    Cell key is the STEM (globally unique). The family is carried in `group` rather than
    encoded in the key, because stems like `open_llama_3b` have no hyphen-delimited
    family prefix to parse back out.
    """
    out = []
    for fam in families:
        stems = [s for s in C.LLM.get(fam, [])
                 if stems_filter is None or s in stems_filter]
        for rank, stem in enumerate(stems):
            p = C.LLM_PARAMS.get(stem)
            lab = C.LLM_LABEL.get(stem, stem).split("_", 1)[-1].upper()
            out.append((stem, fam, rank,
                        f"{lab}\n{p/1000:.1f}B" if p else lab,
                        C.llm_path(FS_.GRID, stem)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES,
                    choices=list(C.LLM))
    ap.add_argument("--stems", nargs="+", default=None)
    ap.add_argument("--k-mknn", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--tail-min", type=int, default=FS_.TAIL_MIN)
    ap.add_argument("--excl", type=int, default=0,
                    help="Bar neighbours within +-N windows (N*4 s). See the vision "
                         "script — the exclusion decides the answer, so run both.")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--llm-repo", default=CS.DEFAULT_LLM_REPO)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args()

    FS_.run_self_tests()
    if args.self_test_only:
        return
    FS_.setup_env(args.hf_token, args.llm_repo)

    targets = build_targets(args.families, args.stems)
    print(f"  {len(targets)} LLM targets: {[t[0] for t in targets]}")
    FS_.sweep(targets, Path(args.out_dir) / "circular_shift_fmri_llm.npz",
              k=args.k_mknn, tail_min=args.tail_min, excl=args.excl,
              do_validate=not args.no_validate, modality="llm")


if __name__ == "__main__":
    main()
