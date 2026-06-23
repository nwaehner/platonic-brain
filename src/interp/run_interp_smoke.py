"""
run_interp_smoke.py — one-command smoke test for the interp redesign (Components 1-5).

Builds a tiny feature cache (few windows, small label bank) for the reve + neurolm + clip4s
grids, then runs every study in --smoke mode, wrapping each in try/except so one failure does
not abort the rest. Lets you eyeball the outputs and flag problems before the full sweep.

Run with the env that has torch (CLIP) + skimage + empath + statsmodels, e.g.:
  /Users/.../miniconda3/envs/platonic-brain/bin/python src/interp/run_interp_smoke.py
Outputs land under src/interp/outputs/{features,enrichment,probing,caption_sim}/.
"""

from __future__ import annotations

import argparse
import traceback

import _interp_common as IC
import interp_features as F
import neighbour_enrichment as NE
import feature_linear_probing as P
import caption_similarity as CS
import vis_lang_probing as VL
import _common as C


def _step(name, fn):
    print(f"\n{'=' * 70}\n# {name}\n{'=' * 70}")
    try:
        fn()
        print(f"[OK] {name}")
        return True
    except Exception:
        print(f"[FAIL] {name}\n{traceback.format_exc()}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Smoke test for interp Components 1-5.")
    ap.add_argument("--no-clip", dest="use_clip", action="store_false")
    ap.add_argument("--max-windows", type=int, default=96)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    print("CLIP available:", IC.clip_available(), "| use_clip:", args.use_clip)

    grids = ["reve", "neurolm", "clip4s"]
    results = {}

    def build_feats():
        for g in grids:
            F.build(g, smoke=True, max_windows=args.max_windows, use_clip=args.use_clip)
    results["Comp1 features"] = _step("Comp 1 — feature stacks", build_feats)

    def comp2():
        for kind in NE.REGIMES:
            print(f"\n--- {kind} ---")
            out = NE.run_regime(kind, True, k=10, R=50)
            NE.save_regime(kind, out)
    results["Comp2 enrichment"] = _step("Comp 2 — neighbourhood variance ratio (6 regimes)", comp2)

    def comp3():
        ev = P._eeg_variants(True)
        vv = P._vision_variants(["dinov2"], True)
        df = P.build_table(ev, vv, True, k=C.K_MKNN_DEFAULT, folds=3, max_per_tier=4)
        if df.empty:
            raise RuntimeError("empty probe table")
        IC.save_npz("probing/feature_probing__smoke.npz",
                    **{c: df[c].to_numpy() for c in df.columns})
        P.plot_fit(df, "cross_mknn", "Probe R² vs cross mKNN (3a, smoke)",
                   "probing/probe_cross__a_fit__smoke.png")
        P.plot_ancova(df, "cross_mknn", "Family ANCOVA (3b, smoke)",
                      "probing/probe_cross__b_ancova__smoke.png")
        if df["intra_mknn"].notna().any():
            P.plot_fit(df, "intra_mknn", "Probe R² vs intramodal mKNN (3a-intra, smoke)",
                       "probing/probe_intra__a_fit__smoke.png")
    results["Comp3 probing"] = _step("Comp 3 — feature linear-probing (EEG↔vision)", comp3)

    def comp4():
        for regime in ["eeg_only", "intersection_eeg_vision", "intersection_llm_vision"]:
            print(f"\n--- {regime} ---")
            results_r = CS.run_regime(regime, True, k=10, R=50)
            CS.save_and_plot(regime, results_r)
    results["Comp4 caption-var"] = _step("Comp 4 — caption variance ratio", comp4)

    def comp5():
        df = VL.build_table(VL._vision_variants(["dinov2", "vjepa2"], True),
                            VL._llms(True), True, k=C.K_MKNN_DEFAULT, folds=3, max_per_tier=4)
        if df.empty:
            raise RuntimeError("empty vis-lang table")
        IC.save_npz("probing/vis_lang_probing__smoke.npz",
                    **{c: df[c].to_numpy() for c in df.columns})
        vfam = [a for a in P.FAM_COLORS if a in C.VISION]
        VL.P.plot_fit(df, "cross_mknn", "Vision↔language mKNN (5·3a, smoke)",
                      "probing/vislang_cross__a_fit__smoke.png", row_col="llm", fam_pool=vfam)
    results["Comp5 vis-lang"] = _step("Comp 5 — vision↔language probing (4 s)", comp5)

    print(f"\n{'=' * 70}\nSMOKE SUMMARY")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()
