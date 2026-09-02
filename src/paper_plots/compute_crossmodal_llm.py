"""
compute_crossmodal_llm.py — calibrated mKNN + CKA for every EEG variant × LLM stem.

This is the missing computation the paper needs: the shipped summary_crossmodal.npz has
CKA all-NaN (it was written from the raw --from-hf gridsearch path). Here we run the real
Aristotelian calibration for BOTH metrics, reusing the exact functions in
eeg_crossmodal_decodability.py (_encode, _aristotelian_mknn, _aristotelian_cka). LLM caption
embeddings are already extracted on each EEG model's native window grid (llms/<grid>/…), so
no common-grid reconciliation is needed — partner W matches EEG W.

Output: data/paper-plots/_cache/crossmodal_llm_calibrated.npz
  one aligned array per field over N = (#variants × #LLM stems) rows:
  eeg, eeg_model, eeg_size, family, partner, partner_family,
  mk_obs, mk_cal, mk_tau, mk_p, ck_obs, ck_cal, ck_tau, ck_p

Usage:
  python src/paper_plots/compute_crossmodal_llm.py               # 14 variants × 9 LLM
  python src/paper_plots/compute_crossmodal_llm.py --smoke       # femba+luna, 2 LLM (cached)
  python src/paper_plots/compute_crossmodal_llm.py --models femba,luna --K 100
"""

from __future__ import annotations

import argparse

import numpy as np

import _paper_common as P
from _paper_common import C, IC
import eeg_crossmodal_decodability as XM          # reuse its calibrated mKNN + CKA + _encode

import gc
import glob
import os
from pathlib import Path

_SMOKE_LLM = ["bloomz-560m", "bloomz-1b7"]


def _evict_llm(grid, stem):
    """Delete the cached LLM-caption blob for (grid, stem) to bound disk on a full disk.
    LLM captions live in the triniborrell repo, separate from the EEG repo (nitrox639),
    so evicting them never touches the EEG embeddings we still need. Files are re-downloadable."""
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    pat = str(hub / "datasets--triniborrell--platonic-embeddings" / "snapshots" / "*"
              / "llms" / grid / f"{stem}_layerwise.npz")
    for link in glob.glob(pat):
        try:
            blob = os.path.realpath(link)
            if os.path.isfile(blob):
                os.remove(blob)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Calibrated cross-modal mKNN+CKA, EEG × LLM.")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT, help="mKNN neighbours (5)")
    ap.add_argument("--K", type=int, default=100, help="permutations for calibration")
    ap.add_argument("--alpha", type=float, default=C.ALPHA_DEFAULT)
    ap.add_argument("--smoke", action="store_true", help="femba+luna × 2 LLM (cached)")
    ap.add_argument("--models", default=None, help="comma list of EEG models")
    ap.add_argument("--keep-cache", action="store_true",
                    help="do NOT evict each LLM-caption blob after use (default: evict, to "
                         "bound disk — each is downloaded once, used for all sizes, then removed)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="succeed even if some (variant, LLM) pairs are still missing "
                         "(default: exit non-zero so an orchestrator keeps resuming)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and recompute from scratch")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    P.ensure_dirs()

    models = (args.models.split(",") if args.models
              else ["femba", "luna"] if args.smoke else list(C.EEG.keys()))
    stems = _SMOKE_LLM if args.smoke else P.LLM_STEMS

    # ── resumable checkpoint: reload any partial artifact, skip already-done pairs ──────
    rows, done = [], set()
    if P.CROSSMODAL_NPZ.exists() and not args.fresh:
        z = np.load(P.CROSSMODAL_NPZ, allow_pickle=True)
        for i in range(len(z["eeg"])):
            r = {f: (str(z[f][i]) if z[f].dtype.kind in "US" else float(z[f][i])) for f in z.files}
            rows.append(r); done.add((r["eeg"], r["partner"]))
        print(f"resuming: {len(rows)} pairs already checkpointed; continuing.", flush=True)

    def _save():
        tmp = P.CROSSMODAL_NPZ.with_suffix(".tmp.npz")      # must end .npz (np.savez appends it)
        np.savez(tmp, **{f: np.array([r[f] for r in rows]) for f in rows[0]})
        os.replace(tmp, P.CROSSMODAL_NPZ)                    # atomic: never a half-written cache

    def _encode_model(em):
        enc = {}
        for esz in C.EEG[em]["sizes"]:
            emb_e = IC.load_eeg_emb(em, esz)                 # (L, W, D) subject-mean
            if emb_e is None:
                print(f"  [skip] {em}-{esz}: embedding unavailable"); continue
            emb_e = emb_e.astype(np.float64)
            nn_e, grams_e = XM._encode(emb_e, args.k)
            enc[esz] = dict(emb=emb_e, nn=nn_e, grams=grams_e, W=emb_e.shape[1])
        return enc

    # grid-outer / stem-inner: encode a model's sizes once (EEG cached), then per LLM stem
    # download the caption embedding a single time, score all sizes, checkpoint, and evict —
    # so peak LLM-cache footprint is ~one file AND a kill loses at most one stem of work.
    skipped = []
    for em in models:
        fam = C.EEG[em]["family"]
        enc = None                                           # built lazily, only if work remains
        for stem in stems:
            todo = [s for s in C.EEG[em]["sizes"] if (f"{em}-{s}", stem) not in done]
            if not todo:
                continue                                     # fully cached → no download
            if enc is None:
                enc = _encode_model(em)
            todo = [s for s in todo if s in enc]
            if not todo:
                continue
            emb_p0 = IC.load_llm_grid(em, stem)              # captions on em's grid (once)
            if emb_p0 is None:
                print(f"    [skip] {em} × {stem}: no LLM embedding")
                skipped += [f"{em}-{s} × {stem}" for s in todo]
                continue
            nn_p_full, grams_p_full = XM._encode(emb_p0.astype(np.float64), args.k)

            for esz in todo:
                e = enc[esz]
                eeg_key = f"{em}-{esz}"
                W, Wp = e["W"], emb_p0.shape[1]
                if Wp != W:                                  # defensive truncation to common W
                    w = min(W, Wp)
                    nn_use, grams_use = e["nn"][:, :w], [XM._norm_gram(C.l2(e["emb"][:, :w][l]))
                                                         for l in range(e["emb"].shape[0])]
                    nn_p, grams_p = XM._encode(emb_p0[:, :w].astype(np.float64), args.k)
                else:
                    nn_use, grams_use, nn_p, grams_p = e["nn"], e["grams"], nn_p_full, grams_p_full

                seed_mk = abs(hash((eeg_key, stem, "mk"))) & 0x7FFFFFFF
                seed_ck = abs(hash((eeg_key, stem, "ck"))) & 0x7FFFFFFF
                mk_obs, mk_cal, mk_tau, mk_p = XM._aristotelian_mknn(
                    nn_use, nn_p, args.K, args.alpha, seed_mk)
                ck_obs, ck_cal, ck_tau, ck_p = XM._aristotelian_cka(
                    grams_use, grams_p, args.K, args.alpha, seed_ck)

                rows.append(dict(
                    eeg=eeg_key, eeg_model=em, eeg_size=esz, family=fam,
                    partner=stem, partner_family=C.LLM_FAMILY_OF.get(stem, stem.split("-")[0]),
                    mk_obs=mk_obs, mk_cal=mk_cal, mk_tau=mk_tau, mk_p=mk_p,
                    ck_obs=ck_obs, ck_cal=ck_cal, ck_tau=ck_tau, ck_p=ck_p))
                done.add((eeg_key, stem))
                print(f"  {eeg_key} × {stem}: mKNN={mk_obs:.3f}(cal={mk_cal:.3f})  "
                      f"CKA={ck_obs:.3f}(cal={ck_cal:.3f})", flush=True)

            del emb_p0, nn_p_full, grams_p_full
            if not args.keep_cache:
                _evict_llm(em, stem)
            gc.collect()
            _save()                                          # checkpoint after every stem
        if enc:
            del enc
        gc.collect()

    if not rows:
        raise SystemExit("no pairs computed — nothing to save")
    _save()
    expected = sum(len(C.EEG[m]["sizes"]) for m in models) * len(stems)
    ck = np.array([r["ck_cal"] for r in rows])
    print(f"\nSaved {P.CROSSMODAL_NPZ}  ({len(rows)}/{expected} pairs; "
          f"ck_cal finite={np.isfinite(ck).mean():.0%})")
    if len(rows) < expected and not args.allow_partial:
        raise SystemExit(
            f"partial: {len(rows)}/{expected} pairs — checkpoint is safe; re-run to resume "
            f"the rest (skipped now: {skipped[:3]}…). Pass --allow-partial to accept as-is.")


if __name__ == "__main__":
    main()
