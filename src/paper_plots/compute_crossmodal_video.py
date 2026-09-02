"""
compute_crossmodal_video.py — calibrated mKNN + CKA for every EEG variant × video model.

EEG→Video analogue of compute_crossmodal_llm.py. The shipped summary_crossmodal.npz has vision
CKA all-NaN (raw gridsearch mKNN only), so we recompute both metrics with the real Aristotelian
calibration, reusing eeg_crossmodal_decodability's _encode / _aristotelian_{mknn,cka}. Vision
embeddings are extracted on each EEG family's grid (vision/<arch>/<arch>_<size>__<family>.npz),
so partner W matches EEG W — no common-grid reconciliation.

Partners: 12 video variants from C.VISION (DINOv2 ×4, VideoMAE ×2, VideoMAE-K400 ×3, V-JEPA2 ×3).
14 EEG × 12 = 168 pairs, K=100.

Resumable (atomic checkpoint after each partner; --fresh to restart) and disk-safe: each vision
blob is downloaded once per EEG family, scored against all that model's sizes, then evicted (it is
re-downloadable). Output: data/paper-plots/_cache/crossmodal_video_calibrated.npz.

Usage:
  HF_HOME=$PWD/.hf_cache python src/paper_plots/compute_crossmodal_video.py --K 100
  … --smoke      (femba+luna × dinov2-small,vjepa2-large)
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
from pathlib import Path

import numpy as np

import _paper_common as P
from _paper_common import C, IC
import eeg_crossmodal_decodability as XM

_SMOKE_VIDEO = [("dinov2", "large"), ("vjepa2", "large")]


def video_variants(smoke=False):
    if smoke:
        return _SMOKE_VIDEO
    return [(a, s) for a in C.VISION for s in C.VISION[a]["sizes"]]


def _evict(relpath):
    """Delete the cached blob for one embedding file (any dataset snapshot). Safe: targets a single
    file by its repo-relative path, never the whole repo, so EEG embeddings are untouched."""
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    for link in glob.glob(str(hub / "datasets--*" / "snapshots" / "*" / relpath)):
        try:
            blob = os.path.realpath(link)
            if os.path.isfile(blob):
                os.remove(blob)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Calibrated cross-modal mKNN+CKA, EEG × video.")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=C.ALPHA_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default=None, help="comma list of EEG models")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    P.ensure_dirs()

    models = (args.models.split(",") if args.models
              else ["femba", "luna"] if args.smoke else list(C.EEG.keys()))
    partners = video_variants(args.smoke)                    # list of (arch, size)

    rows, done = [], set()
    if P.CROSSMODAL_VIDEO_NPZ.exists() and not args.fresh:
        z = np.load(P.CROSSMODAL_VIDEO_NPZ, allow_pickle=True)
        for i in range(len(z["eeg"])):
            r = {f: (str(z[f][i]) if z[f].dtype.kind in "US" else float(z[f][i])) for f in z.files}
            rows.append(r); done.add((r["eeg"], r["partner"]))
        print(f"resuming: {len(rows)} pairs already checkpointed.", flush=True)

    def _save():
        tmp = P.CROSSMODAL_VIDEO_NPZ.with_suffix(".tmp.npz")
        np.savez(tmp, **{f: np.array([r[f] for r in rows]) for f in rows[0]})
        os.replace(tmp, P.CROSSMODAL_VIDEO_NPZ)

    def _encode_model(em):
        enc = {}
        for esz in C.EEG[em]["sizes"]:
            emb = IC.load_eeg_emb(em, esz)
            if emb is None:
                print(f"  [skip] {em}-{esz}: no embedding"); continue
            emb = emb.astype(np.float64)
            nn, grams = XM._encode(emb, args.k)
            enc[esz] = dict(emb=emb, nn=nn, grams=grams, W=emb.shape[1])
        return enc

    skipped = []
    for em in models:
        fam = C.EEG[em]["family"]
        enc = None
        for (arch, vsz) in partners:
            plabel = f"{arch}-{vsz}"
            todo = [s for s in C.EEG[em]["sizes"] if (f"{em}-{s}", plabel) not in done]
            if not todo:
                continue
            if enc is None:
                enc = _encode_model(em)
            todo = [s for s in todo if s in enc]
            if not todo:
                continue
            emb_p0 = IC.load_vision_grid(arch, vsz, fam)
            if emb_p0 is None:
                print(f"    [skip] {em} × {plabel}: no vision embedding")
                skipped += [f"{em}-{s} × {plabel}" for s in todo]
                continue
            nn_p_full, grams_p_full = XM._encode(emb_p0.astype(np.float64), args.k)

            for esz in todo:
                e = enc[esz]
                eeg_key = f"{em}-{esz}"
                W, Wp = e["W"], emb_p0.shape[1]
                if Wp != W:
                    w = min(W, Wp)
                    nn_use = e["nn"][:, :w]
                    grams_use = [XM._norm_gram(C.l2(e["emb"][:, :w][l])) for l in range(e["emb"].shape[0])]
                    nn_p, grams_p = XM._encode(emb_p0[:, :w].astype(np.float64), args.k)
                else:
                    nn_use, grams_use, nn_p, grams_p = e["nn"], e["grams"], nn_p_full, grams_p_full
                sm_ = abs(hash((eeg_key, plabel, "mk"))) & 0x7FFFFFFF
                sc_ = abs(hash((eeg_key, plabel, "ck"))) & 0x7FFFFFFF
                mk_obs, mk_cal, mk_tau, mk_p = XM._aristotelian_mknn(nn_use, nn_p, args.K, args.alpha, sm_)
                ck_obs, ck_cal, ck_tau, ck_p = XM._aristotelian_cka(grams_use, grams_p, args.K, args.alpha, sc_)
                rows.append(dict(eeg=eeg_key, eeg_model=em, eeg_size=esz, family=fam,
                                 partner=plabel, partner_family=arch,
                                 mk_obs=mk_obs, mk_cal=mk_cal, mk_tau=mk_tau, mk_p=mk_p,
                                 ck_obs=ck_obs, ck_cal=ck_cal, ck_tau=ck_tau, ck_p=ck_p))
                done.add((eeg_key, plabel))
                print(f"  {eeg_key} × {plabel}: mKNN={mk_obs:.3f}(cal={mk_cal:.3f})  "
                      f"CKA={ck_obs:.3f}(cal={ck_cal:.3f})", flush=True)

            del emb_p0, nn_p_full, grams_p_full
            if not args.keep_cache:
                _evict(C.VISION[arch]["path"](vsz, fam))
            gc.collect()
            _save()
        if enc:
            del enc
        gc.collect()

    if not rows:
        raise SystemExit("no pairs computed — nothing to save")
    _save()
    expected = sum(len(C.EEG[m]["sizes"]) for m in models) * len(partners)
    ck = np.array([r["ck_cal"] for r in rows])
    print(f"\nSaved {P.CROSSMODAL_VIDEO_NPZ}  ({len(rows)}/{expected} pairs; "
          f"ck_cal finite={np.isfinite(ck).mean():.0%})")
    if len(rows) < expected and not args.allow_partial:
        raise SystemExit(f"partial: {len(rows)}/{expected} — re-run to resume (skipped {skipped[:3]}…).")


if __name__ == "__main__":
    main()
