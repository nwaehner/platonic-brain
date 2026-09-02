"""
compute_intramodal.py — calibrated intramodal mKNN + CKA between consecutive EEG sizes.

For each family and each consecutive size pair (incl reve base↔large), and each subject,
compute the Aristotelian-calibrated mKNN and linear-CKA between the two sizes. The two
sizes share the same native window grid (same W, same subjects), so the geometry is
compared directly — no common-grid reconciliation. Both metrics are computed fresh in one
pass (via the shared _encode → nn + grams), so their K-permutation calibration is identical.

Output: data/paper-plots/_cache/intramodal_full.npz
  pair_keys (P,) of '<model>__<a>__<b>';  per key: <key>__mk_cal (S,), <key>__ck_cal (S,)

Usage:
  python src/paper_plots/compute_intramodal.py               # all 9 pairs
  python src/paper_plots/compute_intramodal.py --smoke        # femba+luna pairs (cached)
"""

from __future__ import annotations

import argparse
import gc
import os

import numpy as np

import _paper_common as P
from _paper_common import C, IC
import eeg_crossmodal_decodability as XM      # _encode, _aristotelian_mknn, _aristotelian_cka


def _load_layerwise(model, size):
    """Subject-resolved EEG embedding (L, W, S, D), left in its native float32 dtype —
    _encode upcasts only the small per-subject (L, W, D) slice, so peak memory stays low
    (the full float64 cast of an (L,W,S,D) tensor OOMs for luna-huge)."""
    try:
        return C.load_npz(C.EEG[model]["fname"](size))["embeddings"]
    except Exception as e:
        print(f"  [skip] {model}-{size}: {type(e).__name__}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Calibrated intramodal mKNN+CKA over size pairs.")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--K", type=int, default=100, help="permutations for calibration")
    ap.add_argument("--alpha", type=float, default=C.ALPHA_DEFAULT)
    ap.add_argument("--smoke", action="store_true", help="femba pairs (cached)")
    ap.add_argument("--models", default=None, help="comma list of EEG models")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoint, recompute all")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)
    P.ensure_dirs()

    models = (set(args.models.split(",")) if args.models
              else {"femba"} if args.smoke else set(C.EEG.keys()))
    pairs = [d for d in P.size_pairs() if d["model"] in models]

    # resumable checkpoint: reload done pairs, save after each (a kill loses ≤ one pair)
    payload, pair_keys = {}, []
    if P.INTRAMODAL_NPZ.exists() and not args.fresh:
        z = np.load(P.INTRAMODAL_NPZ, allow_pickle=True)
        for key in [str(k) for k in z["pair_keys"]]:
            payload[f"{key}__mk_cal"] = z[f"{key}__mk_cal"]
            payload[f"{key}__ck_cal"] = z[f"{key}__ck_cal"]
            pair_keys.append(key)
        print(f"resuming intramodal: {len(pair_keys)} pairs already done.", flush=True)

    def _save():
        tmp = P.INTRAMODAL_NPZ.with_suffix(".tmp.npz")      # must end .npz (np.savez appends it)
        np.savez(tmp, pair_keys=np.array(pair_keys), **payload)
        os.replace(tmp, P.INTRAMODAL_NPZ)

    for d in pairs:
        m, a, b = d["model"], d["a"], d["b"]
        key = f"{m}__{a}__{b}"
        if key in pair_keys:
            continue
        emb_a = _load_layerwise(m, a)
        emb_b = _load_layerwise(m, b)
        if emb_a is None or emb_b is None:
            continue
        S = min(emb_a.shape[2], emb_b.shape[2])
        mk, ck = [], []
        for s in range(S):
            nn_a, grams_a = XM._encode(emb_a[:, :, s, :], args.k)
            nn_b, grams_b = XM._encode(emb_b[:, :, s, :], args.k)
            seed = abs(hash((key, s))) & 0x7FFFFFFF
            _, mk_cal, _, _ = XM._aristotelian_mknn(nn_a, nn_b, args.K, args.alpha, seed)
            _, ck_cal, _, _ = XM._aristotelian_cka(grams_a, grams_b, args.K, args.alpha, seed + 1)
            mk.append(mk_cal); ck.append(ck_cal)
        payload[f"{key}__mk_cal"] = np.array(mk)
        payload[f"{key}__ck_cal"] = np.array(ck)
        pair_keys.append(key)
        print(f"  {key}: S={S}  mKNN_cal={np.mean(mk):.3f}±{np.std(mk):.3f}  "
              f"CKA_cal={np.mean(ck):.3f}±{np.std(ck):.3f}", flush=True)
        del emb_a, emb_b; gc.collect()
        _save()                                              # checkpoint after each pair

    if not pair_keys:
        raise SystemExit("no pairs computed — nothing to save")
    _save()
    print(f"\nSaved {P.INTRAMODAL_NPZ}  ({len(pair_keys)} pairs)")


if __name__ == "__main__":
    main()
