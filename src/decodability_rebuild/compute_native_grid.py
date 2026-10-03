#!/usr/bin/env python
"""
compute_native_grid.py — EEG x (VISION | LANGUAGE) with NO resampling: every pair on
the EEG model's own native window grid.

THE CONTRAST THIS EXISTS TO DRAW. compute_{mknn,cka}_cyclic_vision.py put every model on
one shared 10 s / 1080-bin grid, which means femba and luna are averaged 2:1 before any
neighbourhood or Gram is built, steegformer and neurolm unevenly, and reve not at all —
while the vision side is read once from the 5 s femba_luna grid and averaged 2:1 for
everybody. So reve is compared unsmoothed against a smoothed partner, and luna smoothed
against a smoothed partner. This script removes the resampling entirely:

    EEG model     native   vision file read            language file read        W
    femba          5 s     dinov2_*__femba_luna.npz    llms/femba/*.npz          2160
    luna           5 s     dinov2_*__femba_luna.npz    llms/luna/*.npz           2160
    steegformer    6 s     dinov2_*__steegformer.npz   llms/steegformer/*.npz    1800
    neurolm        8 s     dinov2_*__neurolm.npz       llms/neurolm/*.npz        1350
    reve          10 s     dinov2_*__reve.npz          llms/reve/*.npz           1080

Note the two modalities index their grids differently: the vision files are published per
WINDOW-GRID family, so femba and luna (both 5 s) share dinov2_*__femba_luna.npz, while the
LLM files are published per EEG MODEL, so femba and luna get their own directories even
though the grids coincide.

Each EEG model meets the vision model tiled at exactly its own window length, so the two
sides are registered instant-for-instant with no averaging on either side, and the cyclic
null runs over that grid's own W-1 rotations.

WHAT THIS COSTS, AND WHY THE SHARED-GRID VERSION STILL EXISTS. y is no longer in
comparable units across families: mKNN with k=5 out of 2160 windows is a tighter
neighbourhood than 5 out of 1080, and CKA's Gram is W x W. Within a family every model
shares one grid, so within-family contrasts are clean — which is why the per-family
figure is the honest way to read this, and the family-centred pooled figure is offered
only for comparison with the shared-grid original.

Everything else is unchanged from the shared-grid scripts: subject-mean EEG embeddings,
cosine kNN with k=5, linear CKA on feature-centred Frobenius-normalised Grams, the max
re-taken over all layer pairs independently at every rotation, and the complete cyclic
group C_n as the null with no far-tail restriction.

Saved -> cache/native_{mknn,cka}_<group>.npz    (group = a vision arch or an LLM family)
    <eeg>__<vis>__curves  (W,) float32   metric at every rotation, index 0 == d=0
    eeg_keys, eeg_family, vis_keys, W_of  (one W per EEG key)

Usage:
    python compute_native_grid.py --modality video    --group dinov2
    python compute_native_grid.py --modality language --group bloom
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))
sys.path.insert(0, str(_HERE.parent / "shift_tests"))

import _common as C  # noqa: E402
import fast_shift as FS  # noqa: E402
import _llm_grids as G  # noqa: E402
from compute_mknn_cyclic import CACHE, REPO_EEG, _tok  # noqa: E402
from compute_cka_cyclic_eeg import grams_fft, max_cka_all_shifts  # noqa: E402
from compute_fmri_cka_cyclic import self_test as cka_self_test  # noqa: E402


def load_eeg_native(key):
    """(L, W, D) subject-mean EEG embedding on its OWN grid — no to_common."""
    from huggingface_hub import hf_hub_download
    m, s = key.rsplit("-", 1)
    z = np.load(hf_hub_download(REPO_EEG, C.EEG[m]["fname"](s), repo_type="dataset",
                                token=_tok()), allow_pickle=True)
    emb = z["embeddings"]
    out = emb.mean(axis=2)                       # SUBJECTS AVERAGED, as everywhere else
    del emb
    return np.asarray(out, float)


def partner_spec(modality, group):
    """-> (member keys, grid_key(eeg_model), loader(member, grid) -> (L,W,D))."""
    if modality == "video":
        members = [f"{group}-{vs}" for vs in C.VISION[group]["sizes"]]
        return (members,
                lambda m: C.EEG[m]["family"],          # vision: per window-grid family
                lambda k, g: C.load_npz(
                    C.VISION[group]["path"](k.rsplit("-", 1)[1], g))["embeddings"])
    members = [s for s in C.LLM[group]]
    return (members,
            lambda m: m,                               # language: per EEG model
            lambda k, g: G.load_llm(k, g))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", default="video", choices=["video", "language"])
    ap.add_argument("--group", default="dinov2",
                    help="vision architecture (dinov2, videomae, videomae_ft, vjepa2) "
                         "or LLM family (bloom, openllama, llama)")
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    args = ap.parse_args()

    print("── self-tests (synthetic, no network) ──")
    if not (FS.self_test(verbose=False) and cka_self_test(verbose=False)):
        raise SystemExit("self-test FAILED — refusing to run on real data.")
    print("  fast_shift and cka_all_shifts PASSED")
    C.HF_TOKEN_CACHE = _tok()

    eeg_keys = [f"{m}-{s}" for m in C.EEG for s in C.EEG[m]["sizes"]]
    vis_keys, grid_of, load_partner = partner_spec(args.modality, args.group)
    # group the EEG models by the grid their partner file is published under
    by_grid = {}
    for k in eeg_keys:
        by_grid.setdefault(grid_of(k.rsplit("-", 1)[0]), []).append(k)

    mk, ck, W_of = {}, {}, {}
    for grid, keys in by_grid.items():
        t0 = time.time()
        print(f"\n=== grid {grid} : {', '.join(keys)} ===", flush=True)
        # vision side of this grid, loaded once
        Vk, Vg, Wv = {}, {}, None
        for vk in vis_keys:
            emb = np.asarray(load_partner(vk, grid), float)
            Wv = emb.shape[1] if Wv is None else min(Wv, emb.shape[1])
            Vk[vk], Vg[vk] = emb, None
        print(f"  [{args.modality}] {len(Vk)} models, W={Wv}", flush=True)

        for ek in keys:
            e = load_eeg_native(ek)
            W = min(e.shape[1], Wv)
            W_of[ek] = W
            e = e[:, :W]
            knn_e = C.precompute_knn_layers(e, args.k).astype(np.int32)
            Fe_m = FS.precompute_fft(knn_e, W)
            Fe_c = grams_fft(e, W)
            del e, knn_e
            for vk in vis_keys:
                v = Vk[vk][:, :W]
                Fv_m = FS.precompute_fft(
                    C.precompute_knn_layers(v, args.k).astype(np.int32), W)
                mk[f"{ek}__{vk}__curves"] = FS.max_mknn_all_shifts(
                    Fe_m, Fv_m, W, args.k).astype(np.float32)
                del Fv_m
                Fv_c = grams_fft(v, W)
                ck[f"{ek}__{vk}__curves"] = max_cka_all_shifts(
                    Fe_c, Fv_c, W).astype(np.float32)
                del Fv_c, v
            del Fe_m, Fe_c
            print(f"  [{ek:18}] W={W:5d}  x {len(vis_keys)} {args.group} models "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del Vk, Vg

    meta = dict(eeg_keys=np.array(eeg_keys), vis_keys=np.array(vis_keys),
                eeg_family=np.array([k.rsplit("-", 1)[0] for k in eeg_keys]),
                W_of=np.array([W_of[k] for k in eeg_keys]), arch=args.group,
                modality=args.modality, k_mknn=args.k)
    CACHE.mkdir(parents=True, exist_ok=True)
    for tag, store in (("mknn", mk), ("cka", ck)):
        out = CACHE / f"native_{tag}_{args.group}.npz"
        np.savez_compressed(out, **store, **meta)
        print(f"[done] saved {out.name} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
