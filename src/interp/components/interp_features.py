"""
interp_features.py — Component 1: the shared, library-grounded feature stack.

Builds and caches ONE interpretable descriptor set per grid (an EEG model name, or the
4 s 'clip4s' grid), replacing the hand-coded visual scalars + keyword lexicons of the old
`semantic_or_visual_features.py`. Three tiers, all from processing the real image/text:

  Tier 1  low-level visual (model-free):   motion · spatial-freq · luminance · contrast ·
          edge-density (classical, reused from compute_visual_features) + scikit-image
          GLCM/Haralick · LBP entropy · image entropy · Hasler–Süsstrunk colorfulness.
  Tier 2  high-level visual:               CLIP image embedding + CLIP zero-shot scores
          over a label bank that is NOT hand-curated (COCO/scene/action taxonomy +
          caption-mined nouns). Skipped with a warning if torch/transformers are absent.
  Tier 3  semantic:                        Empath validated lexical categories + a
          sentence/caption text embedding (CLIP text encoder, else TF-IDF fallback).

Cache → outputs/features/<grid>__features.npz with named feature matrices (`feat`,
`feat_names`, `feat_tier`), the dense CLIP image / text embeddings, and per-sentence
embeddings (for Component 4). Re-used by Components 2-5 via `load_features`.

Usage:
  python src/interp/interp_features.py --grid reve --smoke
  python src/interp/interp_features.py --grid clip4s
  python src/interp/interp_features.py --grid all
"""

from __future__ import annotations

# bootstrap: add interp root + all study subdirs to sys.path
import sys as _sys; from pathlib import Path as _Path
_INTERP = _Path(__file__).resolve().parent.parent
for _d in ([_INTERP] + [p for p in _INTERP.iterdir() if p.is_dir() and p.name[0] not in "._o"]):
    if str(_d) not in _sys.path: _sys.path.insert(0, str(_d))

import argparse
import re

import numpy as np

import _interp_common as IC
import _common as C

FRAME_SIZE = 128
_CLASSICAL = ["motion", "spatial_freq", "luminance", "contrast", "edge_density"]
FEAT_DIR = IC.OUT_DIR / "features"


def _classical_scalars(grid, frames_per, W):
    """(W, 5) low-level scalars. Reuse the cached compute_visual_features for full EEG
    grids; otherwise (clip4s, or a smoke subset of an uncached EEG grid) compute just the
    W windows on the fly so a smoke run never builds the whole grid."""
    cache = IC.OUT_DIR / f"visual_features__{grid}.npz"
    if grid in C.EEG and (cache.exists() or W >= IC.grid_n_windows(grid)):
        vis = IC.compute_visual_features(grid)
        return np.asarray(vis["scalars"][:W], dtype=np.float32), list(vis["scalar_names"])
    rows = []
    for frames in frames_per:
        _, scal = IC.window_visual_features(frames)
        rows.append([scal[k] for k in _CLASSICAL])
    return np.asarray(rows, dtype=np.float32), list(_CLASSICAL)


def _split_sentences(text):
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    return parts or ([text.strip()] if (text or "").strip() else [])


def compute_features(grid, max_windows=None, smoke=False, use_clip=True,
                     clip_model="openai/clip-vit-base-patch32"):
    win_sec = IC.grid_win_sec(grid)
    W = IC.grid_n_windows(grid)
    if max_windows:
        W = min(W, max_windows)
    starts = IC.canonical_starts(win_sec)[:W]
    print(f"[{grid}] {W} windows × {win_sec}s")

    IC.ensure_clips()
    frames_per = [IC.get_window_frames(float(t), win_sec, size=FRAME_SIZE) for t in starts]

    # ── Tier 1: low-level visual ──────────────────────────────────────────────
    classical, classical_names = _classical_scalars(grid, frames_per, W)
    sk_rows = [IC.skimage_window_features(f)[0] for f in frames_per]
    sk = np.asarray(sk_rows, dtype=np.float32)
    sk_names = IC._SKIMAGE_NAMES
    low = np.hstack([classical, sk])
    low_names = [f"low:{n}" for n in classical_names + sk_names]
    print(f"  tier-1 low-level: {low.shape[1]} features")

    # ── Tier 3 text (needed early for the label bank mining) ──────────────────
    texts = IC.grid_caption_texts(grid, W)

    # ── Tier 2: high-level visual (CLIP) ──────────────────────────────────────
    clip_img = np.zeros((W, 0), dtype=np.float32)
    highvis, highvis_names = np.zeros((W, 0), dtype=np.float32), []
    if use_clip and IC.clip_available():
        try:
            IC.get_clip(clip_model)
            clip_img = IC.clip_image_embed(frames_per)
            labels, _ = IC.build_label_bank(captions=texts, smoke=smoke)
            highvis = IC.clip_zeroshot(clip_img, labels)
            highvis_names = [f"clip:{l}" for l in labels]
            print(f"  tier-2 high-level: CLIP img {clip_img.shape[1]}d + "
                  f"{len(labels)} zero-shot labels")
        except Exception as e:                          # torch/transformers mismatch, OOM…
            clip_img = np.zeros((W, 0), dtype=np.float32)
            highvis, highvis_names = np.zeros((W, 0), dtype=np.float32), []
            print(f"  tier-2 high-level: SKIPPED (CLIP failed: {type(e).__name__}: {e})")
    else:
        print("  tier-2 high-level: SKIPPED (torch/transformers unavailable)")

    # ── Tier 3: semantic ──────────────────────────────────────────────────────
    emp, emp_names = IC.empath_features(texts)
    sem = emp
    sem_names = [f"sem:{n}" for n in emp_names]
    txt_emb, txt_backend = IC.text_embed(texts)
    print(f"  tier-3 semantic: {sem.shape[1]} empath cats + text-emb[{txt_backend}] "
          f"{txt_emb.shape[1]}d")

    # per-sentence embeddings for Component 4
    sent_text, sent_window = [], []
    for w, t in enumerate(texts):
        for s in _split_sentences(t):
            sent_text.append(s); sent_window.append(w)
    sent_emb, _ = IC.text_embed(sent_text) if sent_text else (np.zeros((0, 1)), "none")

    # ── named feature matrix (interpretable scalars only) ─────────────────────
    feat = np.hstack([low, highvis, sem])
    feat_names = low_names + highvis_names + sem_names
    feat_tier = (["low"] * low.shape[1] + ["highvis"] * highvis.shape[1]
                 + ["sem"] * sem.shape[1])
    print(f"  → {feat.shape[1]} named features total")

    return dict(
        grid=grid, win_sec=win_sec, W=W, starts=starts,
        feat=feat.astype(np.float32), feat_names=np.array(feat_names),
        feat_tier=np.array(feat_tier),
        low=low.astype(np.float32), low_names=np.array(low_names),
        highvis=highvis.astype(np.float32), highvis_names=np.array(highvis_names),
        sem=sem.astype(np.float32), sem_names=np.array(sem_names),
        clip_img=clip_img.astype(np.float32),
        txt_emb=txt_emb.astype(np.float32), txt_backend=txt_backend,
        sent_text=np.array(sent_text, dtype=object),
        sent_window=np.array(sent_window, dtype=np.int64),
        sent_emb=sent_emb.astype(np.float32))


def cache_path(grid, smoke=False):
    return FEAT_DIR / f"{grid}{'__smoke' if smoke else ''}__features.npz"


def load_features(grid, smoke=False):
    """Load a cached feature set as a dict, or None if not built yet."""
    p = cache_path(grid, smoke)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def build(grid, smoke=False, max_windows=None, use_clip=True, force=False,
          clip_model="openai/clip-vit-base-patch32"):
    p = cache_path(grid, smoke)
    if p.exists() and not force:
        print(f"[{grid}] cached → {p.name}")
        return load_features(grid, smoke)
    mw = max_windows or (96 if smoke else None)
    res = compute_features(grid, max_windows=mw, smoke=smoke, use_clip=use_clip,
                           clip_model=clip_model)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **res)
    print(f"  saved → outputs/features/{p.name}")
    return res


def main():
    ap = argparse.ArgumentParser(description="Component 1: shared feature stack.")
    ap.add_argument("--grid", nargs="*", default=["reve"],
                    help="EEG model name(s) and/or 'clip4s', or 'all'.")
    ap.add_argument("--smoke", action="store_true", help="few windows, tiny label bank.")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--no-clip", dest="use_clip", action="store_false")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)

    grids = list(C.EEG) + ["clip4s"] if args.grid == ["all"] else args.grid
    for g in grids:
        build(g, smoke=args.smoke, max_windows=args.max_windows,
              use_clip=args.use_clip, force=args.force, clip_model=args.clip_model)


if __name__ == "__main__":
    main()
