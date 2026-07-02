"""
gridsearch_features.py — clean, interpretable scalar feature stack for the
feature-decodability grid search (`feature_gridsearch.py`).

Motivation: the Component-1/3 probe targets (CLIP zero-shot label scores, Empath
categories) are spiky, high-dimensional and indirect → block-CV Ridge R² collapsed to
hundreds-negative and tracked alignment *backwards*. This module builds a small set of
DIRECT, low-variance interpretable SCALARS per window, in two tiers:

  low  (model-free visual, one frame pass per window):
       luminance · rms_contrast · edge_density · spatial_freq · motion (optical flow) ·
       glcm_{contrast,homogeneity,energy,correlation} · lbp_entropy · tex_entropy ·
       colorfulness · hue_mean · sat_mean · frame_change · shot_cut_rate
  sem  (caption-derived, lexicon-grounded):
       word_count · mean_word_len · sentence_count · type_token_ratio ·
       valence · arousal · dominance (NRC-VAD) · concreteness (Brysbaert) ·
       sentiment (VADER, fallback to NRC valence) · people_rate · action_rate · animate_rate

Vendored lexicons live in `lexicons/` (committed): NRC-VAD-Lexicon (19 971 words, V/A/D in
[0,1]) and Brysbaert concreteness (39 954 words, 1–5). Both are looked up with a light
singularisation fallback; windows with no lexicon hit get NaN (column-mean imputed at probe
time). Reuses the model-free visual helpers in `_interp_common`.

Cache → outputs/gridsearch/features/<grid>[__smoke]__features.npz with `feat` (W,F),
`feat_names`, `feat_tier`. Re-used by `feature_gridsearch.py`.

Usage:
  python src/interp/gridsearch_features.py --grid reve --smoke
  python src/interp/gridsearch_features.py --grid all
"""

from __future__ import annotations

# bootstrap: add interp root + all study subdirs to sys.path
import sys as _sys; from pathlib import Path as _Path
_INTERP = _Path(__file__).resolve().parent.parent
for _d in ([_INTERP] + [p for p in _INTERP.iterdir() if p.is_dir() and p.name[0] not in "._o"]):
    if str(_d) not in _sys.path: _sys.path.insert(0, str(_d))

import argparse
import functools
import re
from pathlib import Path

import numpy as np

import _interp_common as IC
import _common as C

FRAME_SIZE = 96                                          # low-level scalars need no high res
LEX_DIR = Path(__file__).resolve().parent.parent / "lexicons"
FEAT_DIR = IC.OUT_DIR / "gridsearch" / "features"

_WORD_RE = re.compile(r"[a-z]+")

# Keyword banks reused from the model-free lexical attributes in _interp_common.
_PEOPLE = ["person", "people", "man", "men", "woman", "women", "boy", "girl", "child",
           "children", "guy", "lady", "crowd", "he", "she", "they", "human", "someone"]
_ACTION = ["run", "running", "walk", "walking", "fight", "fighting", "jump", "jumping",
           "move", "moving", "drive", "driving", "dance", "dancing", "throw", "chase",
           "fall", "falling", "grab", "push", "ride", "riding", "climb", "swim"]
_ANIMATE = _PEOPLE + ["dog", "cat", "horse", "bird", "animal", "animals", "fish", "cow",
                      "sheep", "bear", "lion", "dragon", "creature", "monster"]


# ── vendored lexicons ─────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _nrc_vad():
    """word → (valence, arousal, dominance) in [0,1]. Tab-separated."""
    d = {}
    p = LEX_DIR / "nrc_vad.txt"
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            w, v, a, dom = parts
            try:
                d[w] = (float(v), float(a), float(dom))
            except ValueError:
                pass
    return d


@functools.lru_cache(maxsize=1)
def _concreteness():
    """word → concreteness (1–5). Comma-separated word,score."""
    d = {}
    p = LEX_DIR / "concreteness.csv"
    for line in p.read_text(encoding="utf-8").splitlines():
        w, _, s = line.partition(",")
        try:
            d[w] = float(s)
        except ValueError:
            pass
    return d


@functools.lru_cache(maxsize=1)
def _vader():
    """VADER analyser or None (graceful fallback to NRC valence-based polarity)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def _lookup(table, word):
    """Direct lookup with a light singularisation / de-inflection fallback."""
    if word in table:
        return table[word]
    for suf in ("s", "es", "ed", "ing"):
        if word.endswith(suf):
            stem = word[: -len(suf)]
            if stem in table:
                return table[stem]
    return None


# ── low-level visual (one frame pass per window) ──────────────────────────────────
_LOW_BASE = ["motion", "spatial_freq", "luminance", "contrast", "edge_density"]


def _low_window(frames):
    """Dict of low-level scalars for one window's RGB frames (NaNs if empty)."""
    import cv2
    base_vec, base = IC.window_visual_features(frames)             # 5 classical scalars
    sk_vec, sk_names = IC.skimage_window_features(frames)          # 7 texture/colour
    out = {k: float(base[k]) for k in _LOW_BASE}
    out.update({n: float(v) for n, v in zip(sk_names, sk_vec)})
    if not frames:
        for k in ("hue_mean", "sat_mean", "frame_change", "shot_cut_rate"):
            out[k] = np.nan
        return out
    hsv = [cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in frames]
    out["hue_mean"] = float(np.mean([h[..., 0].mean() for h in hsv]) / 180.0)
    out["sat_mean"] = float(np.mean([h[..., 1].mean() for h in hsv]) / 255.0)
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for f in frames]
    if len(grays) > 1:
        diffs = [float(np.abs(b - a).mean()) for a, b in zip(grays[:-1], grays[1:])]
        out["frame_change"] = float(np.mean(diffs))
        out["shot_cut_rate"] = float(np.mean([d > 0.25 for d in diffs]))   # hard-cut fraction
    else:
        out["frame_change"] = 0.0
        out["shot_cut_rate"] = 0.0
    return out


_LOW_NAMES = (_LOW_BASE + list(IC._SKIMAGE_NAMES)
              + ["hue_mean", "sat_mean", "frame_change", "shot_cut_rate"])


# ── caption-derived semantic scalars ──────────────────────────────────────────────
_SEM_NAMES = ["word_count", "mean_word_len", "sentence_count", "type_token_ratio",
              "valence", "arousal", "dominance", "concreteness", "sentiment",
              "people_rate", "action_rate", "animate_rate"]


def _sem_window(text):
    text = (text or "").strip()
    words = _WORD_RE.findall(text.lower())
    nw = len(words)
    out = dict.fromkeys(_SEM_NAMES, np.nan)
    out["word_count"] = float(nw)
    out["sentence_count"] = float(len([s for s in re.split(r"[.!?]+", text) if s.strip()]))
    if nw == 0:
        return out
    out["mean_word_len"] = float(np.mean([len(w) for w in words]))
    out["type_token_ratio"] = float(len(set(words)) / nw)
    vad = _nrc_vad(); con = _concreteness()
    vals = [_lookup(vad, w) for w in words]
    vals = [v for v in vals if v is not None]
    if vals:
        arr = np.asarray(vals)
        out["valence"], out["arousal"], out["dominance"] = (
            float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()))
    cons = [_lookup(con, w) for w in words]
    cons = [c for c in cons if c is not None]
    if cons:
        out["concreteness"] = float(np.mean(cons))
    sia = _vader()
    if sia is not None:
        out["sentiment"] = float(sia.polarity_scores(text)["compound"])
    elif not np.isnan(out["valence"]):
        out["sentiment"] = float((out["valence"] - 0.5) * 2.0)            # NRC fallback
    out["people_rate"] = float(sum(w in _PEOPLE for w in words) / nw)
    out["action_rate"] = float(sum(w in _ACTION for w in words) / nw)
    out["animate_rate"] = float(sum(w in _ANIMATE for w in words) / nw)
    return out


# ── per-grid build + cache ────────────────────────────────────────────────────────
def compute_grid(grid, max_windows=None, frame_size=FRAME_SIZE, verbose=True):
    win_sec = IC.grid_win_sec(grid)
    W = IC.grid_n_windows(grid)
    if max_windows:
        W = min(W, max_windows)
    starts = IC.canonical_starts(win_sec)[:W]
    print(f"[{grid}] {W} windows × {win_sec}s  (frame {frame_size}px)")

    IC.ensure_clips()
    texts = IC.grid_caption_texts(grid, W)

    low_rows, sem_rows = [], []
    for i, t in enumerate(starts):
        frames = IC.get_window_frames(float(t), win_sec, size=frame_size)
        low_rows.append([_low_window(frames)[n] for n in _LOW_NAMES])
        sem_rows.append([_sem_window(texts[i])[n] for n in _SEM_NAMES])
        if verbose and i % 200 == 0:
            print(f"    {grid}: window {i}/{W}", flush=True)

    low = np.asarray(low_rows, dtype=np.float32)
    sem = np.asarray(sem_rows, dtype=np.float32)
    feat = np.hstack([low, sem])
    feat_names = [f"low:{n}" for n in _LOW_NAMES] + [f"sem:{n}" for n in _SEM_NAMES]
    feat_tier = ["low"] * low.shape[1] + ["sem"] * sem.shape[1]
    print(f"  → {feat.shape[1]} scalar features ({low.shape[1]} low + {sem.shape[1]} sem)")
    return dict(grid=grid, win_sec=win_sec, W=W, starts=starts,
                feat=feat, feat_names=np.array(feat_names),
                feat_tier=np.array(feat_tier))


def cache_path(grid, smoke=False):
    return FEAT_DIR / f"{grid}{'__smoke' if smoke else ''}__features.npz"


def load_features(grid, smoke=False):
    p = cache_path(grid, smoke)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def build(grid, smoke=False, max_windows=None, frame_size=FRAME_SIZE, force=False):
    p = cache_path(grid, smoke)
    if p.exists() and not force:
        print(f"[{grid}] cached → {p.name}")
        return load_features(grid, smoke)
    mw = max_windows or (96 if smoke else None)
    res = compute_grid(grid, max_windows=mw, frame_size=frame_size)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **res)
    print(f"  saved → outputs/gridsearch/features/{p.name}")
    return res


def main():
    ap = argparse.ArgumentParser(description="Clean scalar feature stack for the grid search.")
    ap.add_argument("--grid", nargs="*", default=["reve"],
                    help="EEG model name(s) and/or 'clip4s', or 'all'.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--frame-size", type=int, default=FRAME_SIZE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)

    grids = list(C.EEG) + ["clip4s"] if args.grid == ["all"] else args.grid
    for g in grids:
        build(g, smoke=args.smoke, max_windows=args.max_windows,
              frame_size=args.frame_size, force=args.force)


if __name__ == "__main__":
    main()
