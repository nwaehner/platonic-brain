"""
_interp_common.py — shared toolkit for the `src/interp/` post-hoc studies.

This module is the single hub for the four interpretability studies. It:

  • bootstraps `sys.path` so the alignment_plots `_common` module is importable
    (the repo has no packages; scripts rely on their own dir being on the path),
    re-exported here as `C`;
  • applies the requested interp plot theme (`_style`) last, so it wins over
    `_common`'s theme;
  • ports the intrinsic-dimensionality estimators from
    `notebooks/manifold_analysis.ipynb` (participation_ratio, twonn_id, local_twonn)
    and adds two model-free robustness estimators (mle_id, effective_rank);
  • ports `mknn_per_window` / `canonical_starts` from
    `notebooks/visualize_mknn_stimuli.ipynb`;
  • ports the CineBrain video-frame machinery (ensure_clips, get_window_frames, …);
  • reimplements the *caption→window* alignment rule from
    `src/extraction_scripts/extract_llm_captions.py` (no torch import needed);
  • adds classical, MODEL-FREE visual features (color histogram, optical-flow motion,
    spatial frequency, luminance/contrast, edge density) — none use a CNN;
  • adds interpretable caption-derived semantic attributes;
  • provides model enumeration over the `_common` registries, HF loaders with a
    4 s-family fallback, small stats helpers (gini/lorenz, OLS R², commonality
    variance partition, partial correlation), and NPZ/PNG save helpers.

Everything reuses the uploaded `nitrox639/platonic-embeddings` embeddings — there is
NO foundation-model inference anywhere. The only raw-pixel access is the classical
visual-feature path (studies 1b / 4), which reads cached CineBrain frames.
"""

from __future__ import annotations

import functools
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ── path bootstrap: make alignment_plots/_common importable, then apply theme ───
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]                      # …/platonic-brain
_ALIGN_DIR = _THIS.parent.parent / "alignment_plots"
for _d in (str(_THIS.parent), str(_ALIGN_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import _common as C          # noqa: E402  (alignment_plots/_common.py)
import _style                # noqa: E402  (overrides _common's theme — keep last)

_style.apply_style()


# ── constants ───────────────────────────────────────────────────────────────────
OUT_DIR = _THIS.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CLIP_SEC          = 4
SEASON7_N_CLIPS   = 2700
SEASON7_TOTAL_SEC = SEASON7_N_CLIPS * CLIP_SEC       # 10_800
SEASON_END_S      = float(SEASON7_TOTAL_SEC)
SOURCE_FPS        = 8
GIF_SIZE          = 160

CLIP_CACHE     = PROJECT_ROOT / ".clip_cache"
CAPTIONS_CACHE = PROJECT_ROOT / "data" / "captions-qwen-2.5-vl-7b.json"
VIDEO_REPO     = "Fudan-fMRI/CineBrain"
CAPTIONS_FILE  = "captions-qwen-2.5-vl-7b.json"

# Default 4 s family for vision/LLM, and the fallbacks used until the 4 s vision
# upload lands (LLM clip4s already exists; vision clip4s may 404 → femba_luna).
DEFAULT_FAMILY        = "clip4s"
FALLBACK_VISION_FAM   = "femba_luna"
FALLBACK_LLM_FAM      = "neurolm"


def set_token(cli_token=None):
    """Resolve and cache the HF token in `_common` so all loaders authenticate."""
    C.HF_TOKEN_CACHE = C.resolve_hf_token(cli_token)
    return C.HF_TOKEN_CACHE


# ── embedding loaders (HF, with graceful 4 s-family fallback) ────────────────────
def load_emb(path):
    """Download `path` from the embeddings dataset and return the float32 array."""
    return C.load_npz(path)["embeddings"].astype(np.float32)


def try_load_emb(path):
    """Return the embeddings array at `path`, or None if it is not on the repo."""
    try:
        return load_emb(path)
    except Exception as e:                            # EntryNotFound / HTTP / …
        warnings.warn(f"could not load {path}: {type(e).__name__}: {e}")
        return None


def load_vision_emb(arch, size, family):
    """(emb, used_family). Falls back to FALLBACK_VISION_FAM if `family` is absent."""
    emb = try_load_emb(C.VISION[arch]["path"](size, family))
    if emb is not None:
        return emb, family
    if family != FALLBACK_VISION_FAM:
        warnings.warn(f"vision {arch}-{size}: family '{family}' unavailable → "
                      f"falling back to '{FALLBACK_VISION_FAM}'")
        emb = try_load_emb(C.VISION[arch]["path"](size, FALLBACK_VISION_FAM))
        if emb is not None:
            return emb, FALLBACK_VISION_FAM
    return None, None


def load_llm_emb(stem, family):
    """(emb, used_family). LLM caption embeddings live under llms/<family>/."""
    emb = try_load_emb(C.llm_path(family, stem))
    if emb is not None:
        return emb, family
    if family != FALLBACK_LLM_FAM:
        warnings.warn(f"llm {stem}: family '{family}' unavailable → "
                      f"falling back to '{FALLBACK_LLM_FAM}'")
        emb = try_load_emb(C.llm_path(FALLBACK_LLM_FAM, stem))
        if emb is not None:
            return emb, FALLBACK_LLM_FAM
    return None, None


# ── model enumeration over the _common registries ────────────────────────────────
def iter_eeg():
    for m, cfg in C.EEG.items():
        for s in cfg["sizes"]:
            yield m, s


def iter_vision():
    for a, cfg in C.VISION.items():
        for s in cfg["sizes"]:
            yield a, s


def iter_llm():
    for fam, stems in C.LLM.items():
        for st in stems:
            yield fam, st


def eeg_window_sec(eeg_model):
    return C.EEG_WINDOWS[eeg_model]["win_sec"]


# ── intrinsic-dimensionality estimators ──────────────────────────────────────────
# participation_ratio / twonn_id / local_twonn are ported verbatim from
# notebooks/manifold_analysis.ipynb; mle_id and effective_rank are added here.
def covariance_eigs(X):
    """Eigenvalues (≥0, descending) of X's centred covariance, via the smaller of
    the Gram / covariance matrices. Compute once and feed the *_from_eigs helpers."""
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[1] > Xc.shape[0]:
        ev = np.linalg.eigvalsh(Xc @ Xc.T).clip(min=0)
    else:
        ev = np.linalg.eigvalsh(Xc.T @ Xc).clip(min=0)
    return np.sort(ev)[::-1]


def pr_from_eigs(ev):
    s = ev.sum()
    return float(s * s / (np.square(ev).sum() + 1e-12))


def effrank_from_eigs(ev):
    s = ev.sum()
    if s <= 0:
        return float("nan")
    p = ev / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def eigspectrum_from_eigs(ev, n_top=64):
    s = ev.sum()
    ev = ev / s if s > 0 else ev
    out = np.full(n_top, np.nan)
    out[:min(n_top, len(ev))] = ev[:n_top]
    return out


def participation_ratio(X):
    """Linear ID. X: (N, D). Returns (Σλ)² / Σλ² of the (centred) covariance."""
    return pr_from_eigs(covariance_eigs(X))


def twonn_id(X, frac_keep=0.9):
    """Non-linear ID (Facco et al. 2017). Slope of -log(1-F(μ)) vs log μ on the
    lower `frac_keep` of μ = r2/r1 (Euclidean 2nd/1st-NN ratio)."""
    X = np.asarray(X)
    if X.shape[0] < 4:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    mu = r2 / np.maximum(r1, 1e-12)
    mu = mu[mu > 1.0 + 1e-12]
    if len(mu) < 4:
        return float("nan")
    mu.sort()
    n = max(int(len(mu) * frac_keep), 4)
    mu = mu[:n]
    F = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    slope, _ = np.polyfit(np.log(mu), -np.log(1 - F), 1)
    return float(slope)


def mle_id(X, k1=10, k2=20):
    """Levina–Bickel MLE intrinsic dimension, averaged (MacKay–Ghahramani) over
    k ∈ [k1, k2]. Model-free, a robustness companion to TwoNN. X: (N, D)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < k2 + 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k2 + 1, metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X)
    logd = np.log(np.maximum(d[:, 1:], 1e-12))        # (n, k2) neighbours 1..k2
    invs = []
    for k in range(k1, k2 + 1):
        # Σ_{j<k} log(d_k / d_j)  /  (k-1), averaged over points
        s = (logd[:, k - 1:k] - logd[:, :k - 1]).sum(axis=1) / (k - 1)
        invs.append(float(s.mean()))
    invm = float(np.mean(invs))
    return float(1.0 / invm) if invm > 0 else float("nan")


def effective_rank(X):
    """exp(spectral entropy) of the covariance eigenvalues — a soft rank in [1, D]."""
    return effrank_from_eigs(covariance_eigs(X))


def eigenspectrum(X, n_top=64):
    """Top-`n_top` covariance eigenvalues, normalised to sum 1, descending."""
    return eigspectrum_from_eigs(covariance_eigs(X), n_top)


def local_twonn(X, k=20):
    """Per-point local TwoNN ID in each point's k cosine-neighbourhood. X (N, D)."""
    Xn = C.l2(X)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xn)
    _, idx = nn.kneighbors(Xn)
    out = np.empty(Xn.shape[0])
    for i in range(Xn.shape[0]):
        out[i] = twonn_id(Xn[idx[i, 1:]])             # exclude self
    return out


# ── per-window mKNN (from visualize_mknn_stimuli.ipynb) ──────────────────────────
def mknn_per_window(ia, ib):
    """Per-window shared-neighbour fraction. ia, ib: (W, k) → (W,) in [0, 1]."""
    return (ia[:, :, None] == ib[:, None, :]).any(-1).mean(axis=-1)


def canonical_starts(window_s):
    return np.arange(0.0, SEASON_END_S, float(window_s), dtype=np.float64)


def best_layer_pair(nn_a, nn_b):
    """argmax mKNN over (a-layer, b-layer). nn_a (LA,W,k), nn_b (LB,W,k).
    Returns (la*, lb*, score)."""
    best = (-1, -1, -1.0)
    for la in range(nn_a.shape[0]):
        for lb in range(nn_b.shape[0]):
            sc = C.mknn_1d(nn_a[la], nn_b[lb])
            if sc > best[2]:
                best = (la, lb, sc)
    return best


# ── captions: load + window alignment (mirrors extract_llm_captions.py) ──────────
def _clip_id(video_path):
    digits = "".join(ch for ch in Path(video_path).stem if ch.isdigit())
    return int(digits)


def load_captions(cache_path=CAPTIONS_CACHE):
    """List of 2700 caption strings (Season 7), ordered by clip id. Downloads the
    JSON from the CineBrain dataset on first use."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        from huggingface_hub import hf_hub_download
        print(f"Downloading {CAPTIONS_FILE} from {VIDEO_REPO} …")
        local = hf_hub_download(VIDEO_REPO, CAPTIONS_FILE, repo_type="dataset",
                                token=C.HF_TOKEN_CACHE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(Path(local).read_bytes())
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    data.sort(key=lambda e: _clip_id(e["video"]))
    return [e["text"] for e in data[:SEASON7_N_CLIPS]]


def overlapping_captions(w, win_sec):
    """Caption indices whose [4c, 4c+4) overlaps window [w·L, (w+1)·L)."""
    w0, w1 = w * win_sec, (w + 1) * win_sec
    c_lo = max(0, w0 // CLIP_SEC - 1)
    c_hi = min(SEASON7_N_CLIPS, w1 // CLIP_SEC + 2)
    return [c for c in range(c_lo, c_hi)
            if min((c + 1) * CLIP_SEC, w1) - max(c * CLIP_SEC, w0) > 0]


def build_window_texts(captions, win_sec, n_windows):
    """(texts[W], n_caps[W]) — concatenate every overlapping caption per window."""
    texts, n_caps = [], []
    for w in range(n_windows):
        sel = overlapping_captions(w, int(win_sec))
        texts.append(" ".join(captions[c] for c in sel))
        n_caps.append(len(sel))
    return texts, np.asarray(n_caps, dtype=np.int64)


# ── interpretable caption attributes (model-free keyword lexicons) ───────────────
_LEXICONS = {
    "person":  ["person", "people", "man", "men", "woman", "women", "boy", "girl",
                "child", "guy", "lady", "crowd", "he ", "she ", "they "],
    "face":    ["face", "smil", "eyes", "expression", "looking", "gaze"],
    "indoor":  ["room", "indoor", "inside", "kitchen", "office", "house", "bedroom",
                "hallway", "wall", "table", "interior"],
    "outdoor": ["outdoor", "outside", "street", "sky", "forest", "field", "mountain",
                "beach", "road", "car ", "tree", "building", "city"],
    "action":  ["run", "walk", "fight", "jump", "move", "driv", "danc", "throw",
                "chase", "fall", "grab", "push", "fast"],
    "talk":    ["talk", "speak", "say", "conversa", "telling", "asks", "shout"],
    "tension": ["tense", "anger", "angry", "fear", "scream", "cry", "danger", "dark",
                "blood", "gun", "weapon", "fight", "threat", "panic", "scared"],
}


def caption_attributes(texts):
    """Return (matrix (W, A), names). Each column is a normalised keyword-hit score
    for one interpretable axis (person / face / indoor / outdoor / action / talk /
    tension). Purely lexical — no model."""
    names = list(_LEXICONS.keys())
    M = np.zeros((len(texts), len(names)), dtype=np.float32)
    for i, t in enumerate(texts):
        low = " " + t.lower() + " "
        for j, key in enumerate(names):
            M[i, j] = sum(low.count(kw) for kw in _LEXICONS[key])
    # per-axis z-ish scaling so axes are comparable
    M = (M - M.mean(0, keepdims=True)) / (M.std(0, keepdims=True) + 1e-9)
    return M, names


# ── CineBrain video frames (ported from manifold_analysis.ipynb) ────────────────
_CLIP_INDEX = {}


def _build_clip_index(clips_dir):
    _CLIP_INDEX.clear()
    for p in Path(clips_dir).rglob("*.mp4"):
        digits = "".join(c for c in p.stem if c.isdigit())
        if digits:
            _CLIP_INDEX[int(digits)] = p
    if _CLIP_INDEX:
        print(f"  indexed {len(_CLIP_INDEX)} clips "
              f"(range {min(_CLIP_INDEX)}..{max(_CLIP_INDEX)})")


def ensure_clips():
    """Make CineBrain frames available locally (download+extract videos.tar once)."""
    if _CLIP_INDEX:
        return
    marker = CLIP_CACHE / ".extracted"
    if marker.exists():
        _build_clip_index(CLIP_CACHE)
        return
    from huggingface_hub import hf_hub_download
    import tarfile
    print("Downloading videos.tar (~2.6 GB) …")
    tar_path = Path(hf_hub_download(VIDEO_REPO, "videos.tar", repo_type="dataset",
                                    token=C.HF_TOKEN_CACHE))
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(CLIP_CACHE)
    marker.touch()
    _build_clip_index(CLIP_CACHE)


@functools.lru_cache(maxsize=256)
def _load_frames(clip_idx):
    import cv2
    cap = cv2.VideoCapture(str(_CLIP_INDEX[clip_idx]))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return tuple(frames)


def get_window_frames(start_s, window_s, size=GIF_SIZE):
    """RGB frames covering [start_s, start_s+window_s), resized to size×size."""
    import cv2
    end_s = start_s + window_s
    first_clip = int(start_s // CLIP_SEC)
    last_clip = min(int(np.ceil(end_s / CLIP_SEC)) - 1, SEASON7_N_CLIPS - 1)
    frames = []
    for c in range(first_clip, last_clip + 1):
        if c not in _CLIP_INDEX:
            continue
        cf = _load_frames(c)
        n = len(cf)
        if n == 0:
            continue
        ct0 = c * CLIP_SEC
        f0 = max(0, int(np.floor((max(start_s, ct0) - ct0) / CLIP_SEC * n)))
        f1 = min(n, int(np.ceil((min(end_s, ct0 + CLIP_SEC) - ct0) / CLIP_SEC * n)))
        for fr in cf[f0:f1]:
            frames.append(cv2.resize(fr, (size, size), interpolation=cv2.INTER_LINEAR))
    return frames


def frames_to_gif(frames, fps=SOURCE_FPS):
    from PIL import Image as PILImage
    pil = [PILImage.fromarray(f) for f in frames]
    buf = io.BytesIO()
    pil[0].save(buf, format="GIF", save_all=True, append_images=pil[1:],
                duration=int(1000 / fps), loop=0, optimize=False)
    return buf.getvalue()


# ── classical, model-free visual features (no CNN) ───────────────────────────────
def window_visual_features(frames):
    """Concatenated classical visual descriptor for one window's frames, plus a
    dict of named scalar summaries. All hand-engineered (no neural network):
      • HSV colour histogram (8×8×8)        → appearance / palette
      • optical-flow motion energy          → how much is moving
      • spatial-frequency (FFT) energy      → texture / detail
      • luminance mean + contrast (std)     → brightness
      • Canny edge density                  → structural busyness
    Returns (vector, scalars_dict). Empty windows return NaNs of the right shape."""
    import cv2
    n_hist = 8 * 8 * 8
    if not frames:
        vec = np.full(n_hist + 5, np.nan, dtype=np.float32)
        scal = dict(motion=np.nan, spatial_freq=np.nan, luminance=np.nan,
                    contrast=np.nan, edge_density=np.nan)
        return vec, scal

    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    # colour histogram (mean over frames, in HSV)
    hist = np.zeros((8, 8, 8), dtype=np.float64)
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                         [0, 180, 0, 256, 0, 256])
        hist += h
    hist = (hist / (hist.sum() + 1e-9)).ravel()

    # optical-flow motion energy (mean Farnebäck magnitude across consecutive frames)
    if len(grays) > 1:
        mags = []
        for a, b in zip(grays[:-1], grays[1:]):
            flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(float(np.linalg.norm(flow, axis=2).mean()))
        motion = float(np.mean(mags))
    else:
        motion = 0.0

    # spatial-frequency energy (mean high-pass FFT magnitude of the mid frame)
    g = grays[len(grays) // 2].astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)
    cy, cx = np.array(mag.shape) // 2
    mag[cy - 4:cy + 4, cx - 4:cx + 4] = 0.0            # drop low frequencies
    spatial_freq = float(mag.mean())

    gray_stack = np.stack(grays).astype(np.float32)
    luminance = float(gray_stack.mean())
    contrast = float(gray_stack.std())
    edge_density = float(np.mean([cv2.Canny(g, 100, 200).mean() / 255.0
                                  for g in grays]))

    scal = dict(motion=motion, spatial_freq=spatial_freq, luminance=luminance,
                contrast=contrast, edge_density=edge_density)
    vec = np.concatenate([hist, np.array([motion, spatial_freq, luminance,
                                          contrast, edge_density])]).astype(np.float32)
    return vec, scal


_VISUAL_SCALAR_NAMES = ["motion", "spatial_freq", "luminance", "contrast",
                        "edge_density"]


def compute_visual_features(eeg_model, frame_size=96, verbose=True):
    """Per-window classical visual descriptor for an EEG model's window grid.
    Cached to OUT_DIR/visual_features__<eeg_model>.npz so it is computed once.
    Returns dict(vec (W, d_v), scalars (W, 5), scalar_names, starts (W,), win_sec)."""
    cache = OUT_DIR / f"visual_features__{eeg_model}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(vec=z["vec"], scalars=z["scalars"],
                    scalar_names=list(z["scalar_names"]), starts=z["starts"],
                    win_sec=int(z["win_sec"]))
    ensure_clips()
    win_sec = eeg_window_sec(eeg_model)
    starts = canonical_starts(win_sec)
    vecs, scals = [], []
    for i, t in enumerate(starts):
        v, s = window_visual_features(get_window_frames(t, win_sec, size=frame_size))
        vecs.append(v)
        scals.append([s[k] for k in _VISUAL_SCALAR_NAMES])
        if verbose and i % 200 == 0:
            print(f"    visual {eeg_model}: {i}/{len(starts)}", flush=True)
    vec = np.asarray(vecs, dtype=np.float32)
    scalars = np.asarray(scals, dtype=np.float32)
    np.savez_compressed(cache, vec=vec, scalars=scalars,
                        scalar_names=np.array(_VISUAL_SCALAR_NAMES),
                        starts=starts, win_sec=win_sec)
    print(f"  saved visual features → {cache.name}")
    return dict(vec=vec, scalars=scalars, scalar_names=_VISUAL_SCALAR_NAMES,
                starts=starts, win_sec=win_sec)


# ── small stats helpers ──────────────────────────────────────────────────────────
def gini(x):
    """Gini coefficient of a non-negative vector (0 = uniform, →1 = concentrated)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    x = x - x.min() if x.min() < 0 else x
    n = len(x)
    cum = np.cumsum(x)
    if cum[-1] <= 0:
        return 0.0
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def lorenz(x):
    """Lorenz-style concentration curve of `x` (descending). Returns (frac_windows,
    cum_share) so cum_share[j] = share of the total carried by the top j windows."""
    x = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    x = np.clip(x, 0, None)
    cum = np.cumsum(x)
    cum = cum / cum[-1] if cum[-1] > 0 else cum
    frac = np.arange(1, len(x) + 1) / len(x)
    return frac, cum


def _design(X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] == 1 and X.shape[1] != 1:
        X = X.T
    return np.hstack([np.ones((X.shape[0], 1)), X])


def r2_ols(y, X):
    """Ordinary-least-squares R² of y on X (intercept added). X: (N,) or (N, P)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    A = _design(X)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def variance_partition(y, blocks):
    """Commonality analysis for predictor blocks {name: (N, P_i)}.
    Returns dict with R² for every non-empty subset (keys = '+'-joined names),
    plus 'unique_<name>' = R²(all) − R²(all but name). For 2 blocks this gives the
    classic unique/unique/shared decomposition."""
    names = list(blocks)
    from itertools import combinations
    subsets = {}
    for r in range(1, len(names) + 1):
        for combo in combinations(names, r):
            key = "+".join(combo)
            X = np.hstack([np.atleast_2d(blocks[n]).reshape(len(y), -1) for n in combo])
            subsets[key] = r2_ols(y, X)
    full = "+".join(names)
    out = {f"R2[{k}]": v for k, v in subsets.items()}
    for n in names:
        rest = [m for m in names if m != n]
        r2_rest = subsets["+".join(rest)] if rest else 0.0
        out[f"unique_{n}"] = subsets[full] - r2_rest
    out["R2_full"] = subsets[full]
    return out


def partial_corr(x, y, z):
    """Pearson correlation of x and y after linearly removing z (z: (N,) or (N,P))."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    Z = _design(z)
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


# ── save helpers ─────────────────────────────────────────────────────────────────
def save_npz(name, **arrays):
    path = OUT_DIR / name
    np.savez_compressed(path, **arrays)
    print(f"  saved → outputs/{name}")
    return path


def savefig(fig, name, dpi=150):
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → outputs/{name}")
    return path
