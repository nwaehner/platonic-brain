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
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"  saved → outputs/{name}")
    return path


def savefig(fig, name, dpi=150):
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → outputs/{name}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Interp redesign (Comp 1-5): 4 s-grid loaders, richer features, neighbourhood
#  statistics, ANCOVA. See ~/.claude/plans/1-so-are-we-lucky-marshmallow.md.
# ══════════════════════════════════════════════════════════════════════════════

# The 4 s ("clip4s") vision + LLM embeddings live in a SEPARATE dataset repo.
REPO_4S = "triniborrell/platonic-embeddings"


def _hf_load_from(path, repo):
    """Download `path` from an arbitrary HF dataset `repo` and return embeddings."""
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo, path, repo_type="dataset", token=C.HF_TOKEN_CACHE)
    return np.load(local, allow_pickle=True)["embeddings"].astype(np.float32)


def try_load_from(path, repo):
    try:
        return _hf_load_from(path, repo)
    except Exception as e:
        warnings.warn(f"could not load {path} from {repo}: {type(e).__name__}: {e}")
        return None


def load_eeg_emb(model, size):
    """Subject-mean EEG embedding (L, W, D) from the nitrox639 eeg/ tree, or None."""
    emb = try_load_emb(C.EEG[model]["fname"](size))           # (L, W, S, D) | nitrox639
    return None if emb is None else emb.mean(axis=2)


def load_eeg_emb_subjects(model, size):
    """Per-SUBJECT EEG embedding (L, W, S, D) — the same array `load_eeg_emb` averages,
    with the subject axis kept. Needed by the nested subject × time CV, which must never
    put a subject in both train and test (see feature_gridsearch.nested_ridge_r2)."""
    return try_load_emb(C.EEG[model]["fname"](size))


def load_vision_grid(arch, size, family):
    """Vision embedding (L, W, D). family='clip4s' → 4 s repo; else EEG-grid (nitrox639)."""
    path = C.VISION[arch]["path"](size, family)
    emb = try_load_from(path, REPO_4S) if family == "clip4s" else try_load_emb(path)
    if emb is None:
        return None
    return emb[None] if emb.ndim == 2 else emb


def load_llm_grid(grid, stem):
    """LLM caption embedding (L, W, D) from the 4 s repo. `grid` is 'clip4s' (4 s window
    grid) or an EEG model name (its window grid). Both live under llms/<grid>/ in REPO_4S."""
    emb = try_load_from(C.llm_path(grid, stem), REPO_4S)
    if emb is None:
        return None
    return emb[None] if emb.ndim == 2 else emb


# ── grid helpers (an EEG model name, or the 4 s 'clip4s' grid) ────────────────────
def grid_win_sec(grid):
    return CLIP_SEC if grid == "clip4s" else eeg_window_sec(grid)


def grid_n_windows(grid):
    return SEASON7_N_CLIPS if grid == "clip4s" else C.EEG_WINDOWS[grid]["n_windows"]


def grid_caption_texts(grid, n=None):
    """Per-window caption text on `grid`. clip4s = one caption per clip; EEG grids
    concatenate the captions overlapping each window (build_window_texts)."""
    caps = load_captions()
    if grid == "clip4s":
        texts = list(caps)
    else:
        texts, _ = build_window_texts(caps, grid_win_sec(grid), grid_n_windows(grid))
    return texts[:n] if n else texts


# ── intramodal pairing rule (shared by Comp 2 neighbour-layer & Comp 3 mKNN) ─────
def intramodal_partner(sizes, i):
    """Adjacent-size partner index for size i (ascending list): pair with the next
    larger size; the largest pairs with the second-largest. Single-size → None."""
    n = len(sizes)
    if n < 2:
        return None
    return i + 1 if i < n - 1 else i - 1


# ══════════════════════════════════════════════════════════════════════════════
#  Richer per-window features (Comp 1)
# ══════════════════════════════════════════════════════════════════════════════
_SKIMAGE_NAMES = ["glcm_contrast", "glcm_homogeneity", "glcm_energy",
                  "glcm_correlation", "lbp_entropy", "tex_entropy", "colorfulness"]


def skimage_window_features(frames):
    """Classical scikit-image texture/colour descriptors for one window's frames.
    Returns (vec(7,), names). Empty windows → NaNs. Highly-cited, model-free:
    GLCM/Haralick (contrast/homogeneity/energy/correlation), LBP entropy, image
    entropy, Hasler–Süsstrunk colorfulness."""
    import cv2
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.measure import shannon_entropy
    if not frames:
        return np.full(len(_SKIMAGE_NAMES), np.nan, dtype=np.float32), _SKIMAGE_NAMES
    mid = frames[len(frames) // 2]
    gray = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY)
    g32 = (gray.astype(np.float32) / 256 * 32).astype(np.uint8)       # 32 grey levels
    glcm = graycomatrix(g32, distances=[1], angles=[0, np.pi / 4, np.pi / 2,
                        3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    contrast    = float(graycoprops(glcm, "contrast").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    energy      = float(graycoprops(glcm, "energy").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    lbp_entropy = float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum())
    tex_entropy = float(shannon_entropy(gray))
    rgb = np.stack(frames).astype(np.float32)                         # (F,H,W,3)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = R - G, 0.5 * (R + G) - B
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    vec = np.array([contrast, homogeneity, energy, correlation, lbp_entropy,
                    tex_entropy, colorfulness], dtype=np.float32)
    return vec, _SKIMAGE_NAMES


# ── Empath validated lexical categories (replaces the ad-hoc _LEXICONS) ───────────
EMPATH_CATEGORIES = [
    "violence", "fight", "weapon", "war", "death", "fear", "nervousness", "anger",
    "aggression", "danger", "crime", "love", "affection", "positive_emotion",
    "negative_emotion", "movement", "speed", "vehicle", "driving", "travel", "home",
    "office", "work", "school", "family", "friends", "social", "communication",
    "music", "night", "morning", "body", "eating", "sports", "money", "animal",
    "nature", "urban", "water",
]


def empath_features(texts, categories=None):
    """(matrix (W, C), names) of Empath category scores (normalised). Import-guarded:
    if empath is unavailable returns an empty (W, 0) matrix + warning."""
    cats = categories or EMPATH_CATEGORIES
    try:
        from empath import Empath
    except Exception as e:
        warnings.warn(f"empath unavailable ({type(e).__name__}); skipping semantic "
                      f"lexical tier. `pip install empath` to enable.")
        return np.zeros((len(texts), 0), dtype=np.float32), []
    lex = Empath()
    M = np.zeros((len(texts), len(cats)), dtype=np.float32)
    for i, t in enumerate(texts):
        d = lex.analyze(t or "", categories=cats, normalize=True) or {}
        M[i] = [d.get(c, 0.0) for c in cats]
    return M, list(cats)


# ── CLIP (high-level visual tier + zero-shot labels + text embeddings) ────────────
_CLIP = {}


def clip_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def get_clip(model_name="openai/clip-vit-base-patch32", device=None):
    """Lazy-load a CLIP model + processor (cached). Returns the cache dict."""
    if not _CLIP:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = device or ("cuda" if torch.cuda.is_available()
                         else "mps" if getattr(torch.backends, "mps", None)
                         and torch.backends.mps.is_available() else "cpu")
        print(f"  loading CLIP {model_name} on {dev} …")
        model = CLIPModel.from_pretrained(model_name).to(dev).eval()
        proc = CLIPProcessor.from_pretrained(model_name)
        _CLIP.update(model=model, proc=proc, dev=dev, torch=torch, name=model_name)
    return _CLIP


def clip_image_embed(frames_per_window, max_frames=4, batch=32):
    """CLIP joint-space image embedding per window, mean over up to `max_frames`
    evenly-spaced frames. `frames_per_window`: list (len W) of lists of RGB arrays.
    Returns (W, d) float32 (NaN row for empty windows)."""
    cl = get_clip()
    torch, model, proc, dev = cl["torch"], cl["model"], cl["proc"], cl["dev"]
    d = model.config.projection_dim
    out = np.full((len(frames_per_window), d), np.nan, dtype=np.float32)
    flat_imgs, owner = [], []
    for w, frames in enumerate(frames_per_window):
        if not frames:
            continue
        idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)
        for j in idx:
            flat_imgs.append(frames[j]); owner.append(w)
    if not flat_imgs:
        return out
    embs = []
    with torch.no_grad():
        for s in range(0, len(flat_imgs), batch):
            px = proc(images=flat_imgs[s:s + batch], return_tensors="pt").to(dev)
            e = model.get_image_features(**px)
            embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    embs = np.concatenate(embs, 0)
    owner = np.asarray(owner)
    for w in np.unique(owner):
        out[w] = embs[owner == w].mean(0)
    return out


def clip_text_embed(texts, batch=64):
    """CLIP joint-space text embedding (W, d), l2-normalised."""
    cl = get_clip()
    torch, model, proc, dev = cl["torch"], cl["model"], cl["proc"], cl["dev"]
    embs = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            tok = proc(text=[t or "" for t in texts[s:s + batch]], return_tensors="pt",
                       padding=True, truncation=True, max_length=77).to(dev)
            e = model.get_text_features(**tok)
            embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(embs, 0).astype(np.float32)


def clip_zeroshot(image_emb, labels, prompt="a photo of {}"):
    """Softmax label probabilities per window. image_emb (W, d) already l2-normalised
    (NaN rows allowed). Returns (W, L) with NaN rows preserved."""
    cl = get_clip()
    scale = float(cl["model"].logit_scale.exp().detach().cpu())
    txt = clip_text_embed([prompt.format(l) for l in labels])         # (L, d)
    logits = image_emb @ txt.T * scale                                # (W, L)
    logits -= np.nanmax(logits, axis=1, keepdims=True)
    p = np.exp(logits)
    p /= np.nansum(p, axis=1, keepdims=True)
    return p.astype(np.float32)


def text_embed(texts, prefer_clip=True):
    """Sentence/caption embedding (W, d), l2-normalised. Uses the CLIP text encoder
    when torch is available, else a TF-IDF fallback (sklearn) so the cosine studies
    still run without a GPU stack. Returns (emb, backend_name)."""
    if prefer_clip and clip_available():
        try:
            return clip_text_embed(texts), "clip"
        except Exception as e:
            warnings.warn(f"CLIP text embed failed ({type(e).__name__}); TF-IDF fallback.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(max_features=4096, stop_words="english")
    X = vec.fit_transform([t or "" for t in texts])
    return normalize(np.asarray(X.todense(), dtype=np.float32)), "tfidf"


# ── interpretable label bank (NOT hand-curated; standard taxonomy + caption-mined) ──
# COCO-80 object classes (standard, exhaustive object taxonomy).
_COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
# Compact scene + action banks (stand-ins for Places365 / Kinetics; data-mined terms
# are appended at run time so the set is not cherry-picked).
_SCENES = [
    "indoor scene", "outdoor scene", "kitchen", "bedroom", "living room", "office",
    "hallway", "street", "city", "forest", "field", "mountain", "beach", "sky",
    "road", "restaurant", "store", "stadium", "night scene", "daytime scene",
]
_ACTIONS = [
    "people talking", "a person running", "a person walking", "people fighting",
    "a person driving", "people dancing", "a person falling", "a crowd of people",
    "a close-up of a face", "fast motion", "a violent scene", "a calm scene",
]


def build_label_bank(captions=None, n_mined=30, smoke=False):
    """Interpretable CLIP label bank: standard object/scene/action taxonomy + the most
    frequent caption nouns (data-driven, so the set is exhaustive not cherry-picked).
    Returns (labels, kinds) where kind ∈ {object,scene,action,mined}."""
    if smoke:
        labels = ["person", "car", "indoor scene", "outdoor scene", "a close-up of a face",
                  "a crowd of people", "fast motion", "a violent scene"]
        return labels, ["mixed"] * len(labels)
    labels = list(_COCO80) + list(_SCENES) + list(_ACTIONS)
    kinds = (["object"] * len(_COCO80) + ["scene"] * len(_SCENES)
             + ["action"] * len(_ACTIONS))
    if captions:
        stop = set("the a an and or of to in on at is are was were be been with for "
                   "this that these those it its his her their they he she you we as "
                   "by from into over under then there here who which what while two "
                   "one scene video features shows depicting appears seems various "
                   "different first second".split())
        from collections import Counter
        import re
        cnt = Counter()
        for t in captions:
            for w in re.findall(r"[a-z]{4,}", (t or "").lower()):
                if w not in stop:
                    cnt[w] += 1
        existing = set(labels)
        for w, _ in cnt.most_common():
            if w in existing:
                continue
            labels.append(w); kinds.append("mined")
            if sum(k == "mined" for k in kinds) >= n_mined:
                break
    return labels, kinds


# ══════════════════════════════════════════════════════════════════════════════
#  Neighbourhood statistics (Comp 2 / 4) — no regression on y
# ══════════════════════════════════════════════════════════════════════════════
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def cles_auc(a, b):
    """Common-language effect size P(a > b) (0.5 = no contrast). Rank-based."""
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    try:
        u = mannwhitneyu(a, b, alternative="greater").statistic
    except ValueError:
        return 0.5
    return float(u / (len(a) * len(b)))


def ks_stat(a, b):
    from scipy.stats import ks_2samp
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def benjamini_hochberg(pvals):
    """BH-FDR adjusted q-values for a 1-D array of p-values."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return q
    ps = p[idx]
    order = np.argsort(ps)
    n = len(ps)
    ranked = ps[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    q[idx[order]] = ranked
    return q


def matched_perm_enrichment(fvals, neighbour_sets, R=1000, seed=0):
    """Neighbour-vs-population enrichment of one feature with a matched-size null.
    `fvals` (W_pop,) feature over the whole grid; `neighbour_sets` list of index
    arrays (one per anchor). Null = R draws of random index sets with the SAME sizes.
    Returns dict(mean_shift_sigma, var_ratio, cohens_d, auc, ks, p_mean, p_var,
    n_neighbours, neigh_values, pop_values)."""
    rng = np.random.default_rng(seed)
    f = np.asarray(fvals, float)
    finite = f[np.isfinite(f)]
    pop_mean, pop_sd = finite.mean(), finite.std() + 1e-12
    pop_var = finite.var() + 1e-12
    sizes = [len(s) for s in neighbour_sets if len(s) > 0]
    neigh = np.concatenate([f[s] for s in neighbour_sets if len(s) > 0]) \
        if sizes else np.array([])
    neigh = neigh[np.isfinite(neigh)]
    if neigh.size < 2:
        return dict(mean_shift_sigma=np.nan, var_ratio=np.nan, cohens_d=np.nan,
                    auc=np.nan, ks=np.nan, p_mean=np.nan, p_var=np.nan,
                    n_neighbours=int(neigh.size), neigh_values=neigh, pop_values=finite)
    ntot = neigh.size
    obs_mean, obs_var = neigh.mean(), neigh.var()
    null_mean = np.empty(R); null_var = np.empty(R)
    for r in range(R):
        samp = finite[rng.integers(0, finite.size, ntot)]
        null_mean[r] = samp.mean(); null_var[r] = samp.var()
    p_mean = 2 * min((null_mean >= obs_mean).mean(), (null_mean <= obs_mean).mean())
    p_var = 2 * min((null_var >= obs_var).mean(), (null_var <= obs_var).mean())
    return dict(
        mean_shift_sigma=float((obs_mean - pop_mean) / pop_sd),
        var_ratio=float(obs_var / pop_var),
        cohens_d=cohens_d(neigh, finite),
        auc=cles_auc(neigh, finite), ks=ks_stat(neigh, finite),
        p_mean=float(min(p_mean, 1.0)), p_var=float(min(p_var, 1.0)),
        n_neighbours=int(ntot), neigh_values=neigh, pop_values=finite)


def varratio_test(feat, shared_sets, R=200, min_n=2, seed=0):
    """Per-window variance-ratio enrichment (Comp 2): for every window with ≥min_n shared
    neighbours, ratio_w(f) = var(feat[neighbours]) / var(feat[all]). Tests, per feature,
    whether the median ratio is SMALLER than under a matched-n permutation null (random
    same-size sets). feat (W, F) must be finite. Returns dict or None if too few windows:
      ratios (U, F) per-usable-window ratios · median_ratio (F,) · p (F,) one-sided <null ·
      q (F,) BH-FDR · sizes (U,) · n_used."""
    feat = np.asarray(feat, float)
    W, F = feat.shape
    gv = feat.var(0, ddof=1) + 1e-12
    usable = [np.asarray(s, int) for s in shared_sets if len(s) >= min_n]
    if len(usable) < 3:
        return None
    obs = np.array([feat[s].var(0, ddof=1) / gv for s in usable])     # (U, F)
    med = np.nanmedian(obs, axis=0)                                   # (F,)
    sizes = np.array([len(s) for s in usable])
    rng = np.random.default_rng(seed)
    groups = {int(s): np.where(sizes == s)[0] for s in np.unique(sizes)}
    null_med = np.empty((R, F))
    for r in range(R):
        rat = np.empty((len(usable), F))
        for s, rows in groups.items():
            idx = rng.integers(0, W, size=(len(rows), s))             # (m, s)
            rat[rows] = feat[idx].var(1, ddof=1) / gv                 # (m, F)
        null_med[r] = np.median(rat, axis=0)
    p = (np.sum(null_med <= med[None, :], axis=0) + 1) / (R + 1)
    return dict(ratios=obs, median_ratio=med, p=p, q=benjamini_hochberg(p),
                sizes=sizes, n_used=len(usable))


def trace_cov_ratio_test(emb, shared_sets, R=200, min_n=2, seed=0):
    """Multivariate analogue of varratio_test for the caption embedding (Comp 4):
    per window, cap_ratio = trace(cov(emb[neighbours])) / trace(cov(emb[all])). For unit
    vectors trace(cov)=1−‖centroid‖², so small ratio ⇔ tight (high mutual cosine) cluster.
    Returns dict(ratios (U,), median, p one-sided <null, n_used) or None."""
    emb = np.asarray(emb, float)
    W = emb.shape[0]
    gtv = emb.var(0, ddof=1).sum() + 1e-12
    usable = [np.asarray(s, int) for s in shared_sets if len(s) >= min_n]
    if len(usable) < 3:
        return None
    obs = np.array([emb[s].var(0, ddof=1).sum() / gtv for s in usable])
    med = float(np.median(obs))
    sizes = np.array([len(s) for s in usable])
    rng = np.random.default_rng(seed)
    groups = {int(s): np.where(sizes == s)[0] for s in np.unique(sizes)}
    null_med = np.empty(R)
    for r in range(R):
        rat = np.empty(len(usable))
        for s, rows in groups.items():
            idx = rng.integers(0, W, size=(len(rows), s))
            rat[rows] = emb[idx].var(1, ddof=1).sum(-1) / gtv
        null_med[r] = np.median(rat)
    p = (np.sum(null_med <= med) + 1) / (R + 1)
    return dict(ratios=obs, median=med, p=float(p), n_used=len(usable))


def ancova_family(df):
    """ANCOVA R2 ~ MKNN + C(family). df columns: R2, MKNN, family. Returns the
    family-adjusted MKNN slope, its p, partial η² (type-II SS for MKNN added last), the
    unadjusted slope, and the added-variable residuals (R2|family, MKNN|family).

    Uses statsmodels.api.OLS with explicit dummies (NOT patsy formulas) because the
    module-global `C` (= alignment_plots/_common) shadows patsy's categorical `C()`."""
    import statsmodels.api as sm
    import pandas as pd
    d = df.dropna(subset=["R2", "MKNN", "family"]).copy()
    out = dict(n=len(d), slope_adj=np.nan, p_mknn=np.nan, partial_eta2=np.nan,
               slope_unadj=np.nan, resid_r2=np.array([]), resid_mknn=np.array([]))
    if len(d) < 4 or d["MKNN"].nunique() < 2:
        return out
    out["slope_unadj"] = float(np.polyfit(d["MKNN"], d["R2"], 1)[0])
    if d["family"].nunique() < 2:                    # ANCOVA needs ≥2 families
        out["slope_adj"] = out["slope_unadj"]
        return out
    y = d["R2"].to_numpy(float)
    mk = d["MKNN"].to_numpy(float)
    dum = pd.get_dummies(d["family"], drop_first=True).astype(float).to_numpy()
    Xf = sm.add_constant(dum)                         # family-only design
    Xfull = sm.add_constant(np.column_stack([mk, dum]))
    full = sm.OLS(y, Xfull).fit()
    out["slope_adj"] = float(full.params[1])         # col 0 = const, col 1 = MKNN
    out["p_mknn"] = float(full.pvalues[1])
    reduced = sm.OLS(y, Xf).fit()                    # drop MKNN
    ss_mknn = float(reduced.ssr - full.ssr)          # type-II SS for MKNN
    out["partial_eta2"] = float(ss_mknn / (ss_mknn + full.ssr + 1e-12))
    out["resid_r2"] = (y - reduced.predict(Xf))
    out["resid_mknn"] = (mk - sm.OLS(mk, Xf).fit().predict(Xf))
    return out
