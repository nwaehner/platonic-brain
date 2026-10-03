"""
_common.py — shared toolkit for the alignment_plots scripts.

Single source of truth for:
  • the serif/black plot theme,
  • HuggingFace auth + loaders for the `nitrox639/platonic-embeddings` dataset,
  • the EEG / vision / LLM model registries and display metadata,
  • mKNN kNN primitives and the Aristotelian permutation calibration,
  • the generic cross-modal driver `run_cross_modal`,
  • the two *temporal-confound-aware* null tests (block permutation + shift-null),
  • the performance scores used as plot x-axes (LLM 1-BPB, video Kinetics-400 top-1).

Everything here is lifted from `src/layer_study/cross_modal_layer_study.py` and
`notebooks/analyze_mknn_hf.ipynb` (cells 1, 33, 36) so the scripts in this package
share exactly one implementation.

────────────────────────────────────────────────────────────────────────────────
Why the extra null tests exist (read this — it motivates the whole package)
────────────────────────────────────────────────────────────────────────────────
mKNN rewards *shared neighbour structure*. In naturalistic movie data the window
index is temporal, so window t is a neighbour of t±1 in BOTH the EEG embeddings and
the model embeddings simply because each signal is a smooth function of time. A high
mKNN can therefore reflect a shared *time axis* ("when") rather than a shared
*content geometry* ("what is happening") — and the Platonic Representation Hypothesis
is a claim about content, not time.

The classic Aristotelian null shuffles all windows i.i.d., which destroys the very
temporal structure both modalities share, so it *under*-estimates the baseline and
over-states significance. Two stronger nulls fix this:

  • block_perm_null  — permutes contiguous time-blocks, PRESERVING local
                       autocorrelation in the null. Alignment above this null is
                       evidence of content structure beyond temporal smoothness.
  • shift_null       — re-pairs the two modalities at many temporal offsets d and
                       asks whether the true pairing (d=0) beats temporally-plausible
                       mis-pairings. p = fraction of shifts with score ≥ the d=0 score.

Both supersede temporal exclusion, which is over-conservative (it also discards the
genuine shared content of adjacent windows).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from huggingface_hub import hf_hub_download


# ── Plot theme (shared across every figure in this package) ────────────────────
COLOR = "black"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (14, 9),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "legend.fontsize": "medium",
        "legend.title_fontsize": 22,
        "axes.titlesize": 22,
        "axes.labelsize": "large",
        "ytick.labelsize": 16,
        "xtick.labelsize": 16,
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)


# ── HuggingFace auth ───────────────────────────────────────────────────────────
REPO = "nicolaswaehner/platonic-embeddings"
# Writable dataset for the embeddings WE generate (llms/ + the clip4s vision/). Defaults
# to REPO; set PLATONIC_LLM_REPO (e.g. in .env) to redirect them to your own dataset —
# needed when REPO is read-only for your token. The pre-existing eeg/ and the EEG-family
# vision/ always load from REPO.
LLM_REPO = os.environ.get("PLATONIC_LLM_REPO", REPO)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOKEN_CANDIDATES = [
    _PROJECT_ROOT / "tokens" / "hf_token.txt",
    Path("tokens/hf_token.txt"),
    Path("hf_token.txt"),
    Path.home() / "hf_token.txt",
    Path.home() / ".cache" / "huggingface" / "token",
]

HF_TOKEN_CACHE = None  # may be set by a script's main() from --hf-token


def resolve_hf_token(cli_token=None):
    if cli_token:
        return cli_token.strip()
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    for p in _TOKEN_CANDIDATES:
        if p.exists():
            return p.read_text().strip()
    return None


# Optional local override: if PLATONIC_LOCAL_DIR is set and contains the relative
# path, use it instead of downloading from HF. Lets a Scaleway instance read
# freshly-extracted llms/<eeg>/*.npz locally while still pulling eeg/vision from HF.
LOCAL_DIR = os.environ.get("PLATONIC_LOCAL_DIR")


def fetch(p):
    if LOCAL_DIR:
        local = Path(LOCAL_DIR).expanduser() / p
        if local.exists():
            return str(local)
    # llms/ and the clip4s vision/ are ours → LLM_REPO; eeg/ + EEG-family vision/ → REPO.
    ours = p.startswith("llms/") or (p.startswith("vision/") and p.endswith("__clip4s.npz"))
    repo = LLM_REPO if ours else REPO
    return hf_hub_download(repo, p, repo_type="dataset", token=HF_TOKEN_CACHE)


def load_npz(p):
    return np.load(fetch(p), allow_pickle=True)


# ── EEG model registry ─────────────────────────────────────────────────────────
EEG = {
    "femba":       {"sizes": ["tiny", "base", "large"],  "family": "femba_luna",
                    "fname": lambda s: f"eeg/femba/femba_{s}_tusl_layerwise.npz"},
    "luna":        {"sizes": ["base", "large", "huge"],  "family": "femba_luna",
                    "fname": lambda s: f"eeg/luna/luna_{s}_layerwise.npz"},
    "neurolm":     {"sizes": ["b", "l", "xl"],           "family": "neurolm",
                    "fname": lambda s: f"eeg/neurolm/neurolm_{s}_layerwise.npz"},
    "reve":        {"sizes": ["base", "large"],          "family": "reve",
                    "fname": lambda s: f"eeg/reve/reve_{s}_layerwise.npz"},
    "steegformer": {"sizes": ["small", "base", "large"], "family": "steegformer",
                    "fname": lambda s: f"eeg/steegformer/steegformer_{s}_layerwise.npz"},
}

# EEG window scheme (seconds per window, # windows, captions per window). Mirrors the
# constants in each src/extraction_scripts/extract_<model>.py. Season 7 = 10,800 s.
EEG_WINDOWS = {
    "neurolm":     {"win_sec": 8,  "n_windows": 1350, "caps_per_window": 2},
    "femba":       {"win_sec": 5,  "n_windows": 2160, "caps_per_window": 2},
    "luna":        {"win_sec": 5,  "n_windows": 2160, "caps_per_window": 2},
    "steegformer": {"win_sec": 6,  "n_windows": 1800, "caps_per_window": 2},
    "reve":        {"win_sec": 10, "n_windows": 1080, "caps_per_window": 3},
}

# Models that support the intramodal size-scaling plot (need ≥2 sizes; reve excluded).
INTRA_MODELS = ["femba", "luna", "neurolm", "steegformer"]


# ── Vision model registry ──────────────────────────────────────────────────────
VIDEOMAE_SIZES    = ["base", "large"]
VIDEOMAE_FT_SIZES = ["base", "large", "huge"]
DINOV2_SIZES      = ["small", "base", "large", "giant"]
VJEPA2_SIZES      = ["large", "huge", "giant"]


def vmpath(vs, fam):   return f"vision/videomae/videomae_{vs}__{fam}.npz"
def vmftpath(vs, fam): return f"vision/videomae_finetuned/videomae_ft_kinetics_{vs}__{fam}.npz"
def dnpath(vs, fam):   return f"vision/dinov2/dinov2_{vs}__{fam}.npz"
def vj2path(vs, fam):  return f"vision/vjepa2/vjepa2_{vs}__{fam}.npz"


VISION = {
    "dinov2":      {"sizes": DINOV2_SIZES,      "path": dnpath,   "display": "DINOv2"},
    "videomae":    {"sizes": VIDEOMAE_SIZES,    "path": vmpath,   "display": "VideoMAE"},
    "videomae_ft": {"sizes": VIDEOMAE_FT_SIZES, "path": vmftpath, "display": "VideoMAE-K400"},
    "vjepa2":      {"sizes": VJEPA2_SIZES,      "path": vj2path,  "display": "V-JEPA2"},
}


# ── LLM registry ───────────────────────────────────────────────────────────────
# LLM caption embeddings live under llms/<eeg_model>/<stem>_layerwise.npz and are
# aligned to that EEG model's window scheme (re-extracted per family).
LLM = {
    "bloom":     ["bloomz-560m", "bloomz-1b1", "bloomz-1b7", "bloomz-3b", "bloomz-7b1"],
    "openllama": ["open_llama_3b", "open_llama_7b", "open_llama_13b"],
    "llama":     ["llama-13b", "llama-30b", "llama-65b"],  # "33b" ships as llama-30b
}
LLM_FAMILY_OF = {stem: fam for fam, stems in LLM.items() for stem in stems}


def llm_path(eeg_model, stem):
    return f"llms/{eeg_model}/{stem}_layerwise.npz"


# Plain x-tick labels: family_paramsize (e.g. bloom_560m). No color.
LLM_LABEL = {
    "bloomz-560m": "bloom_560m", "bloomz-1b1": "bloom_1b1", "bloomz-1b7": "bloom_1b7",
    "bloomz-3b": "bloom_3b", "bloomz-7b1": "bloom_7b1",
    "open_llama_3b": "openllama_3b", "open_llama_7b": "openllama_7b",
    "open_llama_13b": "openllama_13b",
    "llama-13b": "llama_13b", "llama-30b": "llama_33b", "llama-65b": "llama_65b",
}


# ── Display metadata ───────────────────────────────────────────────────────────
PARAMS = {
    "femba":       {"tiny": 8,   "base": 48,  "large": 78},
    "luna":        {"base": 7,   "large": 43, "huge": 311},
    "neurolm":     {"b": 254,    "l": 500,    "xl": 1696},
    "steegformer": {"small": 22, "base": 86,  "large": 307},
    "reve":        {"base": 69,  "large": 408},
}
DISPLAY = {"femba": "FEMBA", "luna": "LUNA", "neurolm": "NeuroLM",
           "steegformer": "STEEGFormer", "reve": "REVE"}


def fmt_size(model, sz):
    return f"{sz} ({PARAMS[model][sz]}M)"


# Color maps (consistent across plots).
EEG_SIZE_COLORS = {
    "femba":       {"tiny": "#c8b400", "base": "#1a9850", "large": "#5b2182"},
    "luna":        {"base": "#c8b400", "large": "#1a9850", "huge": "#5b2182"},
    "neurolm":     {"b": "#c8b400", "l": "#1a9850", "xl": "#5b2182"},
    "steegformer": {"small": "#c8b400", "base": "#1a9850", "large": "#5b2182"},
    "reve":        {"base": "#1a9850", "large": "#5b2182"},
}
LLM_FAMILY_COLORS = {"bloom": "#1b9e77", "openllama": "#d95f02", "llama": "#7570b3"}
VISION_ARCH_COLORS = {"dinov2": "#1b9e77", "videomae": "#d95f02",
                      "videomae_ft": "#e7298a", "vjepa2": "#7570b3"}
vm_colors   = {"base": "steelblue", "large": "seagreen"}
vmft_colors = {"base": "firebrick", "large": "darkorchid", "huge": "darkorange"}
dn_colors   = {"small": "steelblue", "base": "seagreen",
               "large": "darkorange", "giant": "mediumpurple"}
vj2_colors  = {"large": "steelblue", "huge": "seagreen", "giant": "darkorange"}
VISION_SIZE_COLORS = {"dinov2": dn_colors, "videomae": vm_colors,
                      "videomae_ft": vmft_colors, "vjepa2": vj2_colors}


# ── Performance scores used as plot x-axes ─────────────────────────────────────
# LLM x-axis = 1-bits-per-byte over 4M OpenWebText tokens (Huh et al. 2024, the PRH
# protocol). These exact numbers are not published in a table nor shipped by the
# platonic-rep repo — they are computed. Run src/extraction_scripts/measure_llm_bpb.py
# to produce tokens/llm_bpb.json; it is loaded here. Until present, plots fall back to
# a log-param proxy (a banner warns) so figures still render.
LLM_PERF = {
    "bloomz-560m": None, "bloomz-1b1": None, "bloomz-1b7": None,
    "bloomz-3b": None, "bloomz-7b1": None,
    "open_llama_3b": None, "open_llama_7b": None, "open_llama_13b": None,
    "llama-13b": None, "llama-30b": None, "llama-65b": None,
}
_BPB_JSON = _PROJECT_ROOT / "tokens" / "llm_bpb.json"
if _BPB_JSON.exists():
    # JSON maps stem -> {"perf": 1-bpb, "bpb": ...}; accept a plain number too.
    for _stem, _v in json.loads(_BPB_JSON.read_text()).items():
        if _stem in LLM_PERF:
            LLM_PERF[_stem] = float(_v["perf"]) if isinstance(_v, dict) else float(_v)

# Param-count proxy (millions) — only used to order/label x when LLM_PERF is missing.
LLM_PARAMS = {
    "bloomz-560m": 560, "bloomz-1b1": 1100, "bloomz-1b7": 1700,
    "bloomz-3b": 3000, "bloomz-7b1": 7100,
    "open_llama_3b": 3000, "open_llama_7b": 7000, "open_llama_13b": 13000,
    "llama-13b": 13000, "llama-30b": 33000, "llama-65b": 65000,
}

# Vision param counts (millions) for x-tick labels (verify vs model cards).
VISION_PARAMS = {
    ("dinov2", "small"): 22, ("dinov2", "base"): 86,
    ("dinov2", "large"): 300, ("dinov2", "giant"): 1100,
    ("videomae", "base"): 87, ("videomae", "large"): 305,
    ("videomae_ft", "base"): 87, ("videomae_ft", "large"): 305, ("videomae_ft", "huge"): 633,
    ("vjepa2", "large"): 300, ("vjepa2", "huge"): 600, ("vjepa2", "giant"): 1000,
}


def _fmt_params(m):
    return f"{m/1000:.1f}B" if m >= 1000 else f"{m}M"


def vision_label(arch, size):
    p = VISION_PARAMS.get((arch, size))
    base = f"{VISION[arch]['display']}_{size}"
    return f"{base}_{_fmt_params(p)}" if p is not None else base

# Video x-axis = Kinetics-400 top-1 (action recognition). See plan VIDEO_PERF table.
#   VideoMAE / VideoMAE-ft: arXiv:2203.12602 (HF MCG-NJU/videomae-*-finetuned-kinetics)
#   V-JEPA2 giant, DINOv2 giant: arXiv:2506.09985 (V-JEPA 2 frozen-probe / baseline)
# None = no published K400 number → excluded from the perf-ordered video plots.
VIDEO_PERF = {
    ("videomae", "base"): 80.9, ("videomae", "large"): 84.7,
    ("videomae_ft", "base"): 80.9, ("videomae_ft", "large"): 84.7,
    ("videomae_ft", "huge"): 86.6,
    ("vjepa2", "large"): None, ("vjepa2", "huge"): None,  # TODO arXiv:2506.09985
    ("vjepa2", "giant"): 86.6,
    ("dinov2", "small"): None, ("dinov2", "base"): None,
    ("dinov2", "large"): None, ("dinov2", "giant"): 83.4,
}


def llm_x(stem):
    """x-position for an LLM: 1-BPB perf if available, else log10(params) proxy."""
    v = LLM_PERF.get(stem)
    if v is not None:
        return float(v)
    if not llm_x_is_proxy():
        raise ValueError(
            f"{stem}: no measured 1-BPB but other LLMs have one — would mix a log-param "
            f"proxy (~{np.log10(LLM_PARAMS[stem]):.1f}) with measured 1-BPB (~0-0.3) on the "
            f"same axis. Run measure_llm_bpb.py for {stem}, or drop it from the plot.")
    return float(np.log10(LLM_PARAMS[stem]))


def llm_x_is_proxy():
    return all(v is None for v in LLM_PERF.values())


def llm_axis_tag():
    """Short tag stating which LLM x-axis is active (for figure titles)."""
    return "[x = log-param proxy — run measure_llm_bpb.py]" if llm_x_is_proxy() \
        else "[x = measured 1−BPB, 4M OpenWebText]"


# ── Hyperparameters ────────────────────────────────────────────────────────────
K_MKNN_DEFAULT = 5
K_PERM_DEFAULT = 200
ALPHA_DEFAULT  = 0.05


# ── kNN / mKNN primitives ──────────────────────────────────────────────────────
def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def knn_1d(z, k):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(z)
    _, idx = nn.kneighbors(z)
    return idx[:, 1:].astype(np.int32)


def knn_1d_excl(z, k, excl):
    """Cosine kNN with temporal exclusion: masks ±excl windows around each query."""
    W = len(z)
    zn = l2(z)
    sim = zn @ zn.T
    rows = np.arange(W)[:, None]
    cols = np.clip(rows + np.arange(-excl, excl + 1)[None, :], 0, W - 1)
    sim[rows, cols] = -2.0
    return np.argsort(-sim, axis=1)[:, :k].astype(np.int32)


def mknn_1d(ia, ib):
    return float((ia[:, :, None] == ib[:, None, :]).any(-1).mean())


def precompute_knn(emb, k):
    """Layerwise EEG: emb (n_layers, W, S, D) -> (n_layers, S, W, k)."""
    n_layers, W, Ns, _ = emb.shape
    idx = np.zeros((n_layers, Ns, W, k), dtype=np.int32)
    for l in range(n_layers):
        for s in range(Ns):
            idx[l, s] = knn_1d(l2(emb[l, :, s, :]), k)
    return idx


def precompute_knn_layers(emb, k):
    """Layerwise vision/LLM: emb (n_layers, W, D) -> (n_layers, W, k).
    A stale 2-D (W, D) array is promoted to a single layer for back-compat."""
    emb = np.asarray(emb)
    if emb.ndim == 2:
        emb = emb[None]
    n_layers, W, _ = emb.shape
    idx = np.zeros((n_layers, W, k), dtype=np.int32)
    for l in range(n_layers):
        idx[l] = knn_1d(l2(emb[l]), k)
    return idx


def max_mknn_over_layers(nn_a, nn_b):
    """Max mKNN over all (a-layer, b-layer) pairs. nn_a (LA,W,k), nn_b (LB,W,k)."""
    LA = nn_a.shape[0]
    LB = nn_b.shape[0]
    return max(mknn_1d(nn_a[la], nn_b[lb]) for la in range(LA) for lb in range(LB))


# ── Aristotelian (i.i.d. permutation) calibration ──────────────────────────────
def _calibrate(T_obs, T_null, alpha):
    tau = float(np.quantile(T_null, 1 - alpha))
    s_cal = max(T_obs - tau, 0.0) / (1 - tau) if tau < 1 else 0.0
    p_val = float((1 + (np.asarray(T_null) >= T_obs).sum()) / (len(T_null) + 1))
    return tau, s_cal, p_val


def aristotelian_intra_subj_full(nn_a, nn_b, s, K=K_PERM_DEFAULT, seed=42):
    """Intramodal observed score + full permutation-null array, one subject.
    nn_a, nn_b: (n_layers, S, W, k).  Returns (T_obs, T_null[K])."""
    _, _, W, _ = nn_a.shape
    rng = np.random.default_rng(seed)
    a_s = nn_a[:, s]
    b_s = nn_b[:, s]
    T_obs = max_mknn_over_layers(a_s, b_s)
    T_null = np.zeros(K)
    for ki in range(K):
        perm = rng.permutation(W)
        inv = np.argsort(perm)
        b_perm = inv[b_s[:, perm, :]]
        T_null[ki] = max_mknn_over_layers(a_s, b_perm)
    return T_obs, T_null


def aristotelian_intra_subj(nn_a, nn_b, s, K=K_PERM_DEFAULT, alpha=ALPHA_DEFAULT, seed=42):
    """Intramodal, one subject; searches all EEG-layer × EEG-layer pairs.
    nn_a, nn_b: (n_layers, S, W, k)."""
    T_obs, T_null = aristotelian_intra_subj_full(nn_a, nn_b, s, K=K, seed=seed)
    tau, s_cal, p_val = _calibrate(T_obs, T_null, alpha)
    return T_obs, s_cal, tau, p_val


def aristotelian_cross(nn_eeg_layers, nn_vis_layers, K=K_PERM_DEFAULT, alpha=ALPHA_DEFAULT, seed=42):
    """Cross-modal, one subject; searches all EEG-layer × other-layer pairs.
    nn_eeg_layers: (LE, W, k)   nn_vis_layers: (LV, W, k).
    Returns (T_obs, s_cal, tau, p_val, null_mean) — null_mean = i.i.d.-permutation
    chance level (the baseline drawn in intramodal / video_vs_eeg_nonperf)."""
    LE, W, _ = nn_eeg_layers.shape
    rng = np.random.default_rng(seed)
    T_obs = max_mknn_over_layers(nn_eeg_layers, nn_vis_layers)
    T_null = np.zeros(K)
    for ki in range(K):
        perm = rng.permutation(W)
        inv = np.argsort(perm)
        v_perm = inv[nn_vis_layers[:, perm, :]]
        T_null[ki] = max_mknn_over_layers(nn_eeg_layers, v_perm)
    tau, s_cal, p_val = _calibrate(T_obs, T_null, alpha)
    return T_obs, s_cal, tau, p_val, float(T_null.mean())


def run_cross_modal(eeg_model, vision_arch_label, vision_sizes, vfname, k_mknn,
                    k_perm=K_PERM_DEFAULT, alpha=ALPHA_DEFAULT, verbose=True):
    """Generic cross-modal driver. `vfname(size, family)` -> HF path.
    Returns results[(esz, vs)] = dict(T_obs, s_cal, tau, p), each (S,)."""
    fam = EEG[eeg_model]["family"]
    eeg_sizes = EEG[eeg_model]["sizes"]
    vis_nn = {}
    for vs in vision_sizes:
        vemb = load_npz(vfname(vs, fam))["embeddings"].astype(np.float32)
        if verbose:
            print(f"  {vision_arch_label}-{vs} ({fam}): {vemb.shape}")
        vis_nn[vs] = precompute_knn_layers(vemb, k_mknn)
    results = {}
    for esz in eeg_sizes:
        emb = load_npz(EEG[eeg_model]["fname"](esz))["embeddings"].astype(np.float32)
        n_layers, W, S, _ = emb.shape
        if verbose:
            print(f"  {eeg_model}-{esz}: {emb.shape}  precomputing EEG kNN...")
        nn_eeg = precompute_knn(emb, k_mknn)
        for vs in vision_sizes:
            T_l, sc_l, tau_l, p_l, nm_l = [], [], [], [], []
            for s in range(S):
                T, sc, tau, p, nm = aristotelian_cross(
                    nn_eeg[:, s], vis_nn[vs], K=k_perm, alpha=alpha, seed=42 + s)
                T_l.append(T); sc_l.append(sc); tau_l.append(tau); p_l.append(p); nm_l.append(nm)
            results[(esz, vs)] = dict(
                T_obs=np.array(T_l), s_cal=np.array(sc_l),
                tau=np.array(tau_l), p=np.array(p_l), null_mean=np.array(nm_l))
    return results


# ── Temporal-confound-aware nulls ──────────────────────────────────────────────
def _block_perm(W, block_len, rng):
    """A window permutation that keeps contiguous blocks of `block_len` intact and
    only shuffles the block ORDER. Preserves within-block temporal autocorrelation."""
    n_blocks = int(np.ceil(W / block_len))
    blocks = [np.arange(i * block_len, min((i + 1) * block_len, W)) for i in range(n_blocks)]
    order = rng.permutation(n_blocks)
    return np.concatenate([blocks[i] for i in order])


def block_perm_null(nn_eeg_layers, nn_vis_layers, block_len, K=K_PERM_DEFAULT,
                    alpha=ALPHA_DEFAULT, seed=42):
    """Cross-modal calibration whose null PRESERVES local temporal autocorrelation
    (block-shuffled order) instead of destroying it (i.i.d. permutation). One subject.
    Returns (T_obs, s_cal, tau, p, T_null)."""
    LE, W, _ = nn_eeg_layers.shape
    rng = np.random.default_rng(seed)
    T_obs = max_mknn_over_layers(nn_eeg_layers, nn_vis_layers)
    T_null = np.zeros(K)
    for ki in range(K):
        perm = _block_perm(W, block_len, rng)
        inv = np.argsort(perm)
        v_perm = inv[nn_vis_layers[:, perm, :]]
        T_null[ki] = max_mknn_over_layers(nn_eeg_layers, v_perm)
    tau, s_cal, p_val = _calibrate(T_obs, T_null, alpha)
    return T_obs, s_cal, tau, p_val, T_null


def shift_mknn(eeg_emb_subj, vis_emb, k, d):
    """Uncalibrated max-over-layer-pairs mKNN with the vision/LLM side shifted by d
    windows: pairs EEG[i] with OTHER[i-d] on the overlap region, recomputing kNN on
    the truncated arrays (neighbours change under truncation).
      eeg_emb_subj : (LE, W, D)    vis_emb : (LV, W, D)
    """
    LE, W, _ = eeg_emb_subj.shape
    LV = vis_emb.shape[0]
    i_eeg_lo = max(0, d); i_eeg_hi = W + min(0, d)
    i_oth_lo = -min(0, d); i_oth_hi = W - max(0, d)
    nn_eeg = np.stack([knn_1d(l2(eeg_emb_subj[le, i_eeg_lo:i_eeg_hi]), k) for le in range(LE)])
    nn_oth = np.stack([knn_1d(l2(vis_emb[lv, i_oth_lo:i_oth_hi]), k) for lv in range(LV)])
    return max_mknn_over_layers(nn_eeg, nn_oth)


def shift_null(eeg_emb, vis_emb, k, shifts):
    """Shift-null across subjects. eeg_emb (LE,W,S,D), vis_emb (LV,W,D).
    Returns dict d -> np.array of shape (S,) of uncalibrated max-mKNN, plus the
    p-value of the unshifted (d=0) score vs the distribution over shifts (subject mean)."""
    LE, W, S, _ = eeg_emb.shape
    out = {}
    for d in shifts:
        out[d] = np.array([shift_mknn(eeg_emb[:, :, s, :], vis_emb, k, d) for s in range(S)])
    if 0 in out:
        obs = out[0].mean()
        dist = np.array([out[d].mean() for d in shifts])
        p = float((1 + (dist >= obs).sum()) / (len(dist) + 1))  # includes d=0 itself
    else:
        p = np.nan
    return out, p


# ── Significance: shift-null curve + block-permutation null (generic) ──────────
def significance_for_size(eeg_emb, other_emb, k, shifts, block_len,
                          k_perm=K_PERM_DEFAULT, seed0=42):
    """Run both temporal-confound nulls for one EEG size vs one 'other' modality.
      eeg_emb   : (LE, W, S, D)     other_emb : (LV, W, D)
    Returns a dict with the shift curve (subject mean/std), its p-value, and the
    pooled block-permutation null + its p-value."""
    per_d, shift_p = shift_null(eeg_emb, other_emb, k, shifts)
    shift_mean = np.array([per_d[d].mean() for d in shifts])
    shift_std  = np.array([per_d[d].std() for d in shifts])
    S = eeg_emb.shape[2]
    nn_oth = precompute_knn_layers(other_emb, k)
    T_obs_l, tau_l, null_pool, bp_l = [], [], [], []
    for s in range(S):
        nn_eeg_s = precompute_knn(eeg_emb[:, :, s:s + 1, :], k)[:, 0]   # (LE,W,k)
        T_obs, s_cal, tau, p, T_null = block_perm_null(
            nn_eeg_s, nn_oth, block_len, K=k_perm, seed=seed0 + s)
        T_obs_l.append(T_obs); tau_l.append(tau); bp_l.append(p)
        null_pool.append(T_null)
    return dict(shifts=np.array(shifts), shift_mean=shift_mean, shift_std=shift_std,
                shift_p=shift_p, block_T_obs=np.array(T_obs_l),
                block_tau=np.array(tau_l), block_null=np.concatenate(null_pool),
                block_p=float(np.mean(bp_l)))


def plot_significance(axes2d, titles, sigs):
    """Render the two temporal-confound nulls as a clearly separated 2-row grid.

      axes2d : array shape (2, N).
      row 0  = SHIFT-NULL — alignment vs temporal offset d (a curve; d=0 starred in red).
               Real content alignment makes the d=0 point stand above the shifted ones.
      row 1  = BLOCK-PERMUTATION NULL — histogram of the null mKNN distribution with the
               observed (d=0) value drawn as a red vertical line. Real alignment sits to
               the right of the bulk of the null.
    Both p-values are annotated. (These are two *different* tests of the same question.)
    """
    for j, (title, d) in enumerate(zip(titles, sigs)):
        ax_s, ax_b = axes2d[0][j], axes2d[1][j]

        # row 0 — shift-null curve
        x = d["shifts"]
        ax_s.plot(x, d["shift_mean"], color="#4575b4", lw=1.6, marker="o", ms=3,
                  label="shifted pairing", zorder=2)
        ax_s.fill_between(x, d["shift_mean"] - d["shift_std"], d["shift_mean"] + d["shift_std"],
                          color="#4575b4", alpha=0.15, zorder=1)
        i0 = int(np.where(x == 0)[0][0])
        ax_s.scatter([0], [d["shift_mean"][i0]], color="crimson", s=120, zorder=5,
                     marker="*", label="observed (d=0)")
        ax_s.set_title(f"{title}\nshift-null  p={d['shift_p']:.3f} ({sig_stars(d['shift_p'])})",
                       fontsize=11)
        ax_s.set_xlabel("temporal shift d (windows)", fontsize=9)
        ax_s.grid(alpha=0.3)

        # row 1 — block-permutation null histogram
        nullv = d["block_null"]
        obs = float(np.mean(d["block_T_obs"]))
        ax_b.hist(nullv, bins=30, color="grey", alpha=0.65, label="block-perm null")
        ax_b.axvline(obs, color="crimson", lw=2.2, label="observed (d=0)")
        ax_b.axvline(float(np.quantile(nullv, 0.95)), color="black", ls="--", lw=1.0,
                     label="null 95th pct")
        ax_b.set_title(f"block-perm  p={d['block_p']:.3f} ({sig_stars(d['block_p'])})",
                       fontsize=11)
        ax_b.set_xlabel("mKNN (uncalibrated)", fontsize=9)
        ax_b.grid(alpha=0.3)

    axes2d[0][0].set_ylabel("shift-null: mKNN vs offset", fontsize=10)
    axes2d[1][0].set_ylabel("block-null: count", fontsize=10)
    axes2d[0][-1].legend(fontsize=7)
    axes2d[1][-1].legend(fontsize=7)


def make_significance_fig(titles, sigs):
    """Build the 2×N significance figure and return (fig, axes2d)."""
    n = len(titles)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 8.4), squeeze=False)
    plot_significance(axes, titles, sigs)
    return fig, axes


# ── Small plot helpers ─────────────────────────────────────────────────────────
def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def tight_ylim(ax, lows, highs, pad_frac=0.08):
    lo, hi = float(np.min(lows)), float(np.max(highs))
    pad = max((hi - lo) * pad_frac, 1e-3)
    ax.set_ylim(lo - pad, hi + pad)
