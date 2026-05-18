"""
cross_modal_layer_study.py

Layer-vs-layer Aristotelian (permutation-calibrated) MKNN alignment between
each EEG foundation model and each vision foundation model on the HuggingFace
dataset `nitrox639/platonic-embeddings`.

Both sides are layerwise: vision NPZs on HF are now (n_layers, W, D). For each
subject, this computes a (L_eeg, L_vis) calibrated alignment matrix; subject
means become heatmaps.

Outputs (one per (eeg_model, vision_arch) pair, 10 total):
    src/layer_study/outputs/{eeg_model}__{vision_arch}.png
    src/layer_study/outputs/{eeg_model}__{vision_arch}.npz

PNG layout: rows = EEG sizes, cols = vision sizes, each cell a heatmap
    (y = EEG layer with prefix initial, x = vision layer with prefix initial,
     color = calibrated MKNN, subject-averaged). White × marks the peak cell.

NPZ keys (per eeg_size × vis_size):
    s_cal__{eeg_size}__{vis_size}   (L_e, L_v, S)
    T_obs__{eeg_size}__{vis_size}   (L_e, L_v, S)
    tau__{eeg_size}__{vis_size}     (L_e, L_v, S)
    p__{eeg_size}__{vis_size}       (L_e, L_v, S)
    layer_ticks_eeg__{eeg_size}     ['N0', 'N1', ...]
    layer_ticks_vis__{vis_size}     ['D0', 'D1', ...]
Plus scalar metadata: eeg_model, vision_arch, k_mknn, k_perm, alpha.

Usage:
    python cross_modal_layer_study.py                       # all 10 pairs
    python cross_modal_layer_study.py --eeg-model femba --vision-arch dinov2
    python cross_modal_layer_study.py --smoke               # quick test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from huggingface_hub import hf_hub_download


# ── Plot theme ────────────────────────────────────────────────────────────────
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
        # colour-consistent theme
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "grid.color": COLOR,
    }
)


# ── HuggingFace auth ──────────────────────────────────────────────────────────
REPO = 'nitrox639/platonic-embeddings'

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOKEN_CANDIDATES = [
    Path('scripts/hf_token.txt'),                  # CWD-relative (notebook style)
    _PROJECT_ROOT / 'scripts' / 'hf_token.txt',    # project root
    Path.home() / '.cache' / 'huggingface' / 'token',
]


def _resolve_hf_token(cli_token=None):
    if cli_token:
        return cli_token.strip()
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN'].strip()
    for p in _TOKEN_CANDIDATES:
        if p.exists():
            return p.read_text().strip()
    return None


HF_TOKEN_CACHE = None  # set in main() from CLI


def fetch(p):
    return hf_hub_download(REPO, p, repo_type='dataset', token=HF_TOKEN_CACHE)


def load_npz(p):
    return np.load(fetch(p), allow_pickle=True)


# ── Model registry (mirror of analyze_mknn_hf.ipynb) ──────────────────────────
EEG = {
    'femba':       {'sizes': ['tiny', 'base', 'large'],  'family': 'femba_luna',
                    'fname': lambda s: f'eeg/femba/femba_{s}_tusl_layerwise.npz'},
    'luna':        {'sizes': ['base', 'large', 'huge'],  'family': 'femba_luna',
                    'fname': lambda s: f'eeg/luna/luna_{s}_layerwise.npz'},
    'neurolm':     {'sizes': ['b', 'l', 'xl'],           'family': 'neurolm',
                    'fname': lambda s: f'eeg/neurolm/neurolm_{s}_layerwise.npz'},
    'reve':        {'sizes': ['base', 'large'],          'family': 'reve',
                    'fname': lambda s: f'eeg/reve/reve_{s}_layerwise.npz'},
    'steegformer': {'sizes': ['small', 'base', 'large'], 'family': 'steegformer',
                    'fname': lambda s: f'eeg/steegformer/steegformer_{s}_layerwise.npz'},
}
VIDEOMAE_SIZES = ['base', 'large']
DINOV2_SIZES   = ['small', 'base', 'large', 'giant']


def vmpath(vs, fam): return f'vision/videomae/videomae_{vs}__{fam}.npz'
def dnpath(vs, fam): return f'vision/dinov2/dinov2_{vs}__{fam}.npz'


VISION = {
    'dinov2':   {'sizes': DINOV2_SIZES,   'path': dnpath},
    'videomae': {'sizes': VIDEOMAE_SIZES, 'path': vmpath},
}


# ── Display metadata ──────────────────────────────────────────────────────────
PARAMS = {
    'femba':       {'tiny': 8,   'base': 48,  'large': 78},
    'luna':        {'base': 7,   'large': 43, 'huge': 311},
    'neurolm':     {'b': 254,    'l': 500,    'xl': 1696},
    'steegformer': {'small': 22, 'base': 86,  'large': 307},
    'reve':        {'base': 69,  'large': 408},
}
DISPLAY = {'femba': 'FEMBA', 'luna': 'LUNA', 'neurolm': 'NeuroLM',
           'steegformer': 'STEEGFormer', 'reve': 'REVE'}
VISION_DISPLAY = {'dinov2': 'DINOv2', 'videomae': 'VideoMAE'}


def fmt_size(model, sz): return f'{sz} ({PARAMS[model][sz]}M)'


vm_colors = {'base': 'steelblue', 'large': 'seagreen'}
dn_colors = {'small': 'steelblue', 'base': 'seagreen',
             'large': 'darkorange', 'giant': 'mediumpurple'}
VISION_COLORS = {'dinov2': dn_colors, 'videomae': vm_colors}


# ── Layer-type initials (from hook architecture, see plan) ────────────────────
LAYER_INITIAL = {
    'femba':       'N',   # LayerNorm after each BiMamba block
    'luna':        'B',   # RotaryTransformerBlock
    'neurolm':     'G',   # GPT-2 block
    'reve':        'F',   # FeedForward (post-block)
    'steegformer': 'B',   # ViT transformer block
}

VISION_LAYER_INITIAL = {
    'dinov2':   'D',      # DINOv2 ViT block
    'videomae': 'V',      # VideoMAE transformer block
}


# ── Hyperparameters ───────────────────────────────────────────────────────────
K_MKNN_DEFAULT = 5
K_PERM_DEFAULT = 200
ALPHA_DEFAULT  = 0.05


# ── kNN / MKNN primitives (verbatim from analyze_mknn_hf.ipynb) ───────────────
def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def knn_1d(z, k):
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute').fit(z)
    _, idx = nn.kneighbors(z)
    return idx[:, 1:].astype(np.int32)


def mknn_1d(ia, ib):
    return float((ia[:, :, None] == ib[:, None, :]).any(-1).mean())


def precompute_knn(emb, k):
    """emb (n_layers, W, S, D) → (n_layers, S, W, k)."""
    n_layers, W, Ns, _ = emb.shape
    idx = np.zeros((n_layers, Ns, W, k), dtype=np.int32)
    for l in range(n_layers):
        for s in range(Ns):
            idx[l, s] = knn_1d(l2(emb[l, :, s, :]), k)
    return idx


def precompute_knn_vision(emb, k):
    """emb (L, W, D) → (L, W, k). Vision has no subject axis."""
    L, W, _ = emb.shape
    idx = np.zeros((L, W, k), dtype=np.int32)
    for l in range(L):
        idx[l] = knn_1d(l2(emb[l]), k)
    return idx


# ── Per-layer-pair Aristotelian cross-modal MKNN ──────────────────────────────
def aristotelian_cross_layer_grid(nn_eeg_layers, nn_vis_layers, K, alpha, seed):
    """
    Calibrated MKNN over every (EEG_layer, vision_layer) pair, one subject.

    nn_eeg_layers : (L_e, W, k) EEG kNN indices for one subject across all EEG layers
    nn_vis_layers : (L_v, W, k) vision kNN indices across all vision layers (no subj axis)

    Returns dict of (L_e, L_v) arrays: T_obs, tau, s_cal, p.

    Compared to analyze_mknn_hf.ipynb's `aristotelian_cross`:
      - no max over layers (returns a full matrix)
      - permutation null is collected per cell → cell-wise tau and p-value
      - vision side is also layerwise here (the HF NPZs were updated)
    """
    L_e, W, _ = nn_eeg_layers.shape
    L_v       = nn_vis_layers.shape[0]
    rng       = np.random.default_rng(seed)

    T_obs = np.zeros((L_e, L_v), dtype=np.float64)
    for i in range(L_e):
        for j in range(L_v):
            T_obs[i, j] = mknn_1d(nn_eeg_layers[i], nn_vis_layers[j])

    T_null = np.zeros((K, L_e, L_v), dtype=np.float64)
    v_perm = np.empty_like(nn_vis_layers)
    for ki in range(K):
        perm = rng.permutation(W)
        inv  = np.argsort(perm)
        for j in range(L_v):
            v_perm[j] = inv[nn_vis_layers[j][perm, :]]
        for i in range(L_e):
            for j in range(L_v):
                T_null[ki, i, j] = mknn_1d(nn_eeg_layers[i], v_perm[j])

    tau   = np.quantile(T_null, 1 - alpha, axis=0)                    # (L_e, L_v)
    s_cal = np.maximum(T_obs - tau, 0.0) / np.maximum(1 - tau, 1e-12)
    p     = (1.0 + (T_null >= T_obs[None]).sum(axis=0)) / (K + 1)

    return {'T_obs': T_obs, 'tau': tau, 's_cal': s_cal, 'p': p}


# ── Compute one (eeg_model, vision_arch) pair ─────────────────────────────────
def compute_pair(eeg_model, vision_arch, k_mknn, k_perm, alpha,
                 eeg_size_filter=None, vis_size_filter=None):
    """
    Returns:
        results: nested dict results[eeg_size][vis_size] = {
            'T_obs' (L_e, L_v, S), 's_cal' (L_e, L_v, S),
            'tau'   (L_e, L_v, S), 'p'    (L_e, L_v, S)
        }
        layer_ticks_eeg: dict eeg_size -> list[str] of length L_e
        layer_ticks_vis: dict vis_size -> list[str] of length L_v
    """
    fam = EEG[eeg_model]['family']
    eeg_sizes = EEG[eeg_model]['sizes']
    vis_sizes = VISION[vision_arch]['sizes']
    vis_path  = VISION[vision_arch]['path']

    if eeg_size_filter is not None:
        eeg_sizes = [s for s in eeg_sizes if s in eeg_size_filter]
    if vis_size_filter is not None:
        vis_sizes = [s for s in vis_sizes if s in vis_size_filter]

    # Load vision kNN (layerwise) once per size
    vis_nn = {}
    layer_ticks_vis = {}
    v_initial = VISION_LAYER_INITIAL[vision_arch]
    for vs in vis_sizes:
        vemb = load_npz(vis_path(vs, fam))['embeddings'].astype(np.float32)
        print(f'  {vision_arch}-{vs} ({fam}): {vemb.shape}')
        if vemb.ndim != 3:
            raise ValueError(f'Expected (L, W, D) vision NPZ, got {vemb.shape} '
                             f'for {vis_path(vs, fam)}')
        vis_nn[vs] = precompute_knn_vision(vemb, k_mknn)   # (L_v, W, k)
        L_v = vis_nn[vs].shape[0]
        layer_ticks_vis[vs] = [f'{v_initial}{i}' for i in range(L_v)]

    results = {}
    layer_ticks_eeg = {}
    e_initial = LAYER_INITIAL[eeg_model]

    for esz in eeg_sizes:
        emb = load_npz(EEG[eeg_model]['fname'](esz))['embeddings'].astype(np.float32)
        L_e, W, S, _ = emb.shape
        print(f'  {eeg_model}-{esz}: {emb.shape}  precomputing EEG kNN...')
        nn_eeg = precompute_knn(emb, k_mknn)               # (L_e, S, W, k)
        layer_ticks_eeg[esz] = [f'{e_initial}{i}' for i in range(L_e)]

        results[esz] = {}
        for vs in vis_sizes:
            L_v = vis_nn[vs].shape[0]
            T_obs = np.zeros((L_e, L_v, S), dtype=np.float32)
            s_cal = np.zeros((L_e, L_v, S), dtype=np.float32)
            tau   = np.zeros((L_e, L_v, S), dtype=np.float32)
            p_val = np.zeros((L_e, L_v, S), dtype=np.float32)
            print(f'    {eeg_model}-{esz} (L={L_e}) x '
                  f'{vision_arch}-{vs} (L={L_v})')
            for s in range(S):
                out = aristotelian_cross_layer_grid(
                    nn_eeg[:, s], vis_nn[vs],
                    K=k_perm, alpha=alpha, seed=42 + s,
                )
                T_obs[..., s] = out['T_obs']
                tau[...,   s] = out['tau']
                s_cal[..., s] = out['s_cal']
                p_val[..., s] = out['p']
                best_e, best_v = np.unravel_index(
                    int(np.argmax(out['s_cal'])), out['s_cal'].shape)
                print(f'      sub-{s+1:04d}: '
                      f'max s_cal={out["s_cal"].max():.4f} '
                      f'@ (eeg L{best_e}, vis L{best_v})')
            results[esz][vs] = {
                'T_obs': T_obs, 's_cal': s_cal, 'tau': tau, 'p': p_val,
            }

    return results, layer_ticks_eeg, layer_ticks_vis, eeg_sizes, vis_sizes


# ── Save NPZ ──────────────────────────────────────────────────────────────────
def save_results_npz(out_path, eeg_model, vision_arch,
                     results, layer_ticks_eeg, layer_ticks_vis,
                     k_mknn, k_perm, alpha):
    payload = {
        'eeg_model':   np.array(eeg_model),
        'vision_arch': np.array(vision_arch),
        'k_mknn':      np.array(k_mknn),
        'k_perm':      np.array(k_perm),
        'alpha':       np.array(alpha),
    }
    for esz, ticks in layer_ticks_eeg.items():
        payload[f'layer_ticks_eeg__{esz}'] = np.array(ticks)
    for vs, ticks in layer_ticks_vis.items():
        payload[f'layer_ticks_vis__{vs}'] = np.array(ticks)
    for esz, by_vs in results.items():
        for vs, arrs in by_vs.items():
            for key in ('T_obs', 's_cal', 'tau', 'p'):
                payload[f'{key}__{esz}__{vs}'] = arrs[key]   # (L_e, L_v, S)
    np.savez(out_path, **payload)
    print(f'  Saved NPZ: {out_path}')


# ── Plot grid: heatmaps, rows = eeg sizes, cols = vision sizes ────────────────
def plot_pair(out_path, eeg_model, vision_arch,
              results, layer_ticks_eeg, layer_ticks_vis,
              eeg_sizes, vis_sizes, k_mknn, k_perm):
    nrows = len(eeg_sizes)
    ncols = len(vis_sizes)

    # Global vmax across all subplots so heatmaps are visually comparable
    vmax = max(
        float(results[esz][vs]['s_cal'].mean(axis=-1).max())
        for esz in eeg_sizes for vs in vis_sizes
    )
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.0, nrows * 4.2),
                             squeeze=False)

    last_im = None
    for r, esz in enumerate(eeg_sizes):
        e_ticks = layer_ticks_eeg[esz]
        L_e = len(e_ticks)
        for c, vs in enumerate(vis_sizes):
            ax = axes[r][c]
            v_ticks = layer_ticks_vis[vs]
            L_v = len(v_ticks)

            s_cal_mean = results[esz][vs]['s_cal'].mean(axis=-1)  # (L_e, L_v)

            im = ax.imshow(s_cal_mean, origin='lower', aspect='auto',
                           cmap='viridis', vmin=0.0, vmax=vmax,
                           interpolation='nearest')
            last_im = im

            ax.set_xticks(np.arange(L_v))
            ax.set_yticks(np.arange(L_e))
            x_rot = 0 if L_v <= 16 else 60
            y_rot = 0
            ax.set_xticklabels(v_ticks, fontsize=8, rotation=x_rot)
            ax.set_yticklabels(e_ticks, fontsize=8, rotation=y_rot)

            ax.set_title(
                f'{DISPLAY[eeg_model]}-{fmt_size(eeg_model, esz)}  ×  '
                f'{VISION_DISPLAY[vision_arch]}-{vs}', fontsize=11,
            )
            if c == 0:
                ax.set_ylabel(f'{DISPLAY[eeg_model]} layer', fontsize=10)
            if r == nrows - 1:
                ax.set_xlabel(f'{VISION_DISPLAY[vision_arch]} layer', fontsize=10)

            # Mark the peak cell
            best_e, best_v = np.unravel_index(int(np.argmax(s_cal_mean)),
                                              s_cal_mean.shape)
            ax.scatter([best_v], [best_e],
                       marker='x', s=60, c='white',
                       linewidths=2, zorder=3)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                            shrink=0.85, pad=0.02)
        cbar.set_label('calibrated MKNN  (s_cal, subj. mean)', fontsize=10)

    fig.suptitle(
        f'{DISPLAY[eeg_model]}  ×  {VISION_DISPLAY[vision_arch]}  '
        f'— layer × layer Aristotelian MKNN  (k={k_mknn}, K_perm={k_perm})',
        fontsize=13,
    )
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved PNG: {out_path}')


# ── CLI driver ────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description='Per-layer Aristotelian MKNN: EEG FMs vs vision FMs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--eeg-model', choices=[*EEG.keys(), 'all'], default='all')
    ap.add_argument('--vision-arch', choices=[*VISION.keys(), 'all'], default='all')
    ap.add_argument('--out-dir', default='src/layer_study/outputs')
    ap.add_argument('--k-mknn', type=int, default=K_MKNN_DEFAULT)
    ap.add_argument('--k-perm', type=int, default=K_PERM_DEFAULT)
    ap.add_argument('--alpha',  type=float, default=ALPHA_DEFAULT)
    ap.add_argument('--smoke', action='store_true',
                    help='Smallest EEG size + one vision size, K_perm=20.')
    ap.add_argument('--hf-token', default=None,
                    help='HuggingFace token (overrides HF_TOKEN env / token files).')
    return ap.parse_args()


def main():
    global HF_TOKEN_CACHE
    args = parse_args()

    HF_TOKEN_CACHE = _resolve_hf_token(args.hf_token)
    if HF_TOKEN_CACHE is None:
        print('WARN: no HF token found. The dataset is gated; downloads will '
              '401 unless you authenticate. Options:\n'
              '  1) huggingface-cli login\n'
              '  2) export HF_TOKEN=...\n'
              '  3) put token in scripts/hf_token.txt\n'
              '  4) pass --hf-token ...')

    eeg_models  = list(EEG.keys())     if args.eeg_model   == 'all' else [args.eeg_model]
    vision_archs = list(VISION.keys()) if args.vision_arch == 'all' else [args.vision_arch]

    k_mknn = args.k_mknn
    k_perm = 20 if args.smoke else args.k_perm
    alpha  = args.alpha

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'EEG models  : {eeg_models}')
    print(f'Vision archs: {vision_archs}')
    print(f'k_mknn={k_mknn}  k_perm={k_perm}  alpha={alpha}')
    print(f'Out dir     : {out_dir}')

    for eeg_model in eeg_models:
        for vision_arch in vision_archs:
            print(f'\n=== {eeg_model}  ×  {vision_arch} ===')

            eeg_filter = vis_filter = None
            if args.smoke:
                eeg_filter = [EEG[eeg_model]['sizes'][0]]
                vis_filter = [VISION[vision_arch]['sizes'][0]]

            results, layer_ticks_eeg, layer_ticks_vis, eeg_sizes, vis_sizes = \
                compute_pair(
                    eeg_model, vision_arch,
                    k_mknn=k_mknn, k_perm=k_perm, alpha=alpha,
                    eeg_size_filter=eeg_filter, vis_size_filter=vis_filter,
                )

            stem = f'{eeg_model}__{vision_arch}'
            if args.smoke:
                stem += '__smoke'
            save_results_npz(out_dir / f'{stem}.npz', eeg_model, vision_arch,
                             results, layer_ticks_eeg, layer_ticks_vis,
                             k_mknn, k_perm, alpha)
            plot_pair(out_dir / f'{stem}.png', eeg_model, vision_arch,
                      results, layer_ticks_eeg, layer_ticks_vis,
                      eeg_sizes, vis_sizes, k_mknn, k_perm)

    print('\nDone.')


if __name__ == '__main__':
    main()
