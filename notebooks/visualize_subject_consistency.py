from pathlib import Path
import os, io, base64, tarfile, functools, gc, ssl
import numpy as np

# Windows corporate-CA workaround
ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault('CURL_CA_BUNDLE', '')
os.environ.setdefault('REQUESTS_CA_BUNDLE', '')
import cv2
from PIL import Image as PILImage
from sklearn.neighbors import NearestNeighbors
from huggingface_hub import hf_hub_download

# ── HuggingFace ───────────────────────────────────────────────────────────────
EMBED_REPO = 'nitrox639/platonic-embeddings'
VIDEO_REPO = 'Fudan-fMRI/CineBrain'

for _tp in (Path('tokens/hf_token.txt'), Path('hf_token.txt'),
            Path.home() / 'hf_token.txt',
            Path('..') / 'tokens' / 'hf_token.txt'):
    if _tp.exists():
        os.environ['HF_TOKEN'] = _tp.read_text().strip()
        break
HF_TOKEN = os.environ.get('HF_TOKEN')
assert HF_TOKEN, 'No HF token found — put it in tokens/hf_token.txt'

def fetch_embed(p):
    return hf_hub_download(EMBED_REPO, p, repo_type='dataset', token=HF_TOKEN,
                           local_files_only=True)

def load_npz(p):
    return np.load(fetch_embed(p), allow_pickle=True)

# ── EEG model registry + best layer per model ────────────────────────────────
EEG = {
    'femba':       {'size': 'large', 'window_s': 5,
                    'fname': lambda s: f'eeg/femba/femba_{s}_tusl_layerwise.npz'},
    'luna':        {'size': 'huge',  'window_s': 5,
                    'fname': lambda s: f'eeg/luna/luna_{s}_layerwise.npz'},
    'neurolm':     {'size': 'xl',    'window_s': 8,
                    'fname': lambda s: f'eeg/neurolm/neurolm_{s}_layerwise.npz'},
    'steegformer': {'size': 'large', 'window_s': 6,
                    'fname': lambda s: f'eeg/steegformer/steegformer_{s}_layerwise.npz'},
    'reve':        {'size': 'large', 'window_s': 10,
                    'fname': lambda s: f'eeg/reve/reve_{s}_layerwise.npz'},
}

BEST_LAYER = {
    'femba':       1,
    'luna':        10,
    'neurolm':     11,
    'steegformer': 23,
    'reve':        21,
}

DISPLAY = {'femba': 'FEMBA', 'luna': 'LUNA', 'neurolm': 'NeuroLM',
           'steegformer': 'STEEGFormer', 'reve': 'REVE'}

# ── Clip parameters ───────────────────────────────────────────────────────────
CLIP_SECONDS = 4.0
SOURCE_FPS   = 8
MAX_CLIP_IDX = 2699
SEASON_END_S = (MAX_CLIP_IDX + 1) * CLIP_SECONDS

# ── Hyperparameters ───────────────────────────────────────────────────────────
K_MKNN   = 5
TOP_N    = 5
GIF_SIZE = 160

# ── kNN helpers ───────────────────────────────────────────────────────────────
def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def knn_1d(z, k):
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine',
                          algorithm='brute').fit(z)
    _, idx = nn.kneighbors(z)
    return idx[:, 1:].astype(np.int32)

def canonical_starts(window_s):
    return np.arange(0.0, SEASON_END_S, float(window_s), dtype=np.float64)

def precompute_knn_per_subject(emb, layer, k):
    S, W = emb.shape[2], emb.shape[1]
    idx = np.zeros((S, W, k), dtype=np.int32)
    for s in range(S):
        idx[s] = knn_1d(l2(emb[layer, :, s, :]), k)
    return idx

def compute_pairwise_score(per_subj_nn):
    S, W, k = per_subj_nn.shape
    scores = np.zeros(W, dtype=np.float32)
    for w in range(W):
        nb_count = {}
        for s in range(S):
            for n in per_subj_nn[s, w].tolist():
                nb_count[n] = nb_count.get(n, 0) + 1
        scores[w] = sum(c * (c - 1) // 2 for c in nb_count.values())
    return scores

# ── Video helpers ─────────────────────────────────────────────────────────────
CB_ROOT     = Path(__file__).resolve().parent.parent
CLIP_CACHE  = CB_ROOT / '.clip_cache'
CLIP_MARKER = CLIP_CACHE / '.extracted'
_CLIP_INDEX: dict = {}

def _build_clip_index(clips_dir):
    global _CLIP_INDEX
    _CLIP_INDEX.clear()
    for p in clips_dir.rglob('*.mp4'):
        digits = ''.join(c for c in p.stem if c.isdigit())
        if digits:
            _CLIP_INDEX[int(digits)] = p
    print(f'  Indexed {len(_CLIP_INDEX)} clips', flush=True)

def ensure_clips():
    if _CLIP_INDEX:
        return
    if CLIP_MARKER.exists():
        _build_clip_index(CLIP_CACHE)
        return
    print('Downloading videos.tar (~2.6 GB)...', flush=True)
    tar_path = Path(hf_hub_download(
        repo_id=VIDEO_REPO, filename='videos.tar',
        repo_type='dataset', token=HF_TOKEN))
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(CLIP_CACHE)
    CLIP_MARKER.touch()
    _build_clip_index(CLIP_CACHE)

@functools.lru_cache(maxsize=256)
def _load_clip_frames(clip_idx):
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
    end_s      = start_s + window_s
    first_clip = int(start_s // CLIP_SECONDS)
    last_clip  = min(int(np.ceil(end_s / CLIP_SECONDS)) - 1, MAX_CLIP_IDX)
    frames = []
    for c in range(first_clip, last_clip + 1):
        clip_frames = _load_clip_frames(c)
        n       = len(clip_frames)
        clip_t0 = c * CLIP_SECONDS
        f0 = max(0, int(np.floor((max(start_s, clip_t0) - clip_t0) / CLIP_SECONDS * n)))
        f1 = min(n, int(np.ceil ((min(end_s,   clip_t0 + CLIP_SECONDS) - clip_t0) / CLIP_SECONDS * n)))
        for frame in clip_frames[f0:f1]:
            frames.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR))
    return frames

def frames_to_gif_bytes(frames, fps=SOURCE_FPS):
    pil = [PILImage.fromarray(f) for f in frames]
    buf = io.BytesIO()
    pil[0].save(buf, format='GIF', save_all=True, append_images=pil[1:],
                duration=int(1000/fps), loop=0, optimize=False)
    return buf.getvalue()

def gif_html(frames, fps=SOURCE_FPS, width=GIF_SIZE, border='none'):
    b64 = base64.b64encode(frames_to_gif_bytes(frames, fps)).decode()
    return (f'<img src="data:image/gif;base64,{b64}" width="{width}" '
            f'style="border:{border};image-rendering:auto;"/>')

# ── Visualization builder ─────────────────────────────────────────────────────
_PALETTE = ['#e74c3c','#2980b9','#27ae60','#f39c12','#8e44ad',
            '#16a085','#d35400','#c0392b','#2ecc71','#f1c40f']

def build_top5_html(model, results, gif_size=GIF_SIZE, fps=SOURCE_FPS):
    res         = results[model]
    per_subj_nn = res['per_subj_nn']
    starts_s    = res['starts_s']
    window_s    = res['window_s']
    scores      = res['scores']
    top5        = res['top5']
    S           = res['S']
    layer       = res['layer']

    def gif_card(w, label, border_color=None, bg='#fff'):
        t   = float(starts_s[w])
        bdr = f'3px solid {border_color}' if border_color else '1px solid #ddd'
        g   = gif_html(get_window_frames(t, float(window_s), size=gif_size),
                       fps=fps, width=gif_size, border=bdr)
        return (f'<div style="text-align:center;background:{bg};border-radius:5px;'
                f'padding:4px;flex-shrink:0;">{g}<br/>'
                f'<span style="font-size:9px;font-family:monospace;">'
                f'{label}<br/>w={w} {int(t//60)}m{int(t%60):02d}s</span></div>')

    html = [
        f'<div style="font-family:sans-serif;max-width:1900px;">'
        f'<h2 style="border-bottom:2px solid #333;padding-bottom:6px;">'
        f'{DISPLAY[model]}-{res["eeg_size"]} &nbsp;|&nbsp; layer <b>{layer}</b>'
        f' &nbsp;|&nbsp; k={K_MKNN} &nbsp;|&nbsp; S={S} sujetos'
        f' &nbsp;|&nbsp; top-{TOP_N} por score Σ C(count,2)'
        f'</h2>'
    ]

    for rank, w in enumerate(top5):
        t_anchor = float(starts_s[w])
        sc       = float(scores[w])

        nb_subjects = {}
        for s in range(S):
            for n in per_subj_nn[s, w].tolist():
                nb_subjects.setdefault(n, []).append(s)

        shared = sorted(
            [(n, subjs) for n, subjs in nb_subjects.items() if len(subjs) >= 2],
            key=lambda x: -len(x[1])
        )
        nb_color = {n: _PALETTE[i % len(_PALETTE)] for i, (n, _) in enumerate(shared)}

        anchor_card = gif_card(w, '<b>ANCLA</b>', border_color='#e6a817', bg='#fffbe6')

        if shared:
            legend_items = ''.join(
                f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:3px;">'
                f'<div style="width:14px;height:14px;background:{nb_color[n]};'
                f'border-radius:3px;flex-shrink:0;"></div>'
                f'<span style="font-size:10px;">w={n} — '
                f'{len(subjs)} sujetos: {[s+1 for s in subjs]}'
                f'&nbsp;&nbsp;C({len(subjs)},2)={len(subjs)*(len(subjs)-1)//2}</span>'
                f'</div>'
                for n, subjs in shared
            )
            legend = (f'<div style="background:#f0f0f0;border-radius:5px;padding:8px;'
                      f'min-width:240px;">'
                      f'<div style="font-size:11px;font-weight:bold;margin-bottom:5px;">'
                      f'Vecinos compartidos</div>{legend_items}</div>')
        else:
            legend = ('<div style="font-size:11px;color:#888;padding:8px;">'
                      'Sin vecinos compartidos</div>')

        subj_rows = ''
        for s in range(S):
            cards = ''
            for n in per_subj_nn[s, w].tolist():
                col = nb_color.get(n)
                lbl = (f'S{[sx+1 for sx in nb_subjects[n]]}'.replace(' ', '')
                       if col else f'solo S{s+1}')
                cards += gif_card(n, lbl, border_color=col)
            subj_rows += (
                f'<div style="margin-bottom:8px;">'
                f'<span style="font-size:11px;font-weight:bold;color:#333;">'
                f'Sujeto {s+1}</span>'
                f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:3px;">'
                f'{cards}</div></div>'
            )

        html.append(
            f'<div style="margin:16px 0;padding:12px;background:#f8f8f8;'
            f'border-radius:6px;border:1px solid #ddd;">'
            f'<h3 style="margin:0 0 10px 0;font-size:13px;">'
            f'#{rank+1} &nbsp; w=<b>{w}</b> &nbsp;|&nbsp; '
            f't=<b>{t_anchor:.1f}s</b> ({int(t_anchor//60)}m{int(t_anchor%60):02d}s)'
            f' &nbsp;|&nbsp; score = <b>{sc:.0f}</b>'
            f'</h3>'
            f'<div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:12px;">'
            f'{anchor_card}{legend}</div>'
            f'<hr style="border:none;border-top:1px solid #e0e0e0;margin:8px 0;"/>'
            f'{subj_rows}</div>'
        )

    html.append('</div>')
    return ''.join(html)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ensure_clips()

    results = {}
    for model, cfg in EEG.items():
        eeg_size = cfg['size']
        window_s = cfg['window_s']
        layer    = BEST_LAYER[model]
        starts_s = canonical_starts(window_s)
        W        = len(starts_s)

        print(f"\n{'='*56}", flush=True)
        print(f"  {DISPLAY[model]}-{eeg_size} | layer {layer} | W={W}", flush=True)
        print(f"  Loading EEG: {cfg['fname'](eeg_size)} ...", flush=True)

        eeg_emb = load_npz(cfg['fname'](eeg_size))['embeddings'].astype(np.float32)
        S = eeg_emb.shape[2]
        print(f"  EEG loaded: {eeg_emb.shape} ({eeg_emb.nbytes/1e9:.2f} GB)", flush=True)

        assert layer < eeg_emb.shape[0], \
            f"Layer {layer} out of range for {model} ({eeg_emb.shape[0]} layers)"

        print(f"  Per-subject kNN at layer {layer} ...", flush=True)
        per_subj_nn = precompute_knn_per_subject(eeg_emb, layer, K_MKNN)
        del eeg_emb; gc.collect()

        print(f"  Computing pairwise score ...", flush=True)
        scores = compute_pairwise_score(per_subj_nn)
        top5 = np.argsort(-scores)[:TOP_N]
        print(f"  top-{TOP_N} scores: {scores[top5].tolist()} | w={top5.tolist()}", flush=True)

        results[model] = dict(
            eeg_size=eeg_size, window_s=window_s, layer=layer,
            starts_s=starts_s, per_subj_nn=per_subj_nn,
            scores=scores, top5=top5, W=W, S=S,
        )

    print('\n=== All models ready. Building HTMLs... ===', flush=True)

    OUT_DIR = Path(__file__).parent / 'subject_consistency_html'
    OUT_DIR.mkdir(exist_ok=True)

    for model in EEG:
        print(f"\n  Rendering {DISPLAY[model]} ...", flush=True)
        inner = build_top5_html(model, results)
        layer = results[model]['layer']
        full = (
            f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{DISPLAY[model]}-{results[model]["eeg_size"]} top-{TOP_N}</title>'
            f'</head><body style="margin:20px;background:#fff;">{inner}</body></html>'
        )
        out_path = OUT_DIR / f'top5_{model}_{results[model]["eeg_size"]}_layer{layer}.html'
        out_path.write_text(full, encoding='utf-8')
        print(f'  Saved {out_path}  ({len(full)/1024:.0f} KB)', flush=True)

    print(f'\nAll {len(EEG)} HTMLs saved in {OUT_DIR.resolve()}', flush=True)
