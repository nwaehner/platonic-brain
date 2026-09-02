"""
compute_bert_semantic.py — BERT sentence embeddings of each EEG window's captions.

This builds the target for the new `semantic` feature tier, which replaces the old `mid`
(142 CLIP object-presence) and `high` (39 Empath) visual tiers. The question it poses is:
*can an EEG foundation model's embedding of a window regress the language-model encoding of
what was happening in that window?*

Window → caption alignment is NOT reimplemented: `IC.grid_caption_texts(grid)` already applies
the "include every overlapping caption" rule from `extract_llm_captions.build_window_texts`,
i.e. window w of length L spans [wL, (w+1)L) and takes every 4 s caption c whose [4c, 4c+4)
overlaps it, concatenated with a space. That gives exactly the behaviour asked for — a 6 s
STEEGFormer window gets the 2 clips covering it (one of them fully), a 10 s REVE window gets 3:

    grid          win_sec   windows   captions/window
    femba/luna       5        2160           2
    steegformer      6        1800           2
    neurolm          8        1350           2
    reve            10        1080           3

Encoder: `answerdotai/ModernBERT-base`, the [CLS] position of the last hidden state (768-d).

WHY NOT plain bert-base-uncased: the Qwen-2.5-VL captions are dense (~263 tokens per 4 s clip),
so a window's concatenation runs 524–800 tokens — past BERT's hard 512-token position-embedding
ceiling. Measured share of windows that would be truncated: femba/luna 56%, steegformer 58%,
neurolm 60%, **reve 99%**. That truncation is differential *by grid*, and grid == EEG model, so
it would bias precisely the cross-model comparison these figures exist to make. ModernBERT is
the same encoder-only BERT family with an 8192-token context, so the concatenation is encoded
whole on every grid, with zero truncation.

Output → data/interp/features/<grid>__bert.npz
  cls (W,768) · pca24 (W,24) · pca_evr (24,) · texts (W,) · W · win_sec · model_name

Usage:
  python src/paper_plots/compute_bert_semantic.py                    # all 5 EEG grids
  python src/paper_plots/compute_bert_semantic.py --grids femba,luna # smoke
"""

from __future__ import annotations

import argparse

import numpy as np

import _paper_common as P
from _paper_common import IC

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 2048                # > the 1414-token worst case; ModernBERT allows 8192
N_PCA = 24
FEATURES_DIR = P.PROJECT_ROOT / "data" / "interp" / "features"


def bert_path(grid):
    return FEATURES_DIR / f"{grid}__bert.npz"


def pick_device():
    """cuda > mps > cpu. MPS is ~8x faster than CPU here (0.15 vs 1.20 s/window)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def encode(texts, model_name=MODEL_NAME, batch_size=16, max_length=MAX_LENGTH,
           device=None, verbose=True):
    """[CLS] of the last hidden state for each text → (n, 768).

    Asserts nothing was truncated: silent truncation here would bias the longer-window grids
    (see the module docstring), so it must fail loudly rather than quietly shorten REVE."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    if verbose:
        print(f"    encoding {len(texts)} windows on {device}", flush=True)
    n_trunc, longest = 0, 0
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            raw = [len(tok(t, truncation=False)["input_ids"]) for t in batch]
            longest = max(longest, max(raw))
            n_trunc += sum(r > max_length for r in raw)
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state          # (B, T, 768)
            out.append(hidden[:, 0, :].float().cpu().numpy())  # [CLS] = position 0
            if verbose and (i // batch_size) % 20 == 0:
                print(f"    {i}/{len(texts)}", flush=True)
    if n_trunc:
        raise SystemExit(f"{n_trunc}/{len(texts)} windows exceeded max_length={max_length} "
                         f"(longest {longest}) — raise MAX_LENGTH, do not truncate.")
    print(f"    longest window = {longest} tokens (limit {max_length}, 0 truncated)")
    return np.concatenate(out, axis=0).astype(np.float32)


def build(grid, force=False):
    path = bert_path(grid)
    if path.exists() and not force:
        print(f"[{grid}] cached → {path.name}")
        return

    texts = IC.grid_caption_texts(grid)
    W = len(texts)
    n_tok = [len(t.split()) for t in texts]
    print(f"[{grid}] {W} windows × {IC.grid_win_sec(grid)}s  "
          f"({np.mean(n_tok):.0f} words/window, {min(n_tok)}–{max(n_tok)})")

    cls = encode(texts)

    # PCA on the grid's own windows — used as the interpretable-ish "features" for the
    # disaggregated panels, since raw BERT dimensions carry no standalone meaning.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(N_PCA, cls.shape[0], cls.shape[1]), random_state=0)
    comps = pca.fit_transform(cls).astype(np.float32)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, cls=cls, pca24=comps,
        pca_evr=pca.explained_variance_ratio_.astype(np.float32),
        texts=np.array(texts), W=W, win_sec=IC.grid_win_sec(grid), model_name=MODEL_NAME)
    print(f"  → {path.name}  cls{cls.shape}  pca{comps.shape}  "
          f"top-{comps.shape[1]} PCA explains {pca.explained_variance_ratio_.sum():.1%}")


def load(grid):
    """(cls (W,768), pca24 (W,24)) or (None, None) if not built yet."""
    path = bert_path(grid)
    if not path.exists():
        return None, None
    z = np.load(path, allow_pickle=True)
    return z["cls"].astype(np.float64), z["pca24"].astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description="BERT [CLS] semantic targets per EEG window grid.")
    ap.add_argument("--grids", default=None, help="comma list (default: all 5 EEG grids)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    IC.set_token(args.hf_token)

    grids = args.grids.split(",") if args.grids else list(P.C.EEG.keys())
    for g in grids:
        build(g, force=args.force)


if __name__ == "__main__":
    main()
