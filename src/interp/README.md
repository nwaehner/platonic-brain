# `src/interp/` — post-hoc interpretability studies

Four post-hoc studies that explain *why* and *how* the cross-modal "Platonic" alignment
arises between the brain foundation models (EEG FMs), vision models and caption LLMs on the
`nitrox639/platonic-embeddings` dataset. Each study is a runnable script that saves result
`.npz` files **and** publication-style `.png` figures to `outputs/`.

Everything runs on the **uploaded embeddings only — no foundation-model inference**. The one
exception is the *visual axis* of studies 1b/4, which is computed with **classical, model-free
OpenCV/numpy** features over the cached CineBrain frames (no CNN). All cross-window analyses
control for the repo's known temporal-autocorrelation confound (block CV, shift-nulls,
partialling out time).

## Shared modules
- **`_style.py`** — the exact serif/black 16×12 plot theme (applied after `_common`).
- **`_interp_common.py`** — the hub: bootstraps `alignment_plots/_common` (re-exported as `C`),
  ID estimators, `mknn_per_window`, caption loading + window-alignment, classical visual
  features, CineBrain frame access, model enumeration over the registries, HF loaders with a
  4 s-family fallback, stats helpers (gini/Lorenz, OLS R², commonality variance partition,
  partial correlation), and NPZ/PNG save helpers.

## Studies
| Script | What it answers |
|---|---|
| **`high_alignment.py`** (1a) | *Which* fractions of the video carry the alignment — per-window mKNN, Lorenz/Gini concentration, temporal trace, autocorrelation, cross-model agreement, top-chunk montage. |
| **`semantic_or_visual_features.py`** (1b) | Is the alignment driven by *visual* or *semantic* content — variance partitioning of per-window mKNN over model-free visual features vs caption semantics (+ time), partial correlations, a visual→semantic gradient, and a neighbour-content test. |
| **`intrinsic_dimensionality.py`** (2) | The *geometry* of every model's manifold — linear (participation ratio) and non-linear (TwoNN, MLE) ID + effective rank, vs depth / scale / performance, cross-modality, eigenspectrum, and per-window local-ID agreement. |
| **`model_stitching.py`** (3) | *Functional* alignment — fit orthogonal-Procrustes and ridge maps between latent spaces, scored on held-out windows vs a temporal shift-null; model×model and layer×layer R². |
| **`attribute_probing.py`** (4) | *What each model encodes* — linear probes (block-CV) decoding visual vs semantic attributes from every model; decodability heatmap, vs depth, vs scale. |

## Window grids (important)
Vision/LLM embeddings are re-extracted on each EEG model's window grid, so **cross-modal**
studies operate on a fixed grid:
- 1a/1b use each EEG model's own family grid (vision must match the EEG window scheme).
- 3/4 place every model on one shared grid via `--eeg-ref` (its family fixes the vision family
  and LLM caption directory). Run once per `--eeg-ref` to cover all families.
- Study 2 is per-model (no shared grid) and defaults to the 4 s `clip4s` family for vision/LLM,
  falling back to `femba_luna`/`neurolm` until the 4 s vision upload lands.

## Running
Requires the conda env with `sklearn`, `huggingface_hub`, `matplotlib`, `cv2`, `scipy`
(the repo's `python`). A HF token is read from `tokens/hf_token.txt` (or `--hf-token`).

```bash
# Study 2 — fast, embeddings-only
python src/interp/intrinsic_dimensionality.py --modalities eeg --models reve --no-local

# Study 1a / 1b — one EEG family (1b also needs cached CineBrain frames)
python src/interp/high_alignment.py --eeg-model reve --vision-arch videomae
python src/interp/semantic_or_visual_features.py --eeg-model reve

# Study 3 / 4 — one shared grid
python src/interp/model_stitching.py --eeg-ref neurolm
python src/interp/attribute_probing.py --eeg-ref neurolm

# Full grids: drop the restricting flags (heavy). --all-sizes for study 3.
```

Outputs (figures + `.npz`) are written to `src/interp/outputs/`.
