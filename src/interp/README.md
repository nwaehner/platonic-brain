# `src/interp/` — post-hoc interpretability studies

These scripts explain *why* and *how* the cross-modal "Platonic" alignment arises between the
brain foundation models (**EEG FMs**), **vision** models and caption **LLMs**. They take the
*already-extracted* embeddings (no foundation-model inference) and ask three questions:

1. **What content mediates the alignment?** When two models agree on a window's neighbours,
   what do those neighbours share — low-level vision, high-level vision, or semantics?
   → Components 1–2 (+ 4 for captions).
2. **Does stronger alignment imply more feature-decodable representations?**
   → Component 3 (EEG↔vision) and Component 5 (vision↔language).
3. plus four older, independent studies (geometry, stitching, probing, where-in-the-video).

Every analysis runs on the uploaded embeddings; the **only** raw-data access is Component 1,
which reads cached CineBrain video frames + Qwen captions to build interpretable features.

---

## 1. Data: embeddings, grids, captions

### Two HuggingFace datasets
| Repo | Contents | Used by |
|---|---|---|
| **`nitrox639/platonic-embeddings`** | `eeg/<model>/…` (native per-model window grids); `vision/<arch>/…__<eeg_family>.npz` (vision re-extracted on each EEG grid) | EEG-only, EEG∩Vision, EEG∩LLM, Comp 3 |
| **`triniborrell/platonic-embeddings`** | `vision/<arch>/…__clip4s.npz` and `llms/clip4s/<stem>_layerwise.npz` (the **4 s grid**); `llms/<eeg_model>/…` (LLMs on EEG grids) | Vision-only, LLM-only, LLM∩Vision, Comp 5 |

The loaders in `_interp_common.py` (`load_eeg_emb`, `load_vision_grid`, `load_llm_grid`) route
each regime to the right repo/grid automatically.

### Window grids
A "grid" is a window length + count; **cross-modal analyses require both models on the same
grid**, so vision/LLM embeddings are re-extracted per grid.

| Grid | win_sec | #windows | Notes |
|---|---|---|---|
| femba, luna | 5 s | 2160 | share one grid → the `…__femba_luna` vision file is reused for both |
| neurolm | 8 s | 1350 | |
| reve | 10 s | 1080 | |
| steegformer | 6 s | 1800 | |
| **`clip4s`** | 4 s | ~2700 | one window = one Season-7 clip; the LLM/vision 4 s grid |

### Captions
Qwen-2.5-VL-7B caption per 4 s clip (`data/captions-qwen-2.5-vl-7b.json`). On the `clip4s`
grid each window = one caption; on EEG grids the overlapping captions are concatenated
per window (`build_window_texts`).

## 2. Models, families, and the intramodal pairing rule
- **EEG = 5 architectures** (= the ANCOVA `family`): `femba, luna, neurolm, reve, steegformer`
  (14 size variants total; reve has 2 sizes, the rest 3).
- **Vision = 4**: `dinov2, videomae, videomae_ft, vjepa2` (12 variants).
- **LLM = 3**: `bloom, openllama, llama`.
- **Intramodal pairing rule** (one shared definition, used wherever "a model vs its sibling
  size" is needed — Comp 2 intramodal regimes, Comp 3/5 intramodal mKNN): order a model's
  sizes ascending; pair `s_i` with `s_{i+1}` (next larger); the **largest** pairs with the
  **second-largest**. e.g. dinov2 `small→base`, `base→large`, `large→giant`, `giant→large`;
  reve = its single `base↔large` pair.

---

## Component 1 — feature stack  `interp_features.py`
Builds and **caches one interpretable descriptor set per grid** →
`outputs/features/<grid>__features.npz` (a `__smoke` variant for small runs). Three tiers,
all computed from the real image/text:

**Tier 1 — low-level visual (model-free, 12 features).** Per window's frames:
classical OpenCV scalars (`motion` = mean optical-flow magnitude, `spatial_freq` = high-pass
FFT energy, `luminance`, `contrast`, `edge_density` = Canny) **+ scikit-image**: GLCM/Haralick
(`contrast/homogeneity/energy/correlation`), `lbp_entropy`, image `tex_entropy`
(Shannon), Hasler–Süsstrunk `colorfulness`.

**Tier 2 — high-level visual (CLIP).** CLIP image embedding (`openai/clip-vit-base-patch32`,
mean over ≤4 evenly-spaced frames, L2-normalised) **+ CLIP zero-shot** scores: for each label,
`P(label | window) = softmax_label( e_img · e_text(label) )`. **Decision — the label bank is
NOT hand-curated** (avoids cherry-picking): COCO-80 objects + a scene list + an action list +
the most-frequent **caption-mined** nouns, deduped. Columns are named `clip:<label>`.

**Tier 3 — semantic.** **Empath** validated lexical categories scored on the window's caption
text (`sem:<category>`) **+** a per-window caption **text embedding** (CLIP text encoder, else
**TF-IDF** fallback) **+** per-**sentence** embeddings (kept for Component 4).

**Outputs in the npz:** the named matrix `feat (W, F)` with `feat_names` (prefixed
`low:` / `clip:` / `sem:`) and `feat_tier`; dense `clip_img`, `txt_emb`; and
`sent_text/sent_window/sent_emb`. **Decision — CLIP is optional:** if torch/transformers are
unavailable the tier-2 block is skipped (a warning) and `txt_emb` falls back to TF-IDF, so the
whole pipeline still runs (just without the CLIP features).

---

## Component 2 — neighbourhood variance ratio  `neighbour_enrichment.py`
*What content mediates the alignment.* **Every window is a query — there is no "anchor"
selection.** A **regime = a 2-model pair** on a common grid.

| Regime | Grid | The pair (A, B) |
|---|---|---|
| `eeg_only` | EEG native | adjacent EEG sizes (intramodal), per arch |
| `vision_only` | clip4s | adjacent vision sizes |
| `llm_only` | clip4s | adjacent LLM stems |
| `intersection_eeg_vision` | EEG native | EEG variant × vision variant |
| `intersection_eeg_llm` | EEG native | EEG variant × LLM |
| `intersection_llm_vision` | clip4s | vision variant × LLM |

**Steps (per pair):**
1. Layerwise kNN of each model at **k = 10** (`C.precompute_knn_layers`).
2. `best_layer_pair(nn_A, nn_B)` → the layers `(lA*, lB*)` maximising their mKNN (and the mKNN
   score reported in the title).
3. **Shared neighbours** of every window `w`: `S_w = nn_A[lA*][w] ∩ nn_B[lB*][w]`.
4. For each feature `f` and each window with **`|S_w| ≥ 2`**:
   `ratio_w(f) = var(f over S_w) / var(f over all W windows)` (sample variance, ddof=1).
   **`ratio < 1` ⇒ the shared neighbours are tighter on `f` than the grid as a whole ⇒ `f` is
   the content that locally co-clusters / mediates the alignment.**
5. **Effect** = median ratio over usable windows, `R_f`.
6. **Significance test** (`IC.varratio_test`): a **matched-n permutation null** — for each
   draw, replace every window's neighbour set with a *random set of the same size* and
   recompute the median ratio; one-sided `p_f = P(null ≤ R_f)`; **BH-FDR** across features → `q_f`.
   (The null sits near 1 and absorbs the small-n downward bias, so significance means "tighter
   than equally-sized *random* sets", not merely "<1".)

**Why variance ratio and not a mean shift:** a signed `mean(neighbours) − mean(rest)` averaged
over *all* windows cancels (a bright window's neighbours are bright, a dark window's are dark);
the variance ratio is positive-definite and does not cancel.

**Plot:** three subplots — **low-level visual / high-level visual (CLIP) / semantic** — each a
**step-histogram of `ratio_w` per feature** (one colour per feature, dark = small median),
a **vertical line at ratio = 1**, and a legend; `q<0.05` solid, else dashed. Sits-left-of-1 =
localised.
**Outputs:** `outputs/enrichment/<grid>/varratio_<regime>_<label>.png` +
`outputs/enrichment/enrich_<regime>.npz` (`median_ratio, p, q, feature_names, mKNN`).

**Caveats:** (a) at k=10 the intersection is **sparse on full grids** — windows with `<2`
shared neighbours are dropped, so the script prints "usable windows" per pair; bump the
neighbourhood k if too few. (b) very rare Empath categories give `ratio→0` trivially (all
neighbours lack the category) — the null is also ~0 so they are correctly **not** flagged.

---

## Component 3 — feature linear-probing vs alignment  `feature_linear_probing.py`
*Does alignment predict feature-decodability?* For each **EEG variant × vision model** pair:
1. `best_layer_pair(nn_eeg, nn_vis)` → the pair's best EEG layer `le` and cross-modal mKNN.
2. `X = zscore(l2(eeg_emb[le]))`; predict each feature with **Ridge under contiguous-block CV**
   (`attribute_probing.block_cv_r2`, temporal-leakage-safe) → R² per feature.
3. **Mean R² per tier** over a top-variance feature subset (bounds compute). `--probe {eeg,vision,both}` (default eeg).

**Plot 3a (`probe_cross__a_fit`):** one figure, grid **rows = vision models × cols = feature
tier**; **one point per EEG variant**, colour + marker by **family** (5 archs, sizes share a
marker — no size glyph); **x = mean R², y = cross-modal mKNN**; per-subplot OLS linear fit.

**Plot 3b (`probe_cross__b_ancova`) — family-confound ANCOVA `R² ~ MKNN + C(family)`:**
fitted with `statsmodels.api.OLS` + explicit family dummies (**not** a patsy formula — the
module-global `C` = `_common` shadows patsy's categorical `C()`). Reports the family-adjusted
slope **β_adj**, its p, and **partial η²** (type-II SS for MKNN). Rendered as an
**added-variable (partial-regression) plot**: residualise both on `C(family)` and scatter —
**x = resid R² | family, y = resid mKNN | family** (same orientation as 3a).

**Intramodal variants (`…_intra`):** identical, but y = the **intramodal** mKNN (the adjacent-size
pairing's score). → four figures: {cross, intra} × {fit, ANCOVA}.
**Outputs:** `outputs/probing/probe_*` + `feature_probing.npz`.

## Component 5 — vision↔language probing (4 s)  `vis_lang_probing.py`
Component 3 on the `clip4s` grid for the **vision↔language** pair. **Probe vision**: for each
**vision variant × LLM** pair, the best-aligned **vision** layer is probed → mean R² per tier.
Plot grid **rows = LLMs**, **points = vision variants** (colour/marker by vision family),
x = mean R², y = mKNN(vision, language); same 3a/3b + intramodal (vision adjacent sizes).
Reuses `feature_linear_probing.plot_fit/plot_ancova`.

## Component 4 — caption variance ratio  `caption_similarity.py`
The Component-2 variance ratio applied to the **caption embedding**. Same shared sets `S_w`;
per window `cap_ratio_w = trace(cov(txt_emb[S_w])) / trace(cov(txt_emb[all]))`
(`IC.trace_cov_ratio_test`). For unit vectors `trace(cov) = 1 − ‖centroid‖²`, so **small ratio
⇔ neighbour captions cluster tightly ⇔ high mutual cosine**. One-sided matched-n test (<1),
BH across pairs. **Plot:** single-panel histogram of `cap_ratio_w` with a line at 1 (title:
median, p, q, n). Also saves the top interpretable anchor↔neighbour **sentence pairs** (max
cosine) as a table.
**Outputs:** `outputs/caption_sim/<grid>/capvar_<regime>_<label>.png` + `capvar_<regime>.npz`.

---

## Independent studies (separate, unchanged)
| Script | What it answers |
|---|---|
| **`high_alignment.py`** (1a) | *Which* video fractions carry the alignment — per-window mKNN, Lorenz/Gini concentration, temporal trace, autocorrelation, top-chunk montage. |
| **`intrinsic_dimensionality.py`** (2) | Manifold **geometry** — participation ratio, TwoNN/MLE ID, effective rank vs depth/scale/performance. |
| **`model_stitching.py`** (3) | **Functional** alignment — Procrustes/ridge maps between spaces vs a temporal shift-null. |
| **`attribute_probing.py`** (4) | **What each model encodes** — block-CV linear probes of visual/semantic attributes across all models. |
| `semantic_or_visual_features_DEPRECATED.py` | **Retired** Study 1b (OLS variance-partition of the mKNN) — replaced by Components 1–5. |

## Shared modules
- **`_interp_common.py`** — the hub: bootstraps `alignment_plots/_common` (re-exported as `C`);
  two-repo loaders; Component-1 feature helpers (`skimage_window_features`, `empath_features`,
  CLIP `get_clip/clip_image_embed/clip_text_embed/clip_zeroshot`, `build_label_bank`,
  `text_embed`); neighbourhood stats (`varratio_test`, `trace_cov_ratio_test`,
  `benjamini_hochberg`, `ancova_family`); `intramodal_partner`; and the legacy
  ID/mKNN/caption/frame helpers.
- **`_style.py`** — the shared plot theme.

---

## Environment & running

**CLIP needs torch.** This box is an **Intel/x86_64 mac**, where PyTorch's last wheel is 2.2.2
(transformers 5.x rejects it), so use a pinned venv:

```bash
uv venv .venv-interp --python 3.11
uv pip install --python .venv-interp/bin/python "torch==2.2.2" "transformers==4.49.0" \
    "numpy<2" "scikit-image==0.24.0" scikit-learn scipy statsmodels seaborn pandas \
    opencv-python-headless empath huggingface_hub matplotlib
PY=.venv-interp/bin/python      # no torch → CLIP tier skipped, TF-IDF text fallback (still runs)
```

```bash
# one-command smoke test of Components 1–5 (few windows, tiny label bank)
$PY src/interp/run_interp_smoke.py

# build the feature cache for a grid (an EEG model name, 'clip4s', or 'all')
$PY src/interp/interp_features.py --grid reve
$PY src/interp/interp_features.py --grid clip4s

# the studies (drop --smoke for the full sweep; --k sets the mKNN neighbourhood, default 10)
$PY src/interp/neighbour_enrichment.py            # Comp 2 — 6 regimes
$PY src/interp/feature_linear_probing.py          # Comp 3 — EEG↔vision probing + ANCOVA
$PY src/interp/vis_lang_probing.py                # Comp 5 — vision↔language (4 s)
$PY src/interp/caption_similarity.py              # Comp 4 — caption variance ratio

# independent studies (the repo's base python is fine — no torch needed)
python src/interp/high_alignment.py --eeg-model reve
python src/interp/intrinsic_dimensionality.py --modalities eeg --models reve
python src/interp/model_stitching.py --eeg-ref neurolm
python src/interp/attribute_probing.py --eeg-ref neurolm
```

A HF token is read from `tokens/hf_token.txt` (or `--hf-token`).

## Outputs
Figures + `.npz` are written under
`src/interp/outputs/{features,enrichment,probing,caption_sim}/`. **These are generated/cache
artifacts and are git-ignored** — regenerate them by re-running the scripts.
