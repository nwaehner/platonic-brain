# Paper plots — three alignment blocks

Figures for the platonic-brain paper, organised as the three off-diagonal cells of
{same/different architecture} × {same/different modality}. All alignment is
**Aristotelian-calibrated mKNN and CKA** (`s_cal = (T_obs − τ)/(1 − τ)`, permutation null).

| Block | Question | x | y | stat | script |
|---|---|---|---|---|---|
| **(a)** cross-arch, cross-modal (LLM→EEG) | does EEG↔LLM alignment grow with LLM competence? | LLM 1−BPB | calibrated mKNN / CKA, curve per EEG size | **ANCOVA** (group = size): β, p, R² | `block_a.py` |
| **(b)** intra-arch, same-modal (EEG) | as a family scales, do consecutive sizes get more decodable *and* more internally aligned? | mean R² of the size pair | intramodal calibrated mKNN / CKA | **Spearman** ρ, p | `block_b.py` |
| **(c)** cross-modal, per-variant (EEG↔LLM) | controlling for family, does decodability track cross-modal alignment? | ΔR² | Δalignment | **ANCOVA** `R2 ~ align + C(family)`: β, p | `block_c.py` |
| **(size)** alignment vs parameter count | does alignment grow with sheer scale, on either side? | log10 params (LLM, then EEG-FM) | calibrated CKA (row 1) / mKNN (row 2) | **ANCOVA** + permutation p | `block_size.py` |

Feature tiers (blocks b, c): **eeg** = 16 NICE EEG-derived features · **low** = 12 low-level
visual · **semantic** = ModernBERT `[CLS]` encoding (768-d) of the captions covering each
window. The old `mid` (142 CLIP object-presence) and `high` (39 Empath) tiers are retired.

### The `semantic` tier
`compute_bert_semantic.py` takes each EEG window's captions — `IC.grid_caption_texts` applies the
existing "every overlapping 4 s caption, concatenated" rule from
`extract_llm_captions.build_window_texts`, so a 6 s window gets its 2 clips, a 10 s REVE window
its 3 — encodes the concatenation, and stores the `[CLS]` vector. The headline tier R² is the
mean over the 768 dimensions; the per-feature panels disaggregate over the top-24 PCA components
(stored as the parallel tier `semantic_pca`, routed by `P.disagg_tier`).

**Why ModernBERT rather than `bert-base-uncased`:** the Qwen-2.5-VL captions run ~263 tokens per
4 s clip, so a window's concatenation is 524–800 tokens — past BERT's hard 512-token ceiling.
The share of windows that would be truncated varies by grid (femba 56% … **reve 99%**), and grid
== EEG model, so plain BERT would bias exactly the cross-model comparison the figures make.
ModernBERT is the same encoder-only family with an 8192-token context; `encode()` asserts that
nothing was truncated rather than silently shortening.

### Decodability probe: nested subject × time CV
`compute_decodability.py` uses `nested_ridge_r2`, **not** the older `ridge_path_r2`:

- ridge α **and** the embedding layer are chosen on a validation split, never on test (the old
  path maximised both over the test folds — two optimistic biases);
- splits are crossed over **subjects × contiguous time blocks**, 6 Latin-square outer folds, so
  neither the subject nor the stimulus window is shared with training;
- α up to 1e9; per-fold R² and the intercept-only floor are both stored.

**Read R² against `perf_null`, not against zero** — under block CV the chance floor is itself
negative. `_cache/NEGATIVE_R2_DIAGNOSIS.md` has the measurements. Leakage assertions live in
`test_nested_cv.py` (`.venv-interp/bin/python src/paper_plots/test_nested_cv.py`).

## Reproduce

```bash
# full pipeline (needs HF token in tokens/hf_token.txt for the compute steps)
bash src/paper_plots/run_all_paper_plots.sh

# fast sanity on the locally-cached femba+luna EEG
SMOKE=1 bash src/paper_plots/run_all_paper_plots.sh

# plots only — no embeddings, offline, seconds (needs the three _cache/*.npz)
PLOTS_ONLY=1 bash src/paper_plots/run_all_paper_plots.sh
```

### Two-stage design (compute once, plot forever)
The heavy work is three cached artifacts under `data/paper-plots/_cache/`:

- `decodability_allfeatures.npz` — per-variant × per-feature nested-CV R² for the three tiers
  (`compute_decodability.py`). Also carries `perf_folds`, `perf_sd`, `perf_null`, `best_layer`,
  `best_alpha`. Needs `data/interp/features/<grid>__{features,bert}.npz` and the published
  `eeg_features/<grid>__eegfeat.npz`. The NICE tier is now **recomputed** here rather than read
  from `eeg_alignment/summary.npz`, which was produced by the old biased probe.
- `crossmodal_llm_calibrated.npz` — calibrated mKNN **and** CKA for every EEG variant × LLM stem
  (`compute_crossmodal_llm.py`). *This recomputes CKA that the shipped `summary_crossmodal.npz`
  left as NaN.*
- `intramodal_full.npz` — calibrated mKNN + CKA for every consecutive size pair, incl reve
  (`compute_intramodal.py`).

Ship those three `.npz` (upload to `triniborrell/platonic-embeddings`) and any collaborator
reproduces every figure with `PLOTS_ONLY=1` — no model inference, no large downloads.

## Reused code (no metric logic reimplemented)
`_common.py` (EEG registry, size colours, calibration primitives, `llm_x`) ·
`_interp_common.py` (`ancova_family`, `load_eeg_emb`, `savefig`) ·
`eeg_crossmodal_decodability.py` (`_encode`, `_aristotelian_mknn`, `_aristotelian_cka`) ·
`feature_gridsearch.ridge_path_r2`. The shared glue lives in `_paper_common.py`.

## Caveats (see also figure captions)
- **Temporal-autocorrelation confound.** Aristotelian i.i.d.-permutation calibration overstates
  significance on naturalistic-movie EEG (windows are temporally smooth). The repo's
  block-permutation / shift nulls (`_common.significance_for_size`) are the robustness check for
  headline claims.
- **EEG→visual decodability is weak** (tier R² often ≤ 0): EEG carries little low-level visual
  feature information relative to the NICE (eeg) tier — a real result, reflected in block (b)/(c).
  But the sign alone does not establish it: the chance floor is negative too, so compare `perf`
  with `perf_null` and compare variants with each other.
- **`llama-30b/65b`** have no measured 1−BPB → block (a) uses 9 LLM points. `block_size.py` does
  not need 1−BPB, but reads the same cache, so it inherits the same 9 stems.
- **The `eeg` tier's target is the across-subject mean** (only those NICE features are
  published). So its held-out-subject R² is not yet a subject-generalisation result. Rebuilding
  per-subject targets needs a 72 GB CineBrain re-download and a re-run of `eeg_features.py`;
  `nested_ridge_r2` already accepts a `(S, W, F)` target for when that happens.
