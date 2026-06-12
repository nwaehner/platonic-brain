# alignment_plots

Scaling-vs-alignment figures for the platonic-brain project, built on the
Aristotelian (permutation-calibrated) mKNN protocol over the HuggingFace dataset
`nitrox639/platonic-embeddings`. All scripts share [`_common.py`](_common.py)
(theme, HF auth, model registries, kNN + calibration, the null tests, and the
performance scores) and write PNG + NPZ artifacts to `outputs/`.

## Scripts

| Script | What it plots |
|---|---|
| [`language_vs_eeg.py`](language_vs_eeg.py) | EEG × LLM-caption alignment; x = LLM performance (1-BPB OpenWebText). Per-model + combined + per-EEG-size significance. |
| [`video_vs_eeg.py`](video_vs_eeg.py) | EEG × vision alignment; x = Kinetics-400 top-1. Per-model + combined + significance. |
| [`video_vs_language.py`](video_vs_language.py) | Vision × LLM alignment on a shared windowing (default `--window-scheme clip4s` = native 4 s clips); x = LLM performance, one curve per video model. |
| [`video_vs_eeg_nonperf.py`](video_vs_eeg_nonperf.py) | Classic cross-modal view; x = EEG size, one curve per vision size, one figure per vision arch. |
| [`intramodal.py`](intramodal.py) | Within-EEG-model size scaling vs a permutation baseline. |

## Significance / temporal-confound nulls (in `_common.py`)

mKNN can reward a shared *time axis* rather than shared *content* in temporally smooth
movie data. Three nulls disentangle this:

- **i.i.d. permutation** (Aristotelian) — fast calibrated score; over-states significance.
- **block permutation** (`block_perm_null`) — preserves local autocorrelation in the null.
- **shift-null** (`shift_null`) — re-pairs modalities at offsets d; real content alignment
  puts d=0 above the shifted curve.

The significance figures show the shift curve (d=0 starred, top row) and the
block-permutation null histogram (bottom row). **See [SIGNIFICANCE.md](SIGNIFICANCE.md)
for how to read every panel and what each null means.**

## Usage

```bash
# fast sanity runs (cached embeddings; --smoke ⇒ K_perm=20, 5 shifts)
python src/alignment_plots/intramodal.py --smoke --eeg-model neurolm
python src/alignment_plots/language_vs_eeg.py --eeg-model neurolm --smoke
python src/alignment_plots/video_vs_eeg.py --eeg-model neurolm --smoke
python src/alignment_plots/video_vs_language.py --window-scheme neurolm --smoke

# full runs: drop --smoke (proper K_perm and ~30 shifts); --no-significance skips the slow part
python src/alignment_plots/language_vs_eeg.py            # all EEG models, all LLM families
```

HF auth: a token in `tokens/hf_token.txt` (repo root) is picked up automatically, or set
`HF_TOKEN` / pass `--hf-token`.

## Status / data dependencies

- Works today for **NeuroLM × {BLOOM, OpenLLaMA}** and all vision archs (data on HF).
- **LLaMA** and **non-NeuroLM EEG families** need caption re-extraction first
  (see [`../extraction_scripts/extract_llm_captions.py`](../extraction_scripts/extract_llm_captions.py));
  the plotting scripts skip missing `(eeg, model)` pairs with a warning.
- `_common.LLM_PERF` (1-BPB) is unfilled → the LLM x-axis falls back to a log-param proxy
  (a banner warns). Fill it for the true PRH performance axis.
- `_common.VIDEO_PERF` is pre-filled from VideoMAE / V-JEPA 2 papers; V-JEPA2 ViT-L/H K400
  and smaller-DINOv2 K400 are still TODO.
