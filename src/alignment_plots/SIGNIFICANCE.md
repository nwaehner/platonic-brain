# Interpreting the significance figures

All the alignment scripts measure **mKNN**: for each time window, the fraction of its
*k* nearest neighbours in modality A's embedding space that are also neighbours in
modality B's space (max over layer pairs). The question every null below answers is the
same:

> Is the observed window-to-window alignment **more than what temporal smoothness alone
> would produce?**

This matters because in naturalistic movie data the window index *is* time. Window *t* is
a neighbour of *t±1* in **both** modalities simply because each signal is a smooth
function of time. So a high mKNN can reflect a shared **time axis** ("when") rather than a
shared **content geometry** ("what is happening") — and the Platonic Representation
Hypothesis is a claim about content, not time. The three nulls separate the two.

## (a) Aristotelian i.i.d. permutation — the calibration / baseline
Shuffle **all** windows independently and recompute mKNN → a null distribution. The
calibrated score is `s_cal = (T_obs − τ) / (1 − τ)`, with `τ` = 95th percentile of the
null. **Fast, but it destroys temporal order**, so when both modalities are temporally
smooth the null sits too low and significance is *over-stated*.

- This is the **grey baseline band** (null mean → 95th pct) in `intramodal.py` and
  `video_vs_eeg_nonperf.py`. The gap between an observed curve and the band is the
  calibrated signal.

## (b) Block-permutation null — bottom (histogram) row of the significance figures
Shuffle **contiguous time blocks** of length `block_len` instead of single windows. This
**preserves local autocorrelation** inside each block, so the null already contains the
temporal-smoothness structure. If the observed mKNN still lands to the **right** of this
null (p < 0.05), the alignment is **beyond** temporal smoothness — evidence of shared
*content*.

- Read it as: red line (observed) vs grey histogram (null); dashed line = null 95th pct.

## (c) Shift-null — top (curve) row of the significance figures
Re-pair the two modalities at temporal offsets `d` (e.g. LLM[i] vs EEG[i−d]) and recompute
mKNN at each `d`. If alignment is purely temporal-structural, **every offset looks the
same**; if it is content-specific, the true pairing **`d = 0` (red star) stands above** the
shifted ones. `p` = fraction of shifts with mKNN ≥ the `d = 0` value.

## How to read one panel (column)
- **`d = 0` star clearly above the shifted curve** *and* **observed line to the right of the
  block-null bulk** ⇒ genuine **content** alignment.
- **`d = 0` inside the shifted band** *and* **observed inside the block null** ⇒ the apparent
  alignment is the **temporal-autocorrelation confound**, not content.

The two rows are *different tests of the same question*; agreement between them is the
strong result. (With `--smoke` the shift grid is sparse — use a full run for a smooth
curve and a fine p-value.)

## A note on matching windows (what is actually being compared)
Embeddings are compared **index-by-index** at the chosen window granularity, not at 4 s
unless you ask for it:
- `language_vs_eeg` / `video_vs_eeg`: the EEG model's window (e.g. NeuroLM = 8 s, 1350
  windows). Each window aggregates the overlapping 4 s clips (video frames) / 4 s captions
  (text) and is matched to the same-span EEG window.
- `video_vs_language`: a shared windowing set by `--window-scheme`. The default **`clip4s`**
  uses the native 4 s clips (1 caption ↔ 1 clip, 2700 windows, no aggregation on either
  side); passing an EEG model name instead reuses that family's tiling.
