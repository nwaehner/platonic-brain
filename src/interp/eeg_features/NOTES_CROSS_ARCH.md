# Notes: cross-architectural EEG alignment — reconciliation & mKNN

Technical notes for `eeg_feature_alignment.py` and `eeg_crossmodal_decodability.py`.

---

## 1. The alignment problem: different models, different clocks

Every EEG foundation model slices Season 7 (10,800 s) into **non-overlapping windows** of
a fixed duration. The duration is baked into each architecture's training procedure and
cannot be changed after the fact.

| Model family | win\_sec | W windows | Who shares this grid |
|---|---|---|---|
| femba / luna | 5 s | 2 160 | femba\_tiny/base/large, luna\_base/large/huge |
| steegformer | 6 s | 1 800 | steegformer\_small/base/large |
| neurolm | 8 s | 1 350 | neurolm\_b/l/xl |
| reve | 10 s | 1 080 | reve\_base/large |

**Intramodal comparison** (two sizes of the same architecture): both share the exact same
grid → mKNN is computed directly, no reconciliation needed.

**Cross-family comparison** (e.g. femba vs reve): `femba-base` produces a tensor of shape
`(L, 2160, D)` while `reve-base` produces `(L, 1080, D)`.  These can't be compared
element-wise.  `to_common()` solves this by projecting both onto a shared 10 s timeline.

---

## 2. `to_common()`: projecting onto 10 s bins

```python
COMMON_WIN = 10       # bin width in seconds
N_COMMON   = 1080     # 10 800 s / 10 = 1 080 bins
```

### Algorithm (lines 76-84 of eeg_feature_alignment.py)

```
centers[w] = (w + 0.5) * win_sec          # centre of window w in seconds
bin[w]     = floor(centers[w] / 10)       # which 10 s bin this window falls into
out[bin[w]] += emb[:, w, :]               # accumulate embeddings
cnt[bin[w]] += 1
out /= cnt                                 # average (handles bins with >1 window)
```

Every model ends up with shape `(L, 1080, D)` — the same 1080 time-bins — and MKNN/CKA
are now well-defined between them.

### Worked example — first 10 bins (0–100 s)

**reve (10 s, W = 1080)**
```
win  center  bin
  0    5.0     0     ← exactly one window per bin; no averaging
  1   15.0     1
  2   25.0     2
  3   35.0     3
  4   45.0     4
  ...
```
reve is the reference grid — it loses nothing.

---

**femba / luna (5 s, W = 2160)**
```
win  center  bin
  0    2.5     0  ┐
  1    7.5     0  ┘ averaged → bin 0
  2   12.5     1  ┐
  3   17.5     1  ┘ averaged → bin 1
  4   22.5     2  ┐
  5   27.5     2  ┘ averaged → bin 2
  ...
```
Exactly 2 windows per bin, so every bin is a straight average of two consecutive
5 s representations.  Fine-grained temporal structure within a 10 s segment is lost.

---

**neurolm (8 s, W = 1350)**  — LCM(8, 10) = 40 s, so the pattern repeats every 40 s:

```
win  center   bin
  0    4.0      0   ← 1 window
  1   12.0      1   ← 1 window
  2   20.0      2   ┐
  3   28.0      2   ┘ averaged (both centres fall in [20, 30))
  4   36.0      3   ← 1 window
  (then pattern repeats)
```
Most bins hold 1 window; every 4th bin (0-indexed bins 2, 6, 10, …) holds 2.

---

**steegformer (6 s, W = 1800)** — LCM(6, 10) = 30 s, repeating every 5 windows / 3 bins:

```
win  center  bin
  0    3.0     0  ┐
  1    9.0     0  ┘ 2 windows
  2   15.0     1     1 window
  3   21.0     2  ┐
  4   27.0     2  ┘ 2 windows
  5   33.0     3  ┐
  6   39.0     3  ┘ 2 windows
  7   45.0     4     1 window
  ...
```
Pattern: 2, 1, 2, 2, 1, 2, 2, 1, … — non-uniform but bounded (always 1 or 2).

### What is lost

The averaging is a low-pass operation: it discards any representation change that occurs
*within* a 10 s block but not *across* blocks.  reve (10 s native) is unaffected.
femba/luna always lose their within-window variability.  neurolm and steegformer lose it
intermittently.  This is the unavoidable price of comparing models at the coarsest common
resolution.

---

## 3. mKNN: definition and the per-model alignment score

### 3a. Mutual k-nearest neighbours (mKNN)

Given two embedding matrices `A` and `B`, each of shape `(W, D)`, and a neighbour count
`k` (default 5):

1. For each window `w`, find its `k` nearest cosine neighbours in space A:
   `NN_A(w) ⊆ {0, …, W−1} \ {w}`, |set| = k.
2. Independently find its `k` nearest neighbours in space B: `NN_B(w)`.
3. The mKNN score for this window is 1 if `NN_A(w) ∩ NN_B(w) ≠ ∅`, else 0.
   More precisely, the code uses the fraction of B-neighbours that also appear in A:

```python
# mknn_1d in _common.py
def mknn_1d(ia, ib):  # ia, ib: (W, k) index arrays
    return float((ia[:, :, None] == ib[:, None, :]).any(-1).mean())
```

This is `(1/W) Σ_w  |NN_A(w) ∩ NN_B(w)| / k` — the mean, over all windows, of the
fraction of shared neighbours.  Values near 0 mean the two spaces disagree on who is
close to whom; values near 1 mean perfect neighbourhood agreement.

### 3b. Best-layer-pair selection

Each model has multiple layers.  We search **all pairs** (layer a from model A, layer b
from model B) and keep the maximum:

```python
# best_layer_pair in _interp_common.py
for la in range(LA):
    for lb in range(LB):
        sc = mknn_1d(nn_a[la], nn_b[lb])
        if sc > best: best = sc
```

This gives a single scalar per (model A, model B) pair.

### 3c. Pairwise matrix → per-model mean alignment score

For n EEG models in the cross-arch study we compute an n×n matrix:

```
MK[i, j] = best_layer_pair mKNN(model_i, model_j)    (i ≠ j)
MK[i, i] = NaN
```

The **per-model alignment score** (x-axis in the scatter plot) is the mean over all
off-diagonal entries in row i:

```python
mean_mknn[i] = nanmean(MK[i, :])
```

This is the quantity described as: *"For each model in our basket we compute the MKNN
scores against every other model and take the mean across comparisons."*

The analogous quantity for CKA replaces mKNN with the linear CKA of Gram matrices:
`CKA(A, B) = <G_A, G_B>_F / (||G_A||_F ||G_B||_F)` where `G_X = X_c X_c^T / ||X_c X_c^T||_F`.

---

## 4. Aristotelian calibration (cross-modal study)

Raw mKNN has a **chance level** that depends on W: with W windows and k neighbours,
a random baseline of ≈ k/W is expected just from overlap by coincidence.  Larger W
(femba: 2160) → lower raw chance level than smaller W (reve: 1080), so raw scores
are not directly comparable across families.

The **aristotelian calibration** removes this bias via a permutation null:

1. Compute `T_obs = best_layer_pair mKNN(nn_EEG, nn_partner)`.
2. Repeat K=100 times: randomly permute the window indices of the partner kNN array,
   recompute mKNN → collect `T_null[0..K-1]`.
3. Estimate the chance threshold: `τ = quantile(T_null, 1 − α)` with α = 0.05.
4. Calibrate: `s_cal = max(T_obs − τ, 0) / (1 − τ)`.

`s_cal = 0` means the observed mKNN is at or below the permutation-chance level.
`s_cal = 1` means perfect above-chance alignment.  Because τ is computed from the
same W, the calibrated score is comparable across all EEG families.

The cross-modal study (`eeg_crossmodal_decodability.py`) uses `s_cal` on both mKNN and
linear CKA (same permutation logic on the Gram matrices: `G_b_perm = G_b[perm, :][:, perm]`).
