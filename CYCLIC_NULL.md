# The cyclic null: null-calibration for temporally correlated samples

A specialisation of the null-calibration framework of *"Revisiting the Platonic
Representation Hypothesis: An Aristotelian View"* (`aristotelian.pdf`) to data whose
samples are **windows of a continuous recording** rather than i.i.d. draws.

---

## 1. What the paper does

The paper shows that representational similarity metrics are **confounded by network
scale**: width inflates the null baseline through high-dimensional finite-sample effects,
and depth inflates it because reporting a max over layer pairs is a selection effect —

> *"the expected maximum of independent draws exceeds the mean. This inflation grows with
> the number of comparisons, so deeper models can appear more aligned simply because more
> layer pairs are compared."*

**Scalar calibration.** H₀ is *"the absence of a relationship between X and Y beyond
their marginal statistics"*, operationalised through a **permutation group Πₙ acting on
sample indices**: draw π ~ Unif(Πₙ) independently of (X, Y) and evaluate s(X, π(Y)).
With K draws and observed s_obs,

    τ_α   = s_(⌈(1−α)(K+1)⌉)                          right-tail critical value   (9)
    p     = (1 + #{k : s⁽ᵏ⁾ ≥ s_obs}) / (K + 1)                                   (10)
    s_cal = max( (s_obs − τ_α) / (s_max − τ_α), 0 )   max-preserving              (12)

**Aggregation-aware calibration.** When the reported statistic is an aggregate T (e.g. a
max over layer pairs), *the null must match the entire analysis pipeline*. The **same**
permutation π_k is applied to **all layers** of model B,

    S⁽ᵏ⁾_{ℓ,ℓ'} = s( X⁽ᴬ⁾_ℓ , π_k( Y⁽ᴮ⁾_{ℓ'} ) ),   T⁽ᵏ⁾ = T(S⁽ᵏ⁾)                (14)

then τ_α^agg, p^agg and T_cal follow (15)–(17) with T in place of s. For T = max this
recovers the classical **maxT** procedure of Westfall & Young (1993). Defaults: K = 200,
α = 0.05. This is what `_common.aristotelian_cross` implements.

The guarantee is **Assumption 3.1 (exchangeability under the null)**:

    P_H₀(X, Y) = P_H₀(X, π(Y))   for any π ∈ Πₙ

from which p is super-uniform under H₀ (Proposition C.4), giving finite-sample Type-I
control.

---

## 2. Why the i.i.d. instantiation is invalid here

The paper instantiates Πₙ as the **full symmetric group Sₙ** — correct for its data,
which are image datasets with genuinely exchangeable samples.

Our samples are **consecutive windows of one continuous recording**. Window 500 and
window 501 are six seconds apart in the same scene. Both X and Y are smooth functions of
time under *any* hypothesis. A free permutation destroys that autocorrelation, so
s(X, π(Y)) is drawn from a distribution with strictly less structure than the data.

**Assumption 3.1 fails**, and the resulting null is anti-conservative — it understates the
baseline and overstates significance. `_common.py`'s own docstring says exactly this.

Measured consequence (fMRI × vision, clip4s grid, W = 2700): observed mKNN **0.284**,
i.i.d. chance floor **0.0019**, but the far-tail circular-shift null **0.269**. The free
permutation would have declared an overwhelming effect where the honest excess is 5%.

---

## 3. The cyclic null

The paper's framework is stated over an **arbitrary group** Πₙ, not specifically Sₙ. So
the correction stays inside its own formalism: **change the group.**

> **H₀′** — X and Y have no relationship beyond their marginal statistics **and their
> individual temporal autocorrelation structure.**
>
> **Πₙ = Cₙ**, the cyclic group of circular shifts,
>
>     π_d(Y)[i] = Y[(i − d) mod n],   d ∈ {1, …, n−1}
>
> applied with the **same d to every layer** of model B (aggregation-aware, Eq. 14), then
> τ_α^agg, p^agg, T_cal exactly as in (15)–(17).

**Why the guarantees survive.** Randomisation inference requires only that the
transformations form a *group* under which the null distribution is invariant. Cₙ is a
group (closed, identity, inverses), and for a stationary series the distribution is
invariant to cyclic shift. Nothing in the super-uniformity proof uses Πₙ = Sₙ.

**Why it is strictly better here.** It corrects **both** confounders at once:

- *Temporal* — the null now contains the shared time axis, so what survives is
  content-specific structure rather than shared smoothness.
- *Depth* — and this is the decisive part. Under a free permutation every layer collapses
  to near-zero, so maxing over 48 near-zeros is still near-zero: measured i.i.d. nulls for
  fMRI–EEG CKA rose with depth (L=2 → 0.0023, L=12 → 0.0044, L=24 → 0.0066,
  L=48 → 0.0102) but sat ~100× below an observed 0.42, unable to offset anything. Under
  cyclic shift **every layer retains substantial similarity**, so the max over 48
  genuinely exceeds the max over 2 — the selection advantage lands in the same numerical
  regime as the observed value, where subtracting it is meaningful.

**Exhaustive, not sampled.** |Cₙ| = n − 1, small enough to enumerate every shift instead
of sampling K = 200 — an exact test with p-floor 1/n rather than 1/(K+1). Since mKNN and
CKA under rotation are both circular cross-correlations, one FFT yields all n shifts at
once (`shift_tests/fast_shift.py`).

---

## 4. Three caveats

**Stationarity is the price.** Cyclic shift assumes an approximately stationary series.
fMRI is not: low-frequency power exceeds high-frequency by 17–25× with no high-pass
filtering, and wrapping creates a seam where the end meets the beginning. Mitigate by
detrending first, or by using **block permutation** (shuffling contiguous blocks), which
preserves local autocorrelation without requiring global stationarity.

**Effective K ≪ n − 1.** Shifts d and d+1 give nearly identical nulls, so the draws are
heavily dependent; the effective count is roughly n/τ with τ the autocorrelation time. The
nominal 1/n floor overstates the real resolution. Standard remedy: restrict to a far tail,
|d| ≥ d_min (`shift_tests` uses `--tail-min 100`).

**Scores drop, and that is the point.** The cyclic null tests a strictly stronger
hypothesis than the i.i.d. one. Results that looked overwhelming will look modest. Report
both: the i.i.d. calibration answers *"is there any correspondence at all"*, the cyclic
one answers *"is it more than a shared time axis"*. They are nested hypotheses and neither
substitutes for the other.

---

## 5. Relevance to RESULTS.md

The paper's headline is that

> *"the apparent convergence reported by global spectral measures largely disappears after
> calibration, while local neighborhood similarity, but not local distances, retains
> significance"*

**Panel (c) uses linear CKA — a global spectral measure. Panel (b) uses mKNN — local
neighbourhood.** So the paper predicts panel (c) is the vulnerable one and panel (b) the
survivor. An i.i.d. calibration of panel (c) did *not* reproduce that collapse (the null
averaged 0.0048 against an observed 0.4189, changing ρ_adj by nothing), but their
inflation mechanism is the high-dimensional width effect, and the cyclic null tests a
different and stronger hypothesis. Running it is the informative experiment.
